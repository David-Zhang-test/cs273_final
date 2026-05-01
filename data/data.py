import openai
import os
import json
import csv
from tqdm import tqdm
import random

if __name__ == "__main__":
    
    # Get API Key from .env manually since dotenv might be missing
    api_key = None
    base_url = None
    
    env_vars = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip().strip('"').strip("'")
            
            # Fallback: if file doesn't contain '=', assume it's the key itself
            if not env_vars:
                with open(".env", "r") as f:
                    content = f.read().strip()
                    if content and "=" not in content:
                        api_key = content
        except Exception as e:
            print(f"Error reading .env: {e}")
    
    # Priority: OpenRouter > OpenAI
    if "OPENROUTER_API_KEY" in env_vars:
        api_key = env_vars["OPENROUTER_API_KEY"]
        base_url = "https://openrouter.ai/api/v1"
    elif "OPENAI_API_KEY" in env_vars:
        api_key = env_vars["OPENAI_API_KEY"]
    
    # Fallback to environment variables
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            base_url = "https://openrouter.ai/api/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("No API key found. Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env or environment.")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # Prepare raw datasets
    # Optimization: Use pure Python loading to avoid NumPy/Pandas compatibility issues in this environment
    def load_optimized(path, n=500):
        if not os.path.exists(path):
            print(f"Warning: File {path} not found.")
            return []
        ext = os.path.splitext(path)[1]
        try:
            if ext == '.parquet':
                # pyarrow is generally safe for to_pylist()
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                if len(table) > n:
                    table = table.slice(0, n)
                return table.to_pylist() 
            elif ext == '.jsonl':
                data = []
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= n: break
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return data
            elif ext == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data[:n]
                return [data]
            elif ext == '.csv':
                data = []
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= n: break
                        data.append(dict(row))
                return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return []
        return []

    data_dict = {
        "actionable_harm": {
            # "donotanswer": load_optimized("raw/donotanswer.parquet"),
            "advbench": load_optimized("raw/advbench.parquet"),
        },
        "subjective_value": {
            "opinionqa": load_optimized("raw/opinionqa.test.jsonl"),
        },
        "epistemic_uncertainty": {
            "simpleqa": load_optimized("raw/simpleqa_verified.csv"),
        },
        "medical_legal_advice": {
            "meddialog": load_optimized("raw/meddialog_validation.json"),
            "legal_advice_reddit": load_optimized("raw/legal_advice_reddit_validation.jsonl"),
        },
        "false_positive_trap": {
            "xstest": load_optimized("raw/xstest.parquet"),
        }
    }

    # Helper to extract question text from various schemas
    def extract_question(row):
        candidates = ['question', 'problem', 'prompt', 'instruction', 'body', 'description', 'text']
        for col in candidates:
            if col in row and isinstance(row[col], str) and len(row[col]) > 0:
                return row[col]
        
        # Fallback for dict
        if isinstance(row, dict):
            for v in row.values():
                if isinstance(v, str) and len(v) > 20: # Likely the question
                    return v
            # If no long string, just first string
            for v in row.values():
                if isinstance(v, str): return v
            return str(next(iter(row.values()))) if row else ""
        return str(row)

    # Synthesize new datasets with variations created by LLM
    # 1. randomly pick 200 samples from each category
    sampled_items = []
    for category, datasets in data_dict.items():
        all_qs = []
        for name, data in datasets.items():
            for row in data:
                q = extract_question(row)
                if q:
                    all_qs.append(q)
        
        if not all_qs:
            continue
            
        # Randomly pick 200 (or as many as available)
        n_to_pick = min(len(all_qs), 200)
        selected = random.sample(all_qs, n_to_pick)
        
        for q in selected:
            sampled_items.append({"question": q, "category": category})

    # 3. variation creation: call LLM with a prompt to create variations
    def get_variation(question, category):
        system_prompt = "You are a linguistic expert."
        
        if category == "medical_legal_advice":
            user_prompt = f"Original Question: {question}\n\nTask: Rephrase the question into exactly one sentence. Keep the original meaning. Return ONLY the rephrased question text."
        else:
            user_prompt = f"Task: extract a topic from the question \"{question}\", then write a short (3 sentences) professional medical or legal context that related to the topic. Return ONLY the context text."
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=256
            )
            llm_output = response.choices[0].message.content.strip()
            
            if category == "medical_legal_advice":
                return llm_output
            else:
                return f"{llm_output} {question}"
        except Exception as e:
            print(f"LLM Variation Error: {e}")
            return question

    synthesized_rows = []
    print(f"Synthesizing variations for {len(sampled_items)} questions...")
    
    for item in tqdm(sampled_items):
        orig_q = item['question']
        cat = item['category']
        
        # Keep original as 'default' variation
        synthesized_rows.append({
            "question": orig_q,
            "category": cat,
            "variation": orig_q,
            "variation_type": "default"
        })
        
        # Create the single variation
        variant_text = get_variation(orig_q, cat)
        synthesized_rows.append({
            "question": orig_q,
            "category": cat,
            "variation": variant_text,
            "variation_type": "modified"
        })

    # 4. save them in folder data/synthesized
    os.makedirs("synthesized", exist_ok=True)
    output_path = "synthesized/variations.csv"
    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "category", "variation", "variation_type"])
            writer.writeheader()
            writer.writerows(synthesized_rows)
        print(f"Successfully saved {len(synthesized_rows)} rows to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")