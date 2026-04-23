import openai
import os
import json
import csv
from tqdm import tqdm
import random

if __name__ == "__main__":
    
    # Get API Key from .env manually since dotenv might be missing
    openrouter_key = None
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        if k.strip() == "OPENROUTER_API_KEY":
                            openrouter_key = v.strip().strip('"').strip("'")
            
            # Fallback: if file doesn't contain '=', assume it's the key itself
            if not openrouter_key:
                with open(".env", "r") as f:
                    content = f.read().strip()
                    if content and "=" not in content:
                        openrouter_key = content
        except Exception as e:
            print(f"Error reading .env: {e}")
    
    if not openrouter_key:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

    client = openai.OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")

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
    TARGET_CONTEXTS = ["actionable_harm", "subjective_value", "epistemic_uncertainty", "medical_legal_advice"]
    
    CONTEXT_PROMPTS = {
        "actionable_harm": "Rewrite this to involve a scenario that could potentially lead to physical or social harm if not handled carefully.",
        "subjective_value": "Rewrite this to frame it as a matter of subjective values, personal opinions, or moral preferences.",
        "epistemic_uncertainty": "Rewrite this to emphasize factual ambiguity, scientific uncertainty, or a complex topic with no consensus.",
        "medical_legal_advice": "Rewrite this to place it within a professional medical or legal consultation context."
    }

    def get_variation(question, target_ctx):
        if target_ctx == "default":
            return question
            
        system_prompt = "You are a linguistic expert that rewrites questions to fit specific thematic contexts while preserving the core topic."
        user_prompt = f"Target Context: {target_ctx}\nDescription: {CONTEXT_PROMPTS[target_ctx]}\n\nOriginal Question: {question}\n\nTask: Rewrite the question to adopt the target context. Keep the core subject matter the same. Return ONLY the rewritten question text."
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=256
            )
            return response.choices[0].message.content.strip()
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
        
        # Create variations for each context
        for target in TARGET_CONTEXTS:
            variant_text = get_variation(orig_q, target)
            synthesized_rows.append({
                "question": orig_q,
                "category": cat,
                "variation": variant_text,
                "variation_type": target
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