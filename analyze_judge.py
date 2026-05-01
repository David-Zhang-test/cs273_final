import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def parse_judge_response(response_str):
    try:
        data = json.loads(response_str)
        if "top_3" in data:
            return data["top_3"]
        return None
    except:
        return None

def normalize_taxonomy(name):
    # Mapping to standard names to handle slight variations from LLM judge
    name_lower = name.lower()
    if "full compliance" in name_lower:
        return "Full Compliance"
    elif "caveat" in name_lower or "disclaimer" in name_lower:
        return "Caveat / Disclaimer Compliance"
    elif "preach" in name_lower or "lecturing" in name_lower:
        return "Preach / Lecturing Compliance"
    elif "both-sides" in name_lower or "neutrality" in name_lower:
        return "Both-Sides Neutrality"
    elif "pivot" in name_lower or "deflection" in name_lower:
        return "The Pivot / Deflection"
    elif "theory" in name_lower or "partial refusal" in name_lower:
        return "Theory-Only (Partial Refusal)"
    elif "polite full refusal" in name_lower or ("polite" in name_lower and "refusal" in name_lower):
        return "Polite Full Refusal"
    elif "epistemic refusal" in name_lower or "epistemic" in name_lower:
        return "Epistemic Refusal"
    # Exact match if possible
    return name

taxonomy_order = [
    "Full Compliance",
    "Caveat / Disclaimer Compliance",
    "Preach / Lecturing Compliance",
    "Both-Sides Neutrality",
    "The Pivot / Deflection",
    "Theory-Only (Partial Refusal)",
    "Polite Full Refusal",
    "Epistemic Refusal"
]

def main():
    csv_path = "results/variations_judge.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Initialize data structures for Pic 1 and Pic 2
    categories = df['category'].unique().tolist()
    variation_types = df['variation_type'].unique().tolist()

    # Data for Pic 1: weighted sum per category + total
    cat_tax_weights = {cat: {tax: 0.0 for tax in taxonomy_order} for cat in categories}
    total_tax_weights = {tax: 0.0 for tax in taxonomy_order}

    # Data for Pic 2: weighted sum per variation_type
    var_tax_weights = {var: {tax: 0.0 for tax in taxonomy_order} for var in variation_types}

    for index, row in df.iterrows():
        judge_resp = row['judge_response']
        if pd.isna(judge_resp):
            continue
        
        top_3 = parse_judge_response(judge_resp)
        if top_3 is None:
            continue
        
        cat = row['category']
        var = row['variation_type']
        
        for item in top_3:
            tax_name = item.get("category", "")
            prob = item.get("probability", 0.0)
            
            norm_tax = normalize_taxonomy(tax_name)
            if norm_tax in taxonomy_order:
                # Add to Pic 1 data
                if cat in cat_tax_weights:
                    cat_tax_weights[cat][norm_tax] += prob
                total_tax_weights[norm_tax] += prob
                
                # Add to Pic 2 data
                if var in var_tax_weights:
                    var_tax_weights[var][norm_tax] += prob

    # Plot Pic 1: 6 subpictures (5 categories + 1 total)
    fig1, axes1 = plt.subplots(2, 3, figsize=(24, 12))
    axes1 = axes1.flatten()
    
    all_subplots_data = [(cat, cat_tax_weights[cat]) for cat in categories]
    all_subplots_data.append(("Total", total_tax_weights))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (title, data) in enumerate(all_subplots_data):
        ax = axes1[i]
        values = [data[tax] for tax in taxonomy_order]
        ax.barh(taxonomy_order, values, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Weighted Sum of Probability")
        ax.invert_yaxis() # To show the first taxonomy at the top
        
    plt.tight_layout()
    fig1.savefig("results/pic1_categories_taxonomy.png", dpi=300, bbox_inches='tight')
    print("Saved Pic 1 to results/pic1_categories_taxonomy.png")
    
    # Plot Pic 2: Heatmap for variation type vs taxonomy
    heatmap_data = []
    for var in variation_types:
        row_data = [var_tax_weights[var][tax] for tax in taxonomy_order]
        row_sum = sum(row_data)
        if row_sum > 0:
            row_data = [x / row_sum * 100.0 for x in row_data]
        heatmap_data.append(row_data)
        
    heatmap_df = pd.DataFrame(heatmap_data, index=variation_types, columns=taxonomy_order)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
    ax2.set_title("Heatmap of Variation Type vs Abstention Taxonomy")
    ax2.set_xlabel("Abstention Taxonomy")
    ax2.set_ylabel("Variation Type")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    fig2.savefig("results/pic2_variation_heatmap.png", dpi=300, bbox_inches='tight')
    print("Saved Pic 2 to results/pic2_variation_heatmap.png")

if __name__ == "__main__":
    main()
