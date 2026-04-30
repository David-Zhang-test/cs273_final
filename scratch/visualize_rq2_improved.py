import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def generate_improved_rq2_plot():
    path = '/home/davidzhang/dev/nlp/cs273_final/saved_results/sae_sweep/sweep_summary.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    layers = data['selected_layers']
    categories = [
        "Full Compliance", 
        "Polite Full Refusal", 
        "Both-Sides Neutrality", 
        "Theory-Only (Partial Refusal)",
        "Epistemic Refusal"
    ]
    
    results = {cat: [] for cat in categories}
    selected_node_info = []

    for summary in data['layer_summaries']:
        layer = summary['layer']
        indices = summary['top_feature_indices']
        means = summary['group_mean_top_features']
        
        # Filter out invalid categories (like the one with a typo '4. ...')
        valid_means = {k: v for k, v in means.items() if k in categories}
        
        # Find the node that most strongly distinguishes Refusal from Compliance
        # We'll use (Polite Refusal - Full Compliance) as the metric
        refusal_means = np.array(means["Polite Full Refusal"])
        compliance_means = np.array(means["Full Compliance"])
        deltas = refusal_means - compliance_means
        
        best_idx_in_top20 = np.argmax(deltas)
        best_node_id = indices[best_idx_in_top20]
        
        selected_node_info.append(f"L{layer}: Node {best_node_id}")
        
        for cat in categories:
            results[cat].append(means[cat][best_idx_in_top20])

    # Plotting
    plt.figure(figsize=(12, 7))
    
    # Use different markers and line styles
    markers = ['o', 's', '^', 'D', 'v']
    for i, cat in enumerate(categories):
        plt.plot(layers, results[cat], label=cat, marker=markers[i], linewidth=2.5, markersize=8)

    plt.title('RQ2: The Emergence and Sharpening of Refusal Features', fontsize=16, fontweight='bold')
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Mean Activation of Top "Refusal-Discriminant" Node', fontsize=12)
    
    # Annotate the node IDs
    for i, txt in enumerate(selected_node_info):
        plt.annotate(txt, (layers[i], results["Polite Full Refusal"][i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.7)

    plt.legend(title="Behavior Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('/home/davidzhang/dev/nlp/cs273_final/saved_results/RQ2_Refusal_Signal_Sharpening.png', dpi=300)
    print("Saved improved RQ2 plot to saved_results/RQ2_Refusal_Signal_Sharpening.png")

import numpy as np
if __name__ == "__main__":
    generate_improved_rq2_plot()
