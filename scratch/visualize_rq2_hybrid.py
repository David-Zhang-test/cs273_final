import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def generate_hybrid_rq2_plot():
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
    
    # Core nodes to track (from Conclusions.md)
    core_nodes = [44833, 80948, 106628]
    
    results = {cat: [] for cat in categories}
    core_values = []

    for summary in data['layer_summaries']:
        indices = summary['top_feature_indices']
        means = summary['group_mean_top_features']
        
        # 1. Best Refusal Node Tracking (Original logic)
        refusal_means = np.array(means["Polite Full Refusal"])
        compliance_means = np.array(means["Full Compliance"])
        deltas = refusal_means - compliance_means
        
        best_idx_in_top20 = np.argmax(deltas)
        
        for cat in categories:
            results[cat].append(means[cat][best_idx_in_top20])
            
        # 2. Core Backbone Tracking (New logic)
        node_activations = []
        # Measure core activation during Polite Refusal
        refusal_means_all = means["Polite Full Refusal"]
        for node_id in core_nodes:
            if node_id in indices:
                idx = indices.index(node_id)
                node_activations.append(refusal_means_all[idx])
            else:
                node_activations.append(0.0)
        core_values.append(np.mean(node_activations))

    # Plotting
    plt.figure(figsize=(12, 7))
    
    # Original categories
    markers = ['o', 's', '^', 'D', 'v']
    for i, cat in enumerate(categories):
        plt.plot(layers, results[cat], label=cat, marker=markers[i], linewidth=2, alpha=0.7)

    # Core Backbone line (Thicker, distinct color)
    plt.plot(layers, core_values, label="Shared Core Backbone", color='black', linewidth=4, marker='*', markersize=12, linestyle='--')

    plt.title('RQ2: The Emergence of Refusal Circuits vs. Behavioral Choice', fontsize=16, fontweight='bold')
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Mean Activation', fontsize=12)
    
    # No annotations as requested
    
    plt.legend(title="Circuit / Behavior", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('/home/davidzhang/dev/nlp/cs273_final/saved_results/RQ2_Refusal_Signal_Sharpening.png', dpi=300)
    print("Saved updated hybrid RQ2 plot to saved_results/RQ2_Refusal_Signal_Sharpening.png")

if __name__ == "__main__":
    generate_hybrid_rq2_plot()
