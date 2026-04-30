import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def generate_full_rq2_plot():
    path = '/home/davidzhang/dev/nlp/cs273_final/saved_results/sae_sweep/sweep_summary.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    layers = data['selected_layers']
    
    # All 8 categories from the taxonomy
    categories = [
        "Full Compliance", 
        "Caveat / Disclaimer Compliance",
        "Preach / Lecturing Compliance",
        "Both-Sides Neutrality", 
        "The Pivot / Deflection",
        "Theory-Only (Partial Refusal)", 
        "Polite Full Refusal", 
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
        # Use (Polite Refusal - Full Compliance) to find the most 'refusal-discriminant' node at this layer
        if "Polite Full Refusal" in means and "Full Compliance" in means:
            refusal_means = np.array(means["Polite Full Refusal"])
            compliance_means = np.array(means["Full Compliance"])
            deltas = refusal_means - compliance_means
            best_idx_in_top20 = np.argmax(deltas)
        else:
            best_idx_in_top20 = 0 # Fallback
        
        for cat in categories:
            # Handle potential missing categories in the mean activations
            if cat in means:
                results[cat].append(means[cat][best_idx_in_top20])
            else:
                results[cat].append(0.0)
            
        # 2. Core Backbone Tracking
        node_activations = []
        # Measure core activation during Polite Refusal (the most standard refusal)
        refusal_means_all = means.get("Polite Full Refusal", [0]*len(indices))
        for node_id in core_nodes:
            if node_id in indices:
                idx = indices.index(node_id)
                node_activations.append(refusal_means_all[idx])
            else:
                node_activations.append(0.0)
        core_values.append(np.mean(node_activations))

    # Plotting
    plt.figure(figsize=(13, 8))
    
    # Use a distinct color palette for the 8 categories
    palette = sns.color_palette("husl", len(categories))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for i, cat in enumerate(categories):
        plt.plot(layers, results[cat], label=cat, marker=markers[i], linewidth=2, alpha=0.8, color=palette[i])

    # Core Backbone line (Thick, black, dashed)
    plt.plot(layers, core_values, label="Shared Core Backbone", color='black', linewidth=4, marker='*', markersize=14, linestyle='--', zorder=10)

    plt.title('RQ2: Full Behavioral Spectrum vs. Shared Core Backbone', fontsize=18, fontweight='bold')
    plt.xlabel('Model Layer', fontsize=14)
    plt.ylabel('Mean Activation', fontsize=14)
    
    # Adjust legend to be outside the plot
    plt.legend(title="Circuit / Behavior", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('/home/davidzhang/dev/nlp/cs273_final/saved_results/RQ2_Refusal_Signal_Sharpening.png', dpi=300)
    print("Saved full 8-category hybrid RQ2 plot to saved_results/RQ2_Refusal_Signal_Sharpening.png")

if __name__ == "__main__":
    generate_full_rq2_plot()
