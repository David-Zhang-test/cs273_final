import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def generate_architectural_rq2_plot():
    path = '/home/davidzhang/dev/nlp/cs273_final/saved_results/sae_sweep/sweep_summary.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    layers = data['selected_layers']
    
    # Define node groups from Conclusions.md
    node_groups = {
        "Shared Core Backbone": [44833, 80948, 106628],
        "Compliance-Specific": [129304, 32672, 5506],
        "Refusal-Specific (Polite)": [45181, 78807, 40311],
        "Epistemic-Specific": [40311, 42366, 82391]
    }
    
    # Mapping group name to the category we should measure activation on
    # (e.g. for Compliance-Specific nodes, we care about their activation during Compliance)
    group_to_cat = {
        "Shared Core Backbone": "Polite Full Refusal", # Core is most active during refusal
        "Compliance-Specific": "Full Compliance",
        "Refusal-Specific (Polite)": "Polite Full Refusal",
        "Epistemic-Specific": "Epistemic Refusal"
    }

    plot_data = {group: [] for group in node_groups}

    for summary in data['layer_summaries']:
        indices = summary['top_feature_indices']
        means = summary['group_mean_top_features']
        
        for group_name, nodes in node_groups.items():
            target_cat = group_to_cat[group_name]
            cat_means = means.get(target_cat, [0]*len(indices))
            
            # Find activation for each node in this layer
            node_activations = []
            for node_id in nodes:
                if node_id in indices:
                    idx = indices.index(node_id)
                    node_activations.append(cat_means[idx])
                else:
                    node_activations.append(0.0) # Assume 0 if not in top 20 for this layer
            
            plot_data[group_name].append(np.mean(node_activations))

    # Plotting
    plt.figure(figsize=(12, 7))
    
    colors = ['#2c3e50', '#27ae60', '#c0392b', '#8e44ad']
    markers = ['o', 's', 'D', '^']
    
    for i, (group_name, values) in enumerate(plot_data.items()):
        plt.plot(layers, values, label=group_name, marker=markers[i], color=colors[i], linewidth=3, markersize=10)

    plt.title('RQ2: The Architectural Flow of Abstention Behaviors', fontsize=18, fontweight='bold')
    plt.xlabel('Model Layer', fontsize=14)
    plt.ylabel('Mean Activation of Defined Nodes', fontsize=14)
    
    # Add vertical lines for key stages
    plt.axvline(x=15, linestyle='--', color='gray', alpha=0.5)
    plt.text(15.5, 0.5, 'Geometric Organization\n(Compliance Peaking)', fontsize=10, color='gray')
    
    plt.axvline(x=27, linestyle='--', color='gray', alpha=0.5)
    plt.text(25.5, 0.5, 'Readout Sharpening\n(Refusal Specialization)', fontsize=10, color='gray', ha='right')

    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('/home/davidzhang/dev/nlp/cs273_final/saved_results/RQ2_Architectural_Flow.png', dpi=300)
    print("Saved architectural RQ2 plot to saved_results/RQ2_Architectural_Flow.png")

if __name__ == "__main__":
    generate_architectural_rq2_plot()
