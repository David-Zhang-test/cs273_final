import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def generate_rq2_plot():
    df = pd.read_csv('/home/davidzhang/dev/nlp/cs273_final/saved_results/sae_sweep/layer_sweep_summary.csv')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Representational Distance (Geometry)', color=color)
    ax1.plot(df['layer'], df['representational_distance'], marker='o', color=color, linewidth=3, label='Rep. Distance')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('F-Statistic (Decision Sharpening)', color=color)
    ax2.plot(df['layer'], df['f_statistic'], marker='s', color=color, linewidth=3, label='F-Statistic')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('RQ2: The Timeline of LLM Abstention Decisions', fontsize=16, fontweight='bold')
    
    # Annotate key points
    ax1.annotate('Peak Organization (Layer 15)', xy=(15, df[df['layer']==15]['representational_distance'].iloc[0]), 
                 xytext=(10, 145), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    ax2.annotate('Decision Readout (Layer 27)', xy=(27, df[df['layer']==27]['f_statistic'].iloc[0]), 
                 xytext=(20, 98), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    fig.tight_layout()
    plt.savefig('/home/davidzhang/dev/nlp/cs273_final/saved_results/RQ2_Decision_Timeline.png', dpi=300)
    print("Saved RQ2 plot to saved_results/RQ2_Decision_Timeline.png")

def generate_rq3_plot():
    path = '/home/davidzhang/dev/nlp/cs273_final/saved_results/feature_interpretation_27/category_feature_profiles.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Features of interest (Core + Specialized)
    features_to_show = [44833, 80948, 106628, 9897, 76568, 40311, 24431, 6326, 45181]
    categories = list(data.keys())
    
    heatmap_data = []
    for cat in categories:
        row = []
        cat_features = {f['feature_idx']: f['mean_activation'] for f in data[cat]}
        for f_idx in features_to_show:
            row.append(cat_features.get(f_idx, 0))
        heatmap_data.append(row)
    
    # Use only the node number as the column label
    heatmap_df = pd.DataFrame(heatmap_data, index=categories, columns=[str(f) for f in features_to_show])
    
    # Reorder categories for better visual flow
    category_order = [
        "Full Compliance", 
        "Caveat / Disclaimer Compliance", 
        "Preach / Lecturing Compliance", 
        "Both-Sides Neutrality", 
        "The Pivot / Deflection", 
        "Theory-Only (Partial Refusal)", 
        "Polite Full Refusal", 
        "Epistemic Refusal"
    ]
    heatmap_df = heatmap_df.reindex(category_order)

    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_df, annot=True, cmap="YlOrRd", fmt=".1f", cbar_kws={'label': 'Mean Activation'})
    
    plt.title('RQ3: Internal Overlap - Shared Backbone vs. Specialized Nodes (Layer 27)', fontsize=16, fontweight='bold')
    plt.xlabel('SAE Latent Features (Nodes)', fontsize=12)
    plt.ylabel('Refusal Category', fontsize=12)
    
    # Add grouping labels
    plt.text(1.5, -0.5, 'Shared Backbone (Core)', ha='center', va='center', fontweight='bold', color='darkred')
    plt.text(5.5, -0.5, 'Specialized / Style Nodes', ha='center', va='center', fontweight='bold', color='darkblue')

    plt.tight_layout()
    plt.savefig('/home/davidzhang/dev/nlp/cs273_final/saved_results/RQ3_Internal_Overlap.png', dpi=300)
    print("Saved RQ3 plot to saved_results/RQ3_Internal_Overlap.png")

if __name__ == "__main__":
    generate_rq2_plot()
    generate_rq3_plot()
