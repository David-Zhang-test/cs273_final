import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_feature_heatmap(sparse_acts, labels, top_features):
    """
    Plots a heatmap showing how top features activate across different behavior labels.
    sparse_acts: Tensor or array of sparse SAE activations.
    labels: List or array of labels (1-8).
    top_features: List of feature indices to display.
    """
    # Organize data: calculate mean activation of top features for each specific label (1-8)
    heatmap_data = np.zeros((8, len(top_features)))
    labels_arr = np.array(labels)
    
    for label_idx in range(1, 9):
        mask = (labels_arr == label_idx)
        if mask.sum() > 0:
            mean_acts = sparse_acts[mask].mean(dim=0)
            # Assuming mean_acts is a PyTorch tensor, we move to cpu. If numpy, handle it correctly.
            mean_acts_np = mean_acts[top_features].cpu().numpy() if hasattr(mean_acts, 'cpu') else mean_acts[top_features]
            heatmap_data[label_idx-1, :] = mean_acts_np
            
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, xticklabels=top_features, yticklabels=range(1, 9), cmap="YlOrRd")
    plt.xlabel("SAE Feature Index")
    plt.ylabel("Behavioral Taxonomy Label (1-8)")
    plt.title("Feature Activation Strength by Refusal Type")
    plt.show()
