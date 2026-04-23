import torch
import numpy as np

class FeatureProber:
    def __init__(self, sae_model):
        """
        Initializes the prober with a loaded SAE model from sae_lens.
        """
        self.sae = sae_model

    def get_sparse_features(self, dense_activation):
        """Pass dense activations through the SAE encoder."""
        with torch.no_grad():
            # SAE returns feature activations and the reconstructed dense vector
            feature_acts = self.sae.encode(dense_activation)
        return feature_acts

    def analyze_features(self, all_activations, labels):
        """
        Compares feature firing rates between Compliance (1-3) and Refusal (4-8).
        all_activations: Tensor of shape [num_prompts, d_model]
        labels: List of ints
        """
        sparse_acts = self.get_sparse_features(all_activations) # [num_prompts, d_sae]
        
        # Create masks based on our LLM Judge labels
        labels = np.array(labels)
        compliance_mask = (labels >= 1) & (labels <= 3)
        refusal_mask = (labels >= 4) & (labels <= 8)
        
        compliance_mean = sparse_acts[compliance_mask].mean(dim=0)
        refusal_mean = sparse_acts[refusal_mask].mean(dim=0)
        
        # Find features that fire heavily for refusal but not compliance
        feature_diff = refusal_mean - compliance_mean
        top_refusal_features = torch.topk(feature_diff, k=10).indices.tolist()
        
        print(f"Top 10 Latents driving Refusal: {top_refusal_features}")
        return top_refusal_features, sparse_acts
