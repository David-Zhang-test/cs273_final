"""
Feature Interpretation Script
==============================
Analyzes what top SAE features encode by examining activations and patterns.

Key Steps:
1. Load layer 15 SAE features and activation statistics
2. For each top feature, find which judge_categories activate it most
3. Build feature usage profiles (which category uses which feature)
4. Compare patterns across layers to understand signal vs decision
5. Generate interpretability report

Run: python src/sae_analysis/feature_interpreter.py \
       --layer 15 \
       --input_csv saved_results/judge/variations_judge.csv \
       --states_dir saved_states \
       --output_dir saved_results/feature_interpretation
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from sae_lens import SAE
except ImportError:
    SAE = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAE_RELEASE = "llama-3.1-8b-instruct-andyrdt"
DEFAULT_TRAINER = "trainer_0"


@dataclass
class InterpreterConfig:
    layer: int
    input_csv: str
    states_dir: str
    output_dir: str
    release: str
    trainer: str
    device: str
    top_k: int  # How many top features to analyze


def load_sweep_results(layer: int, sweep_dir: str = "saved_results/sae_sweep") -> Dict:
    """Load sweep summary JSON for a layer."""
    json_path = Path(sweep_dir) / f"layer_{layer:02d}_summary.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Sweep results not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def load_activations(states_dir: str, row_indices: List[int], layer: int) -> torch.Tensor:
    """Load SAE-encoded activations for specific rows."""
    activations = []
    
    for row_idx in row_indices:
        state_path = Path(states_dir) / f"prompt_{row_idx:06d}.pt"
        if not state_path.exists():
            logger.warning(f"State file not found: {state_path}")
            continue
        
        try:
            state = torch.load(state_path, map_location="cpu")
            activations.append(state)
        except Exception as e:
            logger.warning(f"Failed to load {state_path}: {e}")
            continue
    
    if not activations:
        raise ValueError("No activation files found!")
    
    return torch.stack(activations, dim=0)


def load_sae_for_layer(layer: int, release: str, trainer: str, device: str) -> Tuple[str, SAE]:
    """Load SAE for layer."""
    if SAE is None:
        raise ImportError("SAE Lens not installed")
    
    sae_id = f"resid_post_layer_{layer}_{trainer}"
    logger.info(f"Loading SAE {sae_id}...")
    sae = SAE.load_from_pretrained(release, sae_id)
    sae.to(device)
    sae.eval()
    
    return sae_id, sae


def get_judge_categories(input_csv: str) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Load judge categories and map rows."""
    category_to_rows = defaultdict(list)
    row_to_category = {}
    
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            category = row.get("judge_category", "unknown")
            category_to_rows[category].append(row_idx)
            row_to_category[row_idx] = category
    
    return dict(category_to_rows), row_to_category


def analyze_feature_activations(
    sae: SAE,
    feature_indices: List[int],
    activations: torch.Tensor,
    category_to_rows: Dict[str, List[int]],
    row_to_category: Dict[int, str],
    device: str,
) -> Dict:
    """
    Analyze how features activate across judge categories.
    
    Args:
        sae: SAE model (used just for reference)
        feature_indices: Top features from sweep
        activations: (N, D) activation tensor
        category_to_rows: Maps category -> row indices
        row_to_category: Maps row index -> category
        device: torch device
    
    Returns:
        Dict with feature analysis per category
    """
    sae_acts = activations.to(device)
    
    # Build index mapping from full activation data
    all_rows = list(range(len(activations)))
    
    feature_analysis = {}
    
    for feat_idx in feature_indices:
        feat_acts = sae_acts[:, feat_idx].cpu().numpy()  # Activations for this feature
        
        # Per-category statistics
        category_stats = {}
        
        for category, row_indices in category_to_rows.items():
            # Find which positions in activations correspond to this category
            mask = np.array([i in row_indices for i in all_rows])
            
            if mask.sum() == 0:
                continue
            
            cat_activations = feat_acts[mask]
            
            category_stats[category] = {
                "mean_activation": float(cat_activations.mean()),
                "std_activation": float(cat_activations.std()),
                "max_activation": float(cat_activations.max()),
                "activation_rate": float((cat_activations > 0).mean()),  # Sparsity
                "num_samples": int(mask.sum()),
            }
        
        # Rank categories by mean activation
        ranked = sorted(category_stats.items(), key=lambda x: x[1]["mean_activation"], reverse=True)
        
        feature_analysis[int(feat_idx)] = {
            "feature_index": int(feat_idx),
            "category_stats": dict(category_stats),
            "top_category": ranked[0][0] if ranked else None,
            "top_category_activation": ranked[0][1]["mean_activation"] if ranked else 0.0,
        }
    
    return feature_analysis


def build_feature_profile(feature_analysis: Dict, category_to_rows: Dict[str, List[int]]) -> Dict:
    """Build which features are most important per category."""
    category_features = defaultdict(list)
    
    for feat_idx, analysis in feature_analysis.items():
        for category, stats in analysis["category_stats"].items():
            activation_strength = stats["mean_activation"]
            category_features[category].append({
                "feature_idx": feat_idx,
                "mean_activation": activation_strength,
                "activation_rate": stats["activation_rate"],
            })
    
    # Sort features per category by mean activation
    for category in category_features:
        category_features[category].sort(key=lambda x: x["mean_activation"], reverse=True)
    
    return dict(category_features)


def run_feature_interpretation(config: InterpreterConfig) -> None:
    """Main feature interpretation pipeline."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Layer {config.layer} Feature Interpretation")
    logger.info(f"Output: {output_dir}")
    
    # Step 1: Load sweep results
    sweep_data = load_sweep_results(config.layer)
    top_feature_indices = sweep_data.get("top_feature_indices", [])[:config.top_k]
    logger.info(f"Analyzing top {len(top_feature_indices)} features")
    
    # Step 2: Load judge categories
    category_to_rows, row_to_category = get_judge_categories(config.input_csv)
    logger.info(f"Found {len(category_to_rows)} judge categories")
    
    # Step 3: Load SAE
    sae_id, sae = load_sae_for_layer(config.layer, config.release, config.trainer, config.device)
    
    # Step 4: Load activations for all rows
    all_row_indices = list(range(len(category_to_rows)))  # Will load from state files
    # Get all row indices from category mapping
    all_row_indices = []
    for rows in category_to_rows.values():
        all_row_indices.extend(rows)
    all_row_indices = sorted(set(all_row_indices))
    
    logger.info(f"Loading activations for {len(all_row_indices)} samples...")
    
    try:
        activations = load_activations(config.states_dir, all_row_indices, config.layer)
    except ValueError as e:
        logger.error(f"Failed to load activations: {e}")
        return
    
    # Step 5: Encode activations through SAE
    logger.info("Encoding activations through SAE...")
    sae.to(config.device)
    with torch.no_grad():
        acts_input = activations.to(config.device)
        sae_feature_acts = sae.encode(acts_input)
    
    logger.info(f"SAE feature shape: {sae_feature_acts.shape}")
    
    # Step 6: Analyze features
    logger.info("Analyzing feature activations per category...")
    feature_analysis = analyze_feature_activations(
        sae,
        top_feature_indices,
        sae_feature_acts,
        category_to_rows,
        row_to_category,
        config.device,
    )
    
    # Step 7: Build category profiles
    logger.info("Building category feature profiles...")
    category_features = build_feature_profile(feature_analysis, category_to_rows)
    
    # Step 8: Save detailed results
    logger.info("Saving results...")
    
    with open(output_dir / "feature_analysis.json", "w", encoding="utf-8") as f:
        # Convert numpy types for JSON serialization
        serializable = {}
        for feat_idx, analysis in feature_analysis.items():
            serializable[str(feat_idx)] = {
                "feature_index": analysis["feature_index"],
                "category_stats": {
                    cat: {
                        "mean_activation": float(s["mean_activation"]),
                        "std_activation": float(s["std_activation"]),
                        "max_activation": float(s["max_activation"]),
                        "activation_rate": float(s["activation_rate"]),
                        "num_samples": int(s["num_samples"]),
                    }
                    for cat, s in analysis["category_stats"].items()
                },
                "top_category": analysis["top_category"],
                "top_category_activation": float(analysis["top_category_activation"]),
            }
        json.dump(serializable, f, indent=2)
    
    # Save category profiles
    with open(output_dir / "category_feature_profiles.json", "w", encoding="utf-8") as f:
        serializable_profiles = {}
        for cat, features in category_features.items():
            serializable_profiles[cat] = [
                {
                    "feature_idx": int(f["feature_idx"]),
                    "mean_activation": float(f["mean_activation"]),
                    "activation_rate": float(f["activation_rate"]),
                }
                for f in features
            ]
        json.dump(serializable_profiles, f, indent=2)
    
    # Generate summary report
    summary = {
        "layer": config.layer,
        "sae_id": sae_id,
        "num_features_analyzed": len(top_feature_indices),
        "num_categories": len(category_to_rows),
        "categories": list(category_to_rows.keys()),
        "total_samples": sum(len(rows) for rows in category_to_rows.values()),
        "key_findings": {
            "features_per_category": {
                cat: len(features) for cat, features in category_features.items()
            },
            "top_discriminative_features": [
                {
                    "feature_idx": int(feat_idx),
                    "top_category": analysis["top_category"],
                    "activation_strength": float(analysis["top_category_activation"]),
                }
                for feat_idx, analysis in sorted(
                    feature_analysis.items(),
                    key=lambda x: x[1]["top_category_activation"],
                    reverse=True,
                )[:10]
            ],
        },
    }
    
    with open(output_dir / "interpretation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Feature interpretation complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  - feature_analysis.json: Per-feature breakdown")
    logger.info(f"  - category_feature_profiles.json: Which features per category")
    logger.info(f"  - interpretation_summary.json: High-level findings")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Interpret what SAE features encode.")
    parser.add_argument("--layer", type=int, default=15, help="Layer to interpret")
    parser.add_argument("--input_csv", type=str, default="saved_results/judge/variations_judge.csv")
    parser.add_argument("--states_dir", type=str, default="saved_states")
    parser.add_argument("--output_dir", type=str, default="saved_results/feature_interpretation")
    parser.add_argument("--release", type=str, default=DEFAULT_SAE_RELEASE)
    parser.add_argument("--trainer", type=str, default=DEFAULT_TRAINER)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=20, help="Top K features to interpret")
    
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    config = InterpreterConfig(
        layer=args.layer,
        input_csv=args.input_csv,
        states_dir=args.states_dir,
        output_dir=args.output_dir,
        release=args.release,
        trainer=args.trainer,
        device=args.device,
        top_k=args.top_k,
    )
    
    run_feature_interpretation(config)


if __name__ == "__main__":
    main()
