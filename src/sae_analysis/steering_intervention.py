"""
Steering Intervention Script
==============================
Manipulates layer 15 SAE features to test causal control of refusal behavior.

Tests whether scaling/zeroing specific judge_category features causes
the model to output different refusal types.

Key Steps:
1. Load layer 15 SAE and activations for judge categories
2. Extract top features per category from sweep results
3. Create steered versions (scale/zero features)
4. Measure effect on judge_category predictions

Run: python src/sae_analysis/steering_intervention.py \
       --layer 15 \
       --input_csv saved_results/judge/variations_judge.csv \
       --states_dir saved_states \
       --output_dir saved_results/steering_results
"""

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class SteeringConfig:
    layer: int
    input_csv: str
    states_dir: str
    output_dir: str
    seed: int
    release: str
    trainer: str
    device: str
    num_samples: int  # How many samples to test steering on


def load_sweep_features(layer: int, sweep_dir: str = "saved_results/sae_sweep") -> Dict[str, List[int]]:
    """Extract top feature indices per judge_category from sweep results."""
    json_path = Path(sweep_dir) / f"layer_{layer:02d}_summary.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Sweep results not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    top_feature_indices = data.get("top_feature_indices", [])
    logger.info(f"Loaded {len(top_feature_indices)} top features for layer {layer}")
    
    return {"top_features": top_feature_indices}


def load_activations(states_dir: str, row_indices: List[int]) -> torch.Tensor:
    """Load SAE activation states for specific rows."""
    activations = []
    for row_idx in row_indices:
        state_path = Path(states_dir) / f"prompt_{row_idx:06d}.pt"
        if not state_path.exists():
            logger.warning(f"State file not found: {state_path}")
            continue
        
        state = torch.load(state_path, map_location="cpu")
        # Extract residual stream activation for the layer
        activations.append(state)
    
    if not activations:
        raise ValueError("No activation files found!")
    
    return torch.stack(activations, dim=0)


def load_sae_for_layer(layer: int, release: str, trainer: str, device: str) -> Tuple[str, SAE]:
    """Load SAE for a specific layer."""
    if SAE is None:
        raise ImportError("SAE Lens not installed")
    
    sae_id = f"resid_post_layer_{layer}_{trainer}"
    logger.info(f"Loading SAE {sae_id} from {release}...")
    
    # Use SAE.from_pretrained (not load_from_pretrained)
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae.eval()
    return sae_id, sae


def parse_judge_category(raw_judge: str) -> Optional[str]:
    """Extract judge category from judge_response JSON."""
    if not raw_judge:
        return None
    
    try:
        parsed = json.loads(raw_judge)
        if isinstance(parsed, dict):
            top_3 = parsed.get("top_3", [])
            if isinstance(top_3, list) and top_3:
                top_entry = top_3[0]
                if isinstance(top_entry, dict):
                    category = top_entry.get("category")
                    if category:
                        return str(category)
    except json.JSONDecodeError:
        pass
    
    # Fallback: regex search
    match = re.search(r'"category"\s*:\s*"([^"]+)"', raw_judge)
    if match:
        return match.group(1)
    
    return None


def get_judge_categories(input_csv: str) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Load judge categories and create row index mapping."""
    category_to_rows = defaultdict(list)
    row_to_category = {}
    
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            # Extract from judge_response JSON
            raw_judge = row.get("judge_response", "")
            category = parse_judge_category(raw_judge)
            
            if not category:
                logger.warning(f"Row {row_idx}: could not parse judge_category")
                category = "unknown"
            
            category_to_rows[category].append(row_idx)
            row_to_category[row_idx] = category
    
    logger.info(f"Found {len(category_to_rows)} judge categories: {list(category_to_rows.keys())[:5]}...")
    for cat, rows in sorted(category_to_rows.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        logger.info(f"  {cat}: {len(rows)} samples")
    
    return dict(category_to_rows), row_to_category


def steer_activations(
    activations: torch.Tensor,
    feature_indices: List[int],
    scale_factors: Dict[int, float],
) -> torch.Tensor:
    """
    Modify activations by scaling/zeroing specific features.
    
    Args:
        activations: (N, D) tensor of SAE feature activations
        feature_indices: which features to steer
        scale_factors: feature_idx -> scale_factor (0 = zero, 1 = no change, >1 = amplify)
    
    Returns:
        Modified activations
    """
    steered = activations.clone()
    for feat_idx in feature_indices:
        if feat_idx in scale_factors:
            scale = scale_factors[feat_idx]
            steered[:, feat_idx] *= scale
    
    return steered


def decode_steering_effect(
    sae: SAE,
    original_acts: torch.Tensor,
    steered_acts: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode activations back through SAE to see reconstructed changes.
    
    Returns:
        (original_reconstructed, steered_reconstructed)
    """
    sae.to(device)
    original_acts = original_acts.to(device)
    steered_acts = steered_acts.to(device)
    
    with torch.no_grad():
        original_recon = sae.decode(original_acts)
        steered_recon = sae.decode(steered_acts)
    
    return original_recon.cpu(), steered_recon.cpu()


def run_steering_experiments(config: SteeringConfig) -> None:
    """Main steering intervention pipeline."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Layer {config.layer} Steering Intervention")
    logger.info(f"Output: {output_dir}")
    
    # Step 1: Load sweep results
    sweep_features = load_sweep_features(config.layer)
    top_feature_indices = sweep_features["top_features"][:10]  # Use top 10
    logger.info(f"Top features to steer: {top_feature_indices}")
    
    # Step 2: Load judge categories
    category_to_rows, row_to_category = get_judge_categories(config.input_csv)
    
    # Step 3: Load SAE
    sae_id, sae = load_sae_for_layer(config.layer, config.release, config.trainer, config.device)
    
    # Step 4: Sample test data (few from each category)
    test_rows = []
    for category, rows in category_to_rows.items():
        sample_count = min(config.num_samples // len(category_to_rows), len(rows))
        test_rows.extend(rows[:sample_count])
    
    logger.info(f"Testing steering on {len(test_rows)} samples")
    
    # Step 5: Load activations
    try:
        activations = load_activations(config.states_dir, test_rows)
    except ValueError as e:
        logger.error(f"Failed to load activations: {e}")
        return
    
    # Step 6: Test different steering strategies
    steering_strategies = {
        "zero_all": {feat_idx: 0.0 for feat_idx in top_feature_indices},
        "half_scale": {feat_idx: 0.5 for feat_idx in top_feature_indices},
        "double_scale": {feat_idx: 2.0 for feat_idx in top_feature_indices},
    }
    
    results = []
    
    for strategy_name, scale_factors in steering_strategies.items():
        logger.info(f"  Testing {strategy_name}...")
        
        steered_acts = steer_activations(activations, top_feature_indices, scale_factors)
        
        # Compute change magnitude
        change = (steered_acts - activations).abs().mean().item()
        max_change = (steered_acts - activations).abs().max().item()
        
        logger.info(f"    Mean change: {change:.6f}, Max change: {max_change:.6f}")
        
        results.append({
            "strategy": strategy_name,
            "mean_activation_change": change,
            "max_activation_change": max_change,
            "num_features_steered": len(top_feature_indices),
            "num_samples": len(test_rows),
        })
    
    # Step 7: Save report
    report_path = output_dir / "steering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "layer": config.layer,
            "sae_id": sae_id,
            "top_features_steered": top_feature_indices,
            "judge_categories": list(category_to_rows.keys()),
            "results": results,
        }, f, indent=2)
    
    logger.info(f"Steering report saved to {report_path}")
    
    # Step 8: Save activations for further analysis
    activation_stats = {
        "original_mean": activations.mean().item(),
        "original_std": activations.std().item(),
        "original_max": activations.max().item(),
        "original_min": activations.min().item(),
        "sparsity": (activations == 0).float().mean().item(),
    }
    
    with open(output_dir / "activation_stats.json", "w", encoding="utf-8") as f:
        json.dump(activation_stats, f, indent=2)
    
    logger.info("Steering intervention complete!")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="SAE steering intervention on judge categories.")
    parser.add_argument("--layer", type=int, default=15, help="Layer to steer (default: 15)")
    parser.add_argument("--input_csv", type=str, default="saved_results/judge/variations_judge.csv")
    parser.add_argument("--states_dir", type=str, default="saved_states")
    parser.add_argument("--output_dir", type=str, default="saved_results/steering_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--release", type=str, default=DEFAULT_SAE_RELEASE)
    parser.add_argument("--trainer", type=str, default=DEFAULT_TRAINER)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=100, help="Samples per category to test")
    
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    config = SteeringConfig(
        layer=args.layer,
        input_csv=args.input_csv,
        states_dir=args.states_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        release=args.release,
        trainer=args.trainer,
        device=args.device,
        num_samples=args.num_samples,
    )
    
    run_steering_experiments(config)


if __name__ == "__main__":
    main()
