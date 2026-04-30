from __future__ import annotations

import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from sae_lens import SAE
except ImportError:  # pragma: no cover - handled at runtime with a clear error message.
    SAE = None


LAYER_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_post"
DEFAULT_SAE_RELEASE = "llama-3.1-8b-instruct-andyrdt"
DEFAULT_TRAINER = "trainer_0"
LAYER_REGEX = re.compile(r"blocks\.(\d+)\.hook_resid_post")
# Valid layers in the andyrdt SAE for Llama 3.1 8B Instruct (sparse layers only)
VALID_LAYERS_BY_RELEASE = {
    "llama-3.1-8b-instruct-andyrdt": {3, 7, 11, 15, 19, 23, 27},
    "andyrdt/saes-llama-3.1-8b-instruct": {3, 7, 11, 15, 19, 23, 27},  # alternate repo id
}


@dataclass
class SweepConfig:
    input_csv: str
    states_dir: str
    output_dir: str
    sample_size: int
    sample_seed: int
    group_by: str
    release: str
    trainer: str
    device: str
    top_k_features: int
    layer_spec: str


@dataclass
class LayerSweepResult:
    layer: int
    sae_id: str
    group_by: str
    sample_size: int
    group_count: int
    separation_score: float
    cramers_v: float
    f_statistic: float
    representational_distance: float
    mean_l0: float
    top_feature_indices: List[int]
    top_feature_spreads: List[float]
    group_counts: Dict[str, int]
    group_mean_top_features: Dict[str, List[float]]


def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv_rows(csv_path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(csv_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_group_label(row: Dict[str, str], group_by: str) -> str:
    if group_by in row and row[group_by] != "":
        return str(row[group_by])

    if group_by == "judge_category":
        raw_judge = row.get("judge_response", "")
        parsed = parse_judge_category(raw_judge)
        if parsed is None:
            raise ValueError("Could not parse judge category from judge_response.")
        return parsed

    raise KeyError(f"Column '{group_by}' is missing from the input row.")


def parse_judge_category(raw_judge: str) -> Optional[str]:
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

    match = re.search(r'"category"\s*:\s*"([^"]+)"', raw_judge)
    if match:
        return match.group(1)

    return None


def infer_available_layers(sample_state: Dict[str, Any]) -> List[int]:
    residual_stream = sample_state.get("residual_stream_last_token", {})
    layer_ids: List[int] = []
    for hook_name in residual_stream.keys():
        match = LAYER_REGEX.match(hook_name)
        if match:
            layer_ids.append(int(match.group(1)))
    return sorted(set(layer_ids))


def parse_layer_spec(layer_spec: str, available_layers: Sequence[int]) -> List[int]:
    if layer_spec.strip().lower() == "all":
        return list(available_layers)

    selected: set[int] = set()
    for chunk in layer_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            for layer in range(start, end + 1):
                if layer in available_layers:
                    selected.add(layer)
        else:
            layer = int(chunk)
            if layer in available_layers:
                selected.add(layer)
    return sorted(selected)


def sample_rows_by_group(
    rows: Sequence[Dict[str, str]],
    sample_size: int,
    group_by: str,
    seed: int,
) -> List[Tuple[int, Dict[str, str]]]:
    # sample_size = -1 means use all rows
    if sample_size < 0 or sample_size >= len(rows):
        return list(enumerate(rows))

    grouped: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        grouped[get_group_label(row, group_by)].append((row_index, row))

    group_names = sorted(grouped.keys())
    total_rows = len(rows)
    rng = random.Random(seed)

    allocations: Dict[str, int] = {}
    fractional_parts: Dict[str, float] = {}
    for group_name in group_names:
        group_size = len(grouped[group_name])
        raw_allocation = sample_size * group_size / total_rows
        allocations[group_name] = min(group_size, int(math.floor(raw_allocation)))
        fractional_parts[group_name] = raw_allocation - math.floor(raw_allocation)

    if sample_size >= len(group_names):
        for group_name in group_names:
            if allocations[group_name] == 0 and len(grouped[group_name]) > 0:
                allocations[group_name] = 1

    current_total = sum(allocations.values())

    if current_total > sample_size:
        ranked_groups = sorted(
            group_names,
            key=lambda name: (fractional_parts[name], allocations[name]),
        )
        while current_total > sample_size:
            for group_name in ranked_groups:
                if current_total <= sample_size:
                    break
                if allocations[group_name] > 1:
                    allocations[group_name] -= 1
                    current_total -= 1
    elif current_total < sample_size:
        ranked_groups = sorted(
            group_names,
            key=lambda name: (-fractional_parts[name], len(grouped[name]) - allocations[name]),
        )
        while current_total < sample_size:
            progress_made = False
            for group_name in ranked_groups:
                if current_total >= sample_size:
                    break
                if allocations[group_name] < len(grouped[group_name]):
                    allocations[group_name] += 1
                    current_total += 1
                    progress_made = True
            if not progress_made:
                break

    sampled: List[Tuple[int, Dict[str, str]]] = []
    for group_name in group_names:
        group_rows = grouped[group_name]
        allocation = min(allocations[group_name], len(group_rows))
        if allocation <= 0:
            continue
        sampled.extend(rng.sample(group_rows, allocation))

    rng.shuffle(sampled)
    return sampled[:sample_size]


def load_state(states_dir: str, prompt_id: int) -> Dict[str, Any]:
    state_path = Path(states_dir) / f"prompt_{prompt_id:06d}.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing saved state: {state_path}")
    return torch.load(state_path, map_location="cpu")


def load_sae_for_layer(release: str, layer: int, trainer: str, device: str):
    if SAE is None:
        raise ImportError(
            "sae_lens is not installed. Install dependencies from requirements.txt first."
        )

    # Check if this layer is valid for the given release
    valid_layers = VALID_LAYERS_BY_RELEASE.get(release, set())
    if valid_layers and layer not in valid_layers:
        raise ValueError(
            f"Layer {layer} is not available in release '{release}'. "
            f"Valid layers: {sorted(valid_layers)}"
        )

    # Extract trainer number (e.g., "trainer_0" -> "0")
    trainer_num = trainer.split("_")[-1] if "_" in trainer else trainer

    # Try candidate SAE IDs with underscore format (official format for this repo)
    candidate_sae_ids = [
        f"resid_post_layer_{layer}_trainer_{trainer_num}",
        f"resid_post_layer_{layer}_trainer_1",  # fallback to trainer_1 if trainer_0 not found
        f"resid_post_layer_{layer}",
        f"layer_{layer}_trainer_{trainer_num}",
        f"layer_{layer}",
    ]

    errors: List[str] = []
    for sae_id in candidate_sae_ids:
        try:
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            return sae_id, sae
        except Exception as exc:  # pragma: no cover - depends on external HF assets.
            errors.append(f"{sae_id}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        f"Could not load an SAE for layer {layer} from release '{release}'.\n{joined_errors}"
    )


def extract_layer_activation(
    state: Dict[str, Any], layer: int, device: str, dtype: torch.dtype
) -> torch.Tensor:
    hook_name = LAYER_HOOK_TEMPLATE.format(layer=layer)
    residual_stream = state.get("residual_stream_last_token", {})
    if hook_name not in residual_stream:
        raise KeyError(f"Layer {layer} is missing from the saved state.")
    activation = residual_stream[hook_name].detach().to(device=device, dtype=dtype)
    return activation


def compute_layer_summary(
    sae,
    sae_id: str,
    layer: int,
    sampled_rows: Sequence[Tuple[int, Dict[str, str]]],
    states_dir: str,
    group_by: str,
    top_k_features: int,
    device: str,
) -> Tuple[LayerSweepResult, torch.Tensor, List[str]]:
    labels: List[str] = []
    activations: List[torch.Tensor] = []

    target_dtype = getattr(sae.W_enc, "dtype", torch.float32)
    for row_index, row in sampled_rows:
        state = load_state(states_dir, row_index)
        activations.append(extract_layer_activation(state, layer, device=device, dtype=target_dtype))
        labels.append(get_group_label(row, group_by))

    stacked_activations = torch.stack(activations, dim=0)
    with torch.no_grad():
        feature_acts = sae.encode(stacked_activations.unsqueeze(1))[:, 0, :]

    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    ordered_labels = sorted(label_to_indices.keys())
    group_counts = {label: len(indices) for label, indices in label_to_indices.items()}
    group_means = {
        label: feature_acts[indices].mean(dim=0) for label, indices in label_to_indices.items()
    }

    global_mean = feature_acts.mean(dim=0)
    between_terms = []
    within_terms = []
    for label, indices in label_to_indices.items():
        group_tensor = feature_acts[indices]
        group_mean = group_means[label]
        between_terms.append(torch.mean((group_mean - global_mean) ** 2))
        within_terms.append(torch.mean((group_tensor - group_mean) ** 2))

    between_score = torch.stack(between_terms).mean() if between_terms else torch.tensor(0.0)
    within_score = torch.stack(within_terms).mean() if within_terms else torch.tensor(1.0)
    separation_score = float((between_score / (within_score + 1e-8)).item())

    # New Metric 1: Cramér's V (association strength 0-1)
    # Measures how strongly SAE features correlate with group membership.
    try:
        n_points = feature_acts.shape[0]
        n_groups = len(label_to_indices)
        
        # Use between/within variance to estimate effect size
        # eta-squared = between_var / total_var
        total_var = between_score + within_score
        eta_squared = float((between_score / (total_var + 1e-8)).item())
        
        # Cramér's V approximation: sqrt(eta_squared / (k-1)) where k is number of groups
        k = n_groups
        cramers_v = float(np.sqrt(eta_squared / (k - 1 + 1e-8))) if k > 1 else 0.0
    except Exception as e:
        cramers_v = 0.0
    
    # New Metric 2: F-statistic (one-way ANOVA)
    # Tests if group means differ significantly in SAE feature space.
    # F = (between_var / df_between) / (within_var / df_within)
    try:
        k = len(label_to_indices)  # number of groups
        n = feature_acts.shape[0]
        df_between = k - 1
        df_within = n - k
        
        between_var = between_score / (df_between + 1e-8) if df_between > 0 else between_score
        within_var = within_score / (df_within + 1e-8) if df_within > 0 else within_score
        f_statistic = float((between_var / (within_var + 1e-8)).item())
    except Exception as e:
        f_statistic = 0.0
    
    # New Metric 3: Representational Distance
    # Average pairwise L2 distance between group centroids, normalized by within-group spread.
    try:
        ordered_labels = sorted(label_to_indices.keys())
        group_centers = [group_means[label] for label in ordered_labels]
        pairwise_dists = []
        for i in range(len(group_centers)):
            for j in range(i + 1, len(group_centers)):
                dist = torch.norm(group_centers[i] - group_centers[j], 2).item()
                pairwise_dists.append(dist)
        mean_pairwise_dist = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
        
        # Normalize by overall feature std
        feat_std = torch.std(feature_acts).item()
        representational_distance = mean_pairwise_dist / (feat_std + 1e-8)
    except Exception as e:
        representational_distance = 0.0

    l0_mean = float((feature_acts > 0).sum(dim=-1).float().mean().item())

    stacked_group_means = torch.stack([group_means[label] for label in ordered_labels], dim=0)
    spread = stacked_group_means.max(dim=0).values - stacked_group_means.min(dim=0).values
    top_k = min(top_k_features, spread.shape[0])
    top_features = torch.topk(spread, k=top_k)
    top_feature_indices = top_features.indices.tolist()
    top_feature_spreads = top_features.values.tolist()
    group_mean_top_features = {
        label: group_means[label][top_feature_indices].detach().cpu().tolist()
        for label in ordered_labels
    }

    result = LayerSweepResult(
        layer=layer,
        sae_id=sae_id,
        group_by=group_by,
        sample_size=len(sampled_rows),
        group_count=len(ordered_labels),
        separation_score=separation_score,
        cramers_v=cramers_v,
        f_statistic=f_statistic,
        representational_distance=representational_distance,
        mean_l0=l0_mean,
        top_feature_indices=top_feature_indices,
        top_feature_spreads=top_feature_spreads,
        group_counts=group_counts,
        group_mean_top_features=group_mean_top_features,
    )

    return result, feature_acts.detach().cpu(), ordered_labels


def save_layer_summary(output_dir: Path, result: LayerSweepResult) -> None:
    output_path = output_dir / f"layer_{result.layer:02d}_summary.json"
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(asdict(result), file_obj, indent=2)


def write_layer_summary_csv(output_dir: Path, results: Sequence[LayerSweepResult]) -> Path:
    csv_rows: List[Dict[str, Any]] = []
    for result in results:
        csv_rows.append(
            {
                "layer": result.layer,
                "sae_id": result.sae_id,
                "group_by": result.group_by,
                "sample_size": result.sample_size,
                "group_count": result.group_count,
                "separation_score": f"{result.separation_score:.6f}",
                "cramers_v": f"{result.cramers_v:.6f}",
                "f_statistic": f"{result.f_statistic:.6f}",
                "representational_distance": f"{result.representational_distance:.6f}",
                "mean_l0": f"{result.mean_l0:.6f}",
                "top_feature_indices": "|".join(map(str, result.top_feature_indices)),
                "top_feature_spreads": "|".join(f"{value:.6f}" for value in result.top_feature_spreads),
            }
        )

    csv_path = output_dir / "layer_sweep_summary.csv"
    write_csv_rows(str(csv_path), csv_rows)
    return csv_path


def plot_layer_scores(output_dir: Path, results: Sequence[LayerSweepResult]) -> Path:
    layers = [result.layer for result in results]
    separation_scores = [result.separation_score for result in results]
    cramers_vs = [result.cramers_v for result in results]
    f_stats = [result.f_statistic for result in results]
    rep_dists = [result.representational_distance for result in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Separation Score (original metric)
    axes[0, 0].plot(layers, separation_scores, marker="o", color="blue")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Separation Score")
    axes[0, 0].set_title("Separation Score (Variance Ratio)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cramér's V (effect size 0-1)
    axes[0, 1].plot(layers, cramers_vs, marker="s", color="green")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Cramér's V")
    axes[0, 1].set_title("Cramér's V (Association Strength)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot 3: F-Statistic (ANOVA)
    axes[1, 0].plot(layers, f_stats, marker="^", color="orange")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("F-Statistic")
    axes[1, 0].set_title("F-Statistic (ANOVA)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Representational Distance
    axes[1, 1].plot(layers, rep_dists, marker="d", color="red")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Representational Distance")
    axes[1, 1].set_title("Representational Distance (Centroid Separation)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    plot_path = output_dir / "layer_sweep_scores.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def build_sample_manifest(
    sampled_rows: Sequence[Tuple[int, Dict[str, str]]], states_dir: str, group_by: str
) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for row_index, row in sampled_rows:
        label = get_group_label(row, group_by)
        manifest.append(
            {
                "row_index": row_index,
                "question": row.get("question", ""),
                "variation": row.get("variation", row.get("question", "")),
                "variation_type": row.get("variation_type", ""),
                "category": row.get("category", ""),
                "group_label": label,
                "state_file": str(Path(states_dir) / f"prompt_{row_index:06d}.pt"),
            }
        )
    return manifest


def run_sweep(
    input_csv: str,
    states_dir: str,
    output_dir: str,
    sample_size: int,
    sample_seed: int,
    group_by: str,
    release: str,
    trainer: str,
    device: str,
    top_k_features: int,
    layer_spec: str = "all",
) -> Path:
    rows = read_csv_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    first_state = load_state(states_dir, 0)
    available_layers = infer_available_layers(first_state)
    if not available_layers:
        raise ValueError("Could not infer any layers from the saved state cache.")

    # Filter available_layers to only those valid for the chosen release
    valid_layers = VALID_LAYERS_BY_RELEASE.get(release, set())
    if valid_layers:
        available_layers = [l for l in available_layers if l in valid_layers]
        if not available_layers:
            raise ValueError(
                f"No valid layers found after filtering to release '{release}'. "
                f"Valid layers for this release: {sorted(valid_layers)}"
            )

    selected_layers = parse_layer_spec(layer_spec, available_layers)
    if not selected_layers:
        raise ValueError(f"No valid layers selected from '{layer_spec}'.")

    sampled_rows = sample_rows_by_group(rows, sample_size=sample_size, group_by=group_by, seed=sample_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_manifest = build_sample_manifest(sampled_rows, states_dir=states_dir, group_by=group_by)
    write_csv_rows(str(output_path / "sample_manifest.csv"), sample_manifest)

    config = SweepConfig(
        input_csv=input_csv,
        states_dir=states_dir,
        output_dir=output_dir,
        sample_size=len(sampled_rows),
        sample_seed=sample_seed,
        group_by=group_by,
        release=release,
        trainer=trainer,
        device=device,
        top_k_features=top_k_features,
        layer_spec=layer_spec,
    )

    results: List[LayerSweepResult] = []
    metadata: Dict[str, Any] = {
        "config": asdict(config),
        "available_layers": available_layers,
        "selected_layers": selected_layers,
        "sample_manifest_path": str(output_path / "sample_manifest.csv"),
        "layer_summaries": [],
    }

    for i, layer in enumerate(selected_layers):
        print(f"[{i+1}/{len(selected_layers)}] Loading SAE for layer {layer}...")
        sae_id, sae = load_sae_for_layer(release=release, layer=layer, trainer=trainer, device=device)
        print(f"[{i+1}/{len(selected_layers)}] Computing summary for layer {layer}...")
        result, _, _ = compute_layer_summary(
            sae=sae,
            sae_id=sae_id,
            layer=layer,
            sampled_rows=sampled_rows,
            states_dir=states_dir,
            group_by=group_by,
            top_k_features=top_k_features,
            device=device,
        )
        results.append(result)
        save_layer_summary(output_path, result)
        metadata["layer_summaries"].append(asdict(result))
        
        # Memory cleanup: explicitly delete SAE to free memory before loading next one
        del sae
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    metadata["best_layer"] = max(results, key=lambda item: item.f_statistic).layer if results else None
    metadata["summary_csv"] = str(write_layer_summary_csv(output_path, results))
    metadata["score_plot"] = str(plot_layer_scores(output_path, results))

    with open(output_path / "sweep_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    return output_path


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run an SAE layer sweep over sampled prompts.")
    parser.add_argument("--input_csv", type=str, default="data/synthesized/variations.csv")
    parser.add_argument("--states_dir", type=str, default="saved_states")
    parser.add_argument("--output_dir", type=str, default="saved_results/sae_sweep")
    parser.add_argument("--sample_size", type=int, default=-1, help="Sample size for stratified sampling (-1 uses all rows)")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument(
        "--group_by",
        type=str,
        default="variation_type",
        choices=["variation_type", "category", "judge_category"],
        help="Column used for stratified sampling and aggregation.",
    )
    parser.add_argument(
        "--sae_release",
        type=str,
        default=DEFAULT_SAE_RELEASE,
        help="Hugging Face SAE release to load.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default=DEFAULT_TRAINER,
        help="SAE subfolder to use for each layer, for example trainer_0.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k_features", type=int, default=20)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated layers or ranges such as '0-7,12,16-31'.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = run_sweep(
        input_csv=args.input_csv,
        states_dir=args.states_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        group_by=args.group_by,
        release=args.sae_release,
        trainer=args.trainer,
        device=args.device,
        top_k_features=args.top_k_features,
        layer_spec=args.layers,
    )
    print(f"Saved SAE sweep outputs to {output_path}")


if __name__ == "__main__":
    main()
