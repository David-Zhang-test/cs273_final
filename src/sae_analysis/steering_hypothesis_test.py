"""
Steering hypothesis test for abstention mechanisms.

This script tests two claims from the SAE interpretation results:
1. Suppressing shared abstention features should reduce refusal behavior.
2. Pushing category-specific features should shift the refusal subtype.

The default configuration targets layer 27, where the feature profiles are
most specialized. A dry-run mode is provided so the experiment setup can be
validated even when a full generation run is too expensive.

Example:
    python src/sae_analysis/steering_hypothesis_test.py \
        --layer 27 \
        --input_csv saved_results/judge/variations_judge.csv \
        --output_dir saved_results/steering_hypothesis_test \
        --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sae_lens import SAE
except ImportError:  # pragma: no cover - handled at runtime
    SAE = None

from src.judger import ResponseJudger
from src.runner import ModelRunner


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAE_RELEASE = "llama-3.1-8b-instruct-andyrdt"
DEFAULT_TRAINER = "trainer_1"


SHARED_CORE_FEATURES = {
    27: [106628, 80948, 24431, 6326],
    23: [76786, 73540, 93403],
}

CATEGORY_FEATURES = {
    27: {
        "Polite Full Refusal": [45181, 78807],
        "Theory-Only (Partial Refusal)": [44833, 76568],
        "Epistemic Refusal": [40311, 42366, 82391],
    },
    23: {
        "Polite Full Refusal": [73372],
        "Theory-Only (Partial Refusal)": [93403],
        "Epistemic Refusal": [76786, 73540, 20176, 10317, 106349],
    },
}

SOURCE_CATEGORIES = [
    "Epistemic Refusal",
    "Polite Full Refusal",
    "Theory-Only (Partial Refusal)",
]


@dataclass
class SteeringExample:
    row_index: int
    question: str
    variation: str
    source_category: str
    source_variation_type: str
    judge_category: str


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
    return None


def load_judged_rows(csv_path: str) -> List[SteeringExample]:
    examples: List[SteeringExample] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as file_obj:
        reader = csv.DictReader(file_obj)
        for idx, row in enumerate(reader):
            judge_category = parse_judge_category(row.get("judge_response", ""))
            if not judge_category:
                continue
            examples.append(
                SteeringExample(
                    row_index=idx,
                    question=row.get("question", ""),
                    variation=row.get("variation", row.get("question", "")),
                    source_category=row.get("category", "unknown"),
                    source_variation_type=row.get("variation_type", "unknown"),
                    judge_category=judge_category,
                )
            )
    return examples


def select_examples(examples: Sequence[SteeringExample], categories: Sequence[str], per_category: int) -> List[SteeringExample]:
    selected: List[SteeringExample] = []
    seen = {category: 0 for category in categories}
    for example in examples:
        if example.judge_category not in seen:
            continue
        if seen[example.judge_category] >= per_category:
            continue
        selected.append(example)
        seen[example.judge_category] += 1
    return selected


def load_sae(layer: int, release: str, trainer: str, device: str):
    if SAE is None:
        raise ImportError("sae_lens is not installed in the configured environment")
    sae_id = f"resid_post_layer_{layer}_{trainer}"
    logger.info("Loading SAE %s from %s", sae_id, release)
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae.eval()
    return sae_id, sae


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def build_direction(sae, feature_ids: Iterable[int], device: str) -> torch.Tensor:
    vectors = []
    for feature_id in feature_ids:
        vectors.append(normalize_vector(sae.W_dec[feature_id].detach().to(device=device, dtype=torch.float32)))
    if not vectors:
        raise ValueError("No features provided for steering direction")
    direction = torch.stack(vectors, dim=0).sum(dim=0)
    return normalize_vector(direction)


def build_hook(direction: torch.Tensor, strength: float):
    def _hook(activations, hook):
        steered = activations.clone()
        steered[:, -1, :] = steered[:, -1, :] + strength * direction.to(device=steered.device, dtype=steered.dtype)
        return steered

    return _hook


def infer_prompt(model_runner: ModelRunner, prompt_text: str, max_new_tokens: int, hook_name: Optional[str] = None, hook_fn=None) -> str:
    chat_format = [{"role": "user", "content": prompt_text}]
    formatted_prompt = model_runner.tokenizer.apply_chat_template(chat_format, tokenize=False)
    tokens = model_runner.model.to_tokens(formatted_prompt)

    if hook_name and hook_fn:
        with model_runner.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            generated_tokens = model_runner.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
    else:
        generated_tokens = model_runner.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            verbose=False,
        )

    return model_runner.tokenizer.decode(generated_tokens[0][tokens.shape[1] :])


def try_judge(prompt_text: str, response_text: str, variation_type: str, judge_model: str, use_local: bool) -> Dict[str, str]:
    try:
        judger = ResponseJudger(judge_model=judge_model, use_local=use_local)
        judged = judger.judge_response(prompt_text=prompt_text, response_text=response_text, variation_type=variation_type)
        parsed = json.loads(judged)
        top_3 = parsed.get("top_3", []) if isinstance(parsed, dict) else []
        top_category = top_3[0]["category"] if top_3 else ""
        return {"judge_response": judged, "top_category": top_category}
    except Exception as exc:  # pragma: no cover - external API/network dependent
        return {"judge_response": f"ERROR: {exc}", "top_category": ""}


def run_experiment(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judged_rows = load_judged_rows(args.input_csv)
    examples = select_examples(judged_rows, args.source_categories, args.per_category)

    core_features = SHARED_CORE_FEATURES.get(args.layer)
    if not core_features:
        raise ValueError(f"No shared-core feature set configured for layer {args.layer}")

    category_features = CATEGORY_FEATURES.get(args.layer)
    if not category_features:
        raise ValueError(f"No category feature sets configured for layer {args.layer}")

    logger.info("Selected %d examples", len(examples))
    for example in examples:
        logger.info("  row=%d source=%s judge=%s", example.row_index, example.source_category, example.judge_category)

    sae_id, sae = load_sae(args.layer, args.release, args.trainer, args.device)
    model_runner = None if args.dry_run else ModelRunner(
        model_name=args.model_name,
        device=args.model_device,
        model_dtype=args.model_dtype,
        n_devices=args.n_devices,
    )

    hook_name = f"blocks.{args.layer}.hook_resid_post"
    core_direction = build_direction(sae, core_features, args.device)
    polite_direction = build_direction(sae, category_features["Polite Full Refusal"], args.device)
    theory_direction = build_direction(sae, category_features["Theory-Only (Partial Refusal)"], args.device)
    epistemic_direction = build_direction(sae, category_features["Epistemic Refusal"], args.device)

    interventions = {
        "core_suppress": (core_direction, -args.core_strength),
        "polite_push": (polite_direction, args.category_strength),
        "theory_push": (theory_direction, args.category_strength),
        "epistemic_push": (epistemic_direction, args.category_strength),
    }

    report = {
        "layer": args.layer,
        "sae_id": sae_id,
        "shared_core_features": core_features,
        "category_features": category_features,
        "selected_examples": [example.__dict__ for example in examples],
        "dry_run": args.dry_run,
        "results": [],
    }

    if args.dry_run:
        logger.info("Dry-run mode enabled; skipping generation.")
        (output_dir / "steering_hypothesis_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    if model_runner is None:
        raise RuntimeError("ModelRunner was not initialized")

    for example in examples:
        baseline = infer_prompt(model_runner, example.variation, args.max_new_tokens)

        judged_baseline = {"judge_response": "", "top_category": ""}
        if args.judge:
            judged_baseline = try_judge(example.variation, baseline, example.source_variation_type, args.judge_model, args.use_local_judge)

        example_result = {
            "row_index": example.row_index,
            "source_category": example.source_category,
            "judge_category": example.judge_category,
            "prompt": example.variation,
            "baseline_response": baseline,
            "baseline_judge": judged_baseline,
            "interventions": {},
        }

        for intervention_name, (direction, strength) in interventions.items():
            hook_fn = build_hook(direction, strength)
            steered = infer_prompt(model_runner, example.variation, args.max_new_tokens, hook_name=hook_name, hook_fn=hook_fn)

            judged_steered = {"judge_response": "", "top_category": ""}
            if args.judge:
                judged_steered = try_judge(example.variation, steered, example.source_variation_type, args.judge_model, args.use_local_judge)

            example_result["interventions"][intervention_name] = {
                "strength": strength,
                "steered_response": steered,
                "steered_judge": judged_steered,
            }

        report["results"].append(example_result)

    out_path = output_dir / "steering_hypothesis_test.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved steering hypothesis test to %s", out_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Test shared-core suppression and category-node steering hypotheses.")
    parser.add_argument("--layer", type=int, default=27, help="SAE layer to steer")
    parser.add_argument("--input_csv", type=str, default="saved_results/judge/variations_judge.csv")
    parser.add_argument("--output_dir", type=str, default="saved_results/steering_hypothesis_test")
    parser.add_argument("--release", type=str, default=DEFAULT_SAE_RELEASE)
    parser.add_argument("--trainer", type=str, default=DEFAULT_TRAINER)
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading SAE and steering directions")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_device", type=str, default="cpu", help="Device passed to ModelRunner")
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--per_category", type=int, default=1)
    parser.add_argument("--source_categories", nargs="*", default=SOURCE_CATEGORIES)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--core_strength", type=float, default=3.0)
    parser.add_argument("--category_strength", type=float, default=3.0)
    parser.add_argument("--judge", action="store_true", help="Judge baseline and steered outputs after generation")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--use_local_judge", action="store_true", help="Use local Ollama-compatible judge endpoint")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration without loading the model")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()