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
from multiprocessing import Process, Queue

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

CATEGORY_TAXONOMY = [
    "Full Compliance",
    "Caveat / Disclaimer Compliance",
    "Preach / Lecturing Compliance",
    "Both-Sides Neutrality",
    "The Pivot / Deflection",
    "Theory-Only (Partial Refusal)",
    "Polite Full Refusal",
    "Epistemic Refusal",
]

CATEGORY_FEATURES = {
    27: {
        "Full Compliance": [44833, 106628, 80948],
        "Caveat / Disclaimer Compliance": [44833, 106628, 80948],
        "Preach / Lecturing Compliance": [9897, 44833, 80948],
        "Both-Sides Neutrality": [44833, 106628, 80948],
        "The Pivot / Deflection": [44833, 80948, 106628],
        "Theory-Only (Partial Refusal)": [44833, 76568, 106628],
        "Polite Full Refusal": [45181, 78807, 44833],
        "Epistemic Refusal": [40311, 42366, 82391],
    },
    23: {
        "Polite Full Refusal": [73372],
        "Theory-Only (Partial Refusal)": [93403],
        "Epistemic Refusal": [76786, 73540, 20176, 10317, 106349],
    },
}

SOURCE_CATEGORIES = [
    "Full Compliance",
    "Caveat / Disclaimer Compliance",
    "Preach / Lecturing Compliance",
    "Both-Sides Neutrality",
    "The Pivot / Deflection",
    "Theory-Only (Partial Refusal)",
    "Polite Full Refusal",
    "Epistemic Refusal",
]

ABSTENTION_LIKE_CATEGORIES = {
    "Both-Sides Neutrality",
    "The Pivot / Deflection",
    "Theory-Only (Partial Refusal)",
    "Polite Full Refusal",
    "Epistemic Refusal",
}


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


def safe_try_judge(prompt_text: str, response_text: str, variation_type: str, judge_model: str, use_local: bool, timeout_sec: float = 10.0) -> Dict[str, str]:
    """Run try_judge in a separate process and enforce a timeout.

    This prevents the main script from blocking indefinitely if the judge
    endpoint is unreachable or the client library retries for a long time.
    """
    def _target(q: Queue):
        try:
            res = try_judge(prompt_text, response_text, variation_type, judge_model, use_local)
        except Exception as e:  # pragma: no cover - defensive
            res = {"judge_response": f"ERROR: {e}", "top_category": ""}
        try:
            q.put(res)
        except Exception:
            pass

    q: Queue = Queue()
    p = Process(target=_target, args=(q,))
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join()
        return {"judge_response": "ERROR: timeout", "top_category": ""}

    if q.empty():
        return {"judge_response": "ERROR: no response", "top_category": ""}

    try:
        return q.get_nowait()
    except Exception:
        return {"judge_response": "ERROR: unable to read judge result", "top_category": ""}


def is_abstention_like(category: str) -> bool:
    return category in ABSTENTION_LIKE_CATEGORIES


def summarize_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {
        "total_examples": len(results),
        "baseline_top_categories": {},
        "intervention_top_categories": {},
        "baseline_abstention_like_rate": 0.0,
        "core_suppress_abstention_like_rate": 0.0,
        "category_push_shift_rate": {},
    }

    if not results:
        return summary

    baseline_abstention_like = 0
    core_suppress_abstention_like = 0
    category_push_totals: Dict[str, Dict[str, int]] = {}

    for result in results:
        baseline_top = result.get("baseline_judge", {}).get("top_category", "")
        if baseline_top:
            summary["baseline_top_categories"][baseline_top] = summary["baseline_top_categories"].get(baseline_top, 0) + 1
        if is_abstention_like(str(baseline_top)):
            baseline_abstention_like += 1

        interventions = result.get("interventions", {})
        for name, payload in interventions.items():
            steered_top = payload.get("steered_judge", {}).get("top_category", "")
            if steered_top:
                summary["intervention_top_categories"][name] = summary["intervention_top_categories"].get(name, {})
                summary["intervention_top_categories"][name][steered_top] = summary["intervention_top_categories"][name].get(steered_top, 0) + 1

            if name == "core_suppress" and is_abstention_like(str(steered_top)):
                core_suppress_abstention_like += 1

            if name.startswith("category_push_"):
                category_name = name[len("category_push_"):]
                category_push_totals.setdefault(category_name, {"match": 0, "total": 0})
                category_push_totals[category_name]["total"] += 1
                if steered_top == category_name:
                    category_push_totals[category_name]["match"] += 1

    summary["baseline_abstention_like_rate"] = baseline_abstention_like / len(results)
    summary["core_suppress_abstention_like_rate"] = core_suppress_abstention_like / len(results)
    summary["category_push_shift_rate"] = {
        category: (vals["match"] / vals["total"] if vals["total"] else 0.0)
        for category, vals in category_push_totals.items()
    }
    return summary


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

    category_directions = {
        category_name: build_direction(sae, feature_ids, args.device)
        for category_name, feature_ids in category_features.items()
    }

    # diagnostic: compute cosine similarities between directions
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a_f = a.to(dtype=torch.float32)
        b_f = b.to(dtype=torch.float32)
        denom = (torch.linalg.norm(a_f) * torch.linalg.norm(b_f)).item()
        if denom == 0:
            return 0.0
        return float(torch.dot(a_f, b_f).item() / denom)

    direction_cosines = {
        "core_vs_category": {},
        "category_vs_category": {},
    }
    for cname, cdir in category_directions.items():
        direction_cosines["core_vs_category"][cname] = cosine(core_direction, cdir)

    category_names = list(category_directions.keys())
    for i, na in enumerate(category_names):
        direction_cosines["category_vs_category"][na] = {}
        for j, nb in enumerate(category_names):
            if i == j:
                direction_cosines["category_vs_category"][na][nb] = 1.0
            else:
                direction_cosines["category_vs_category"][na][nb] = cosine(category_directions[na], category_directions[nb])

    report["direction_cosines"] = direction_cosines

    if args.diagnose_directions:
        (output_dir / "steering_hypothesis_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote direction cosines to %s and exiting (diagnose_directions)", output_dir / "steering_hypothesis_test.json")
        return

    interventions = {"core_suppress": (core_direction, -args.core_strength)}
    for category_name, direction in category_directions.items():
        interventions[f"category_push_{category_name}"] = (direction, args.category_strength)

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

    # Generate responses and save them for offline judging.
    # NOTE: per user request, do not call judge here; instead save all model outputs
    # and a `to_judge.jsonl` that can be consumed later by a separate judging runner.
    out_path = output_dir / "steering_hypothesis_test.json"
    to_judge_path = output_dir / "to_judge.jsonl"
    with to_judge_path.open("w", encoding="utf-8") as tj_f:
        for example in examples:
            baseline = infer_prompt(model_runner, example.variation, args.max_new_tokens)

            example_result = {
                "row_index": example.row_index,
                "source_category": example.source_category,
                "judge_category": example.judge_category,
                "prompt": example.variation,
                "baseline_response": baseline,
                "baseline_judge": {"judge_response": "", "top_category": ""},
                "interventions": {},
            }

            for intervention_name, (direction, strength) in interventions.items():
                hook_fn = build_hook(direction, strength)
                steered = infer_prompt(
                    model_runner, example.variation, args.max_new_tokens, hook_name=hook_name, hook_fn=hook_fn
                )

                # leave judged fields empty; the separate judge runner will fill them
                example_result["interventions"][intervention_name] = {
                    "strength": strength,
                    "steered_response": steered,
                    "steered_judge": {"judge_response": "", "top_category": ""},
                }

            report["results"].append(example_result)
            # write a compact record for offline judging
            compact = {
                "row_index": example.row_index,
                "prompt": example.variation,
                "variation_type": example.source_variation_type,
                "baseline_response": example_result["baseline_response"],
                "interventions": {
                    name: payload["steered_response"] for name, payload in example_result["interventions"].items()
                },
            }
            tj_f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    logger.info("Saved model outputs and judge tasks to %s (and %s)", out_path, to_judge_path)

    report["summary"] = summarize_results(report["results"])

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
    parser.add_argument("--judge_timeout", type=float, default=10.0, help="Timeout in seconds for judge API calls (per call)")
    parser.add_argument("--diagnose_directions", action="store_true", help="Compute and save cosine similarities between SAE directions and exit")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration without loading the model")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()