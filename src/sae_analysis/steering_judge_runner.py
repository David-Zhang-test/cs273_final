"""
Judge runner for steering outputs saved by `steering_hypothesis_test.py`.

This script reads the `to_judge.jsonl` produced by the generator, calls the
`ResponseJudger` for baseline and each intervention, and writes a finalized
JSON report with judged fields and a summary.

Example:
    python src/sae_analysis/steering_judge_runner.py \
        --input_jsonl saved_results/steering_hypothesis_test_run8/to_judge.jsonl \
        --steering_report saved_results/steering_hypothesis_test_run8/steering_hypothesis_test.json \
        --output_dir saved_results/steering_hypothesis_test_run8 \
        --judge_model gpt-4o-mini --judge_timeout 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
from multiprocessing import Process, Queue

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.judger import ResponseJudger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    def _target(q: Queue):
        try:
            res = try_judge(prompt_text, response_text, variation_type, judge_model, use_local)
        except Exception as e:
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


ABSTENTION_LIKE_CATEGORIES = {
    "Both-Sides Neutrality",
    "The Pivot / Deflection",
    "Theory-Only (Partial Refusal)",
    "Polite Full Refusal",
    "Epistemic Refusal",
}


def is_abstention_like(category: str) -> bool:
    return category in ABSTENTION_LIKE_CATEGORIES


def summarize_results(results: List[Dict]) -> Dict:
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


def run_judge(args) -> None:
    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load steering report if provided (to preserve metadata)
    steering_report = None
    if args.steering_report:
        sr_path = Path(args.steering_report)
        if sr_path.exists():
            with sr_path.open("r", encoding="utf-8") as f:
                try:
                    steering_report = json.load(f)
                except Exception:
                    steering_report = None

    results = []
    # iterate the tasks and call judge for baseline + each intervention
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("prompt", "")
            variation_type = item.get("variation_type", "")
            baseline = item.get("baseline_response", "")

            judged_baseline = safe_try_judge(
                prompt, baseline, variation_type, args.judge_model, args.use_local_judge, timeout_sec=args.judge_timeout
            )

            interventions_payload = {}
            for name, steered in item.get("interventions", {}).items():
                judged_steered = safe_try_judge(
                    prompt, steered, variation_type, args.judge_model, args.use_local_judge, timeout_sec=args.judge_timeout
                )
                interventions_payload[name] = {
                    "strength": None,
                    "steered_response": steered,
                    "steered_judge": judged_steered,
                }

            results.append(
                {
                    "row_index": item.get("row_index"),
                    "prompt": prompt,
                    "baseline_response": baseline,
                    "baseline_judge": judged_baseline,
                    "interventions": interventions_payload,
                }
            )

    final_report = {
        "input_jsonl": str(input_jsonl),
        "steering_report": str(args.steering_report) if args.steering_report else None,
        "results": results,
        "summary": summarize_results(results),
    }

    out_path = output_dir / "steering_hypothesis_test_judged.json"
    out_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved judged steering report to %s", out_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run judge on outputs from steering_hypothesis_test.py")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to to_judge.jsonl produced by generator")
    parser.add_argument("--steering_report", type=str, default="", help="Optional path to original steering report json")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--use_local_judge", action="store_true")
    parser.add_argument("--judge_timeout", type=float, default=10.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_judge(args)


if __name__ == "__main__":
    main()
