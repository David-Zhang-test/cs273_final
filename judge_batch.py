#!/usr/bin/env python3
"""
Standalone script to judge all responses using OpenAI Batch API.

Usage:
    python judge_batch.py \
        --response_csv saved_results/response/variations_response.csv \
        --output_csv saved_results/judge/variations_judge.csv \
        --judge_model gpt-4o-mini
"""

import argparse
import os
from src.judger import ResponseJudger


def main():
    parser = argparse.ArgumentParser(
        description="Judge model responses using OpenAI Batch API"
    )
    parser.add_argument(
        "--response_csv",
        type=str,
        default="saved_results/response/variations_response.csv",
        help="Path to CSV with model responses",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="saved_results/judge/variations_judge.csv",
        help="Path to save judged output CSV",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model to use",
    )
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="batch_tmp",
        help="Directory to store batch files",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Seconds between batch status checks",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.response_csv):
        print(f"Error: Response CSV not found: {args.response_csv}")
        return 1

    print("=" * 70)
    print("OpenAI Batch API Judging Pipeline")
    print("=" * 70)
    print(f"Response CSV: {args.response_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Batch Dir: {args.batch_dir}")
    print("=" * 70)

    # Run batch judging
    judger = ResponseJudger(judge_model=args.judge_model, use_local=False)
    judger.judge_csv_batch(
        response_csv_path=args.response_csv,
        judged_output_path=args.output_csv,
        batch_dir=args.batch_dir,
    )

    print("\n" + "=" * 70)
    print("Judging complete!")
    print(f"Output saved to: {args.output_csv}")
    print(f"Batch metadata in: {args.batch_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
