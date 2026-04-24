import argparse
import os

from src.runner import ModelRunner
from src.judger import ResponseJudger


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run inference + judge pipeline on variations.csv")
    parser.add_argument("--input_csv", type=str, default="data/synthesized/variations.csv")
    parser.add_argument(
        "--response_csv",
        type=str,
        default="saved_results/response/variations_response.csv",
    )
    parser.add_argument(
        "--judged_csv",
        type=str,
        default="saved_results/judge/variations_judge.csv",
    )
    parser.add_argument("--states_dir", type=str, default="saved_states")
    parser.add_argument(
        "--inference_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--local", action="store_true", help="Use local OpenAI-compatible endpoint")
    return parser


def main():
    args = build_arg_parser().parse_args()

    os.makedirs("saved_results/judge", exist_ok=True)
    os.makedirs(args.states_dir, exist_ok=True)

    print("--- 1. Inference on Dataset ---")
    runner = ModelRunner(
        model_name=args.inference_model,
        device=args.device,
        model_dtype=args.model_dtype,
    )
    runner.infer_dataset(
        input_csv_path=args.input_csv,
        response_output_csv=args.response_csv,
        states_output_dir=args.states_dir,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
    )

    # print("\n--- 2. Judge Model Responses ---")
    # judger = ResponseJudger(judge_model=args.judge_model, use_local=args.local)
    # judger.judge_csv(
    #     response_csv_path=args.response_csv,
    #     judged_output_path=args.judged_csv,
    # )

    # print("\nPipeline complete.")
    # print(f"Responses: {args.response_csv}")
    # print(f"Judged outputs: {args.judged_csv}")
    # print(f"Saved states: {args.states_dir}")


if __name__ == "__main__":
    main()
