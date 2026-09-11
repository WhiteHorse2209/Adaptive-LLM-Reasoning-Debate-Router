"""CLI executable for running benchmark evaluation comparisons."""

import argparse
import json
import sys
import time

from src.config.config import load_config
from src.evaluation.dataset import get_benchmark_dataset
from src.evaluation.evaluator import BenchmarkEvaluator
from src.provider.factory import get_provider


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning paradigms on curated GSM8K benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Number of benchmark queries to evaluate")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (arithmetic, multi_step_math, logic)")
    parser.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty (EASY, MEDIUM, HARD)")
    parser.add_argument("--profile", type=str, default=None, help="Model profile override from config.json")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "direct", "consistency", "debate", "adaptive"], help="Strategy to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation summary JSON")
    args = parser.parse_args()

    config = load_config()
    profile = config.get_active_profile(args.profile)
    provider = get_provider(profile)

    items = get_benchmark_dataset(limit=args.limit, category=args.category, difficulty=args.difficulty)
    if not items:
        print("No benchmark items match the specified criteria.")
        sys.exit(1)

    print(f"\n==========================================================================================")
    print(f"BENCHMARK EVALUATION (Queries: {len(items)}, Model: {profile.model}, Provider: {profile.provider})")
    print(f"==========================================================================================")

    evaluator = BenchmarkEvaluator(
        provider,
        router_config=config.raw_config.get("router", {}),
        debate_config=config.raw_config.get("debate", {}),
    )

    results = {}
    if args.strategy == "all":
        results = evaluator.run_full_comparison(items)
    elif args.strategy == "direct":
        results["direct"] = evaluator.evaluate_direct(items)
    elif args.strategy == "consistency":
        results["self_consistency"] = evaluator.evaluate_self_consistency(items)
    elif args.strategy == "debate":
        results["debate"] = evaluator.evaluate_debate(items)
    elif args.strategy == "adaptive":
        results["adaptive"] = evaluator.evaluate_adaptive(items)

    print("\n" + "=" * 95)
    print(f"{'STRATEGY':<25} {'ACCURACY':<10} {'CALLS/Q':<10} {'TOKENS/Q':<10} {'LATENCY/Q':<12} {'COST':<8} {'ROUTING BREAKDOWN'}")
    print("-" * 95)

    for key, summary in results.items():
        routing_str = ""
        if summary.routing_breakdown:
            parts = [f"{k[:3]}:{v}%" for k, v in summary.routing_breakdown.items()]
            routing_str = " ".join(parts)
        else:
            routing_str = "-"

        print(
            f"{summary.strategy_name:<25} "
            f"{summary.accuracy * 100:>5.1f}%    "
            f"{summary.mean_calls_per_query:>6.2f}    "
            f"{summary.mean_tokens_per_query:>7.1f}   "
            f"{summary.mean_latency_seconds:>7.2f}s    "
            f"${summary.external_api_cost:>5.4f} "
            f"{routing_str}"
        )
    print("=" * 95 + "\n")

    if args.output:
        out_data = {k: v.model_dump() for k, v in results.items()}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Exported benchmark results to: {args.output}")


if __name__ == "__main__":
    main()
