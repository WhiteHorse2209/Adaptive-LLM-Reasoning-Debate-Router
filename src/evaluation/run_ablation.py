"""CLI executable for running ablation experiments across parameters."""

import argparse
import sys

from src.config.config import load_config
from src.evaluation.ablation import AblationEngine, get_standard_ablation_matrix
from src.evaluation.dataset import get_benchmark_dataset
from src.provider.factory import get_provider


def main():
    parser = argparse.ArgumentParser(description="Execute ablation studies for Adaptive LLM Router")
    parser.add_argument("--suite", type=str, default="all", choices=["all", "thresholds", "debate", "adjudication", "consistency"], help="Ablation experiment subset")
    parser.add_argument("--limit", type=int, default=None, help="Number of benchmark queries per ablation configuration")
    parser.add_argument("--profile", type=str, default=None, help="Model profile override from config.json")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Directory to export ablation results")
    args = parser.parse_args()

    config = load_config()
    profile = config.get_active_profile(args.profile)
    provider = get_provider(profile)

    items = get_benchmark_dataset(limit=args.limit)
    if not items:
        print("No benchmark items available.")
        sys.exit(1)

    all_configs = get_standard_ablation_matrix()
    if args.suite == "thresholds":
        selected_configs = [c for c in all_configs if "threshold" in c.description.lower() or "baseline" in c.experiment_id]
    elif args.suite == "debate":
        selected_configs = [c for c in all_configs if "debate" in c.experiment_id or "baseline" in c.experiment_id]
    elif args.suite == "adjudication":
        selected_configs = [c for c in all_configs if "judge" in c.experiment_id or "baseline" in c.experiment_id]
    elif args.suite == "consistency":
        selected_configs = [c for c in all_configs if "consistency" in c.experiment_id or "baseline" in c.experiment_id]
    else:
        selected_configs = all_configs

    print(f"\n==========================================================================================")
    print(f"ABLATION STUDY EXPERIMENTS (Configs: {len(selected_configs)}, Queries/Config: {len(items)}, Model: {profile.model})")
    print(f"==========================================================================================")

    engine = AblationEngine(provider)
    results = engine.run_ablation_grid(selected_configs, items)

    print("\n" + "=" * 105)
    print(f"{'EXPERIMENT':<30} {'ACCURACY':<10} {'CALLS/Q':<10} {'TOKENS/Q':<10} {'LATENCY/Q':<12} {'EFF_INDEX':<11} {'ROUTING'}")
    print("-" * 105)

    for r in results:
        routing_str = f"Dir:{r.routing_breakdown.get('DIRECT',0)}% SC:{r.routing_breakdown.get('SELF_CONSISTENCY',0)}% Deb:{r.routing_breakdown.get('MULTI_AGENT_DEBATE',0)}%"
        print(
            f"{r.name[:28]:<30} "
            f"{r.accuracy * 100:>5.1f}%    "
            f"{r.mean_calls_per_query:>6.2f}    "
            f"{r.mean_tokens_per_query:>7.1f}   "
            f"{r.mean_latency_seconds:>7.2f}s    "
            f"{r.efficiency_index:>7.3f}    "
            f"{routing_str}"
        )
    print("=" * 105 + "\n")

    json_path = f"{args.output_dir}/ablation_results.json"
    csv_path = f"{args.output_dir}/ablation_summary.csv"
    engine.export_results(results, json_path=json_path, csv_path=csv_path)
    print(f"Saved ablation artifacts:\n  - JSON: {json_path}\n  - CSV:  {csv_path}\n")


if __name__ == "__main__":
    main()
