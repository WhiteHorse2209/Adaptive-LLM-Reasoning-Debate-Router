"""CLI runner for Confidence Calibration (ECE) and 8-Category Failure Mode Analysis."""

import argparse
import json
import sys
from typing import Any, Dict, List

from src.config.config import load_config
from src.evaluation.calibration import compute_calibration
from src.evaluation.dataset import get_benchmark_dataset, is_answer_correct
from src.evaluation.failure_analysis import CATEGORY_DESCRIPTIONS, FailureAnalyzer, FailureCategory
from src.provider.factory import get_provider
from src.router.router import AdaptiveRouter


def render_ascii_reliability_diagram(report) -> str:
    """Formats an ASCII reliability diagram comparing confidence to empirical accuracy."""
    lines = []
    lines.append("\n==========================================================================================")
    lines.append(f"CONFIDENCE RELIABILITY DIAGRAM (Total: {report.total_samples} | ECE: {report.expected_calibration_error:.4f} | MCE: {report.maximum_calibration_error:.4f})")
    lines.append("==========================================================================================")
    lines.append(f"{'BIN':<12} {'SAMPLES':<10} {'CONFIDENCE':<14} {'ACCURACY':<12} {'GAP (ERROR)':<14} {'RELIABILITY VISUAL'}")
    lines.append("-" * 90)

    for b in report.bins:
        bin_label = f"[{b.bin_lower:.1f}, {b.bin_upper:.1f}]"
        # Bar chart: show confidence vs accuracy
        # 10 chars for 0.0-1.0
        acc_bar_len = int(round(b.empirical_accuracy * 10))
        conf_bar_len = int(round(b.mean_confidence * 10))
        visual = f"Acc:[{'#' * acc_bar_len:<10}] Conf:[{'*' * conf_bar_len:<10}]"
        lines.append(
            f"{bin_label:<12} {b.sample_count:<10} {b.mean_confidence:<14.4f} {b.empirical_accuracy:<12.4f} {b.calibration_error:<14.4f} {visual}"
        )

    lines.append("=" * 90)
    lines.append(f"Overall Accuracy: {report.overall_accuracy:.4f} | Mean Confidence: {report.mean_confidence:.4f}")
    lines.append(f"Brier Score: {report.brier_score:.4f} | Overconfident: {'YES' if report.is_overconfident else 'NO (Calibrated/Underconfident)'}")
    return "\n".join(lines)


def render_failure_mode_summary(summary) -> str:
    """Formats an ASCII summary table of the 8-category failure and behavior taxonomy."""
    lines = []
    lines.append("\n==========================================================================================")
    lines.append(f"8-CATEGORY FAILURE & BEHAVIOR TAXONOMY (Total Analyzed: {summary.total_analyzed})")
    lines.append("==========================================================================================")
    lines.append(f"{'CATEGORY':<32} {'COUNT':<8} {'PERCENT':<10} {'DESCRIPTION'}")
    lines.append("-" * 90)

    for cat_name, count in summary.category_counts.items():
        pct = summary.category_percentages.get(cat_name, 0.0)
        desc = CATEGORY_DESCRIPTIONS.get(FailureCategory(cat_name), "")
        lines.append(f"{cat_name:<32} {count:<8} {pct:>6.1f}%    {desc[:40]}...")

    lines.append("=" * 90)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Confidence Calibration and Failure Analysis Runner")
    parser.add_argument("--limit", type=int, default=None, help="Number of benchmark queries to analyze")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty")
    parser.add_argument("--bins", type=int, default=5, help="Number of calibration probability bins")
    parser.add_argument("--profile", type=str, default=None, help="Model profile override from config.json")
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic calibration data for validation")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON summary")
    args = parser.parse_args()

    items = get_benchmark_dataset(limit=args.limit, category=args.category, difficulty=args.difficulty)

    confidences: List[float] = []
    labels: List[bool] = []
    traces: List[Dict[str, Any]] = []

    if args.dry_run or not items:
        # Synthetic representative traces covering calibration bins and failure categories
        sample_data = [
            (0.95, True, "DIRECT", "EASY", None, None, None, False, False),
            (0.88, True, "DIRECT", "EASY", None, None, None, False, False),
            (0.78, False, "DIRECT", "HARD", None, None, None, False, False), # UNDER_ROUTING
            (0.92, True, "MULTI_AGENT_DEBATE", "EASY", True, True, True, False, False), # OVER_ROUTING
            (0.65, True, "MULTI_AGENT_DEBATE", "MEDIUM", False, True, True, False, False), # INITIAL_WRONG_DEBATE_CORRECT
            (0.35, False, "MULTI_AGENT_DEBATE", "HARD", False, False, False, False, False), # INITIAL_WRONG_DEBATE_WRONG
            (0.82, True, "MULTI_AGENT_DEBATE", "HARD", True, True, True, False, False), # INITIAL_CORRECT_DEBATE_CORRECT
            (0.70, False, "MULTI_AGENT_DEBATE", "HARD", True, False, False, False, True), # INITIAL_CORRECT_DEBATE_WRONG
            (0.60, False, "MULTI_AGENT_DEBATE", "HARD", True, False, False, False, False), # JUDGE_SELECTION_ERROR
            (0.40, False, "MULTI_AGENT_DEBATE", "HARD", False, False, False, True, False), # UNANIMOUS_HALLUCINATION
        ]
        for idx, (conf, correct, strat, diff, a1, a2, j_sel, unan, conc) in enumerate(sample_data):
            confidences.append(conf)
            labels.append(correct)
            traces.append({
                "query_id": f"synthetic_{idx+1}",
                "question": f"Sample synthetic reasoning problem {idx+1}",
                "ground_truth": "42",
                "predicted_answer": "42" if correct else "0",
                "is_correct": correct,
                "strategy": strat,
                "difficulty": diff,
                "agent1_correct": a1,
                "agent2_correct": a2,
                "judge_selected_correct": j_sel,
                "unanimous_wrong_premise": unan,
                "concession_to_error": conc,
            })
    else:
        config = load_config()
        profile = config.get_active_profile(args.profile)
        provider = get_provider(profile)
        router = AdaptiveRouter(provider, config=config.raw_config.get("router", {}))

        print(f"\nEvaluating {len(items)} queries for calibration on {profile.model}...")
        for item in items:
            response = router.route_and_execute(item.question)
            correct = is_answer_correct(response.answer, item.ground_truth)
            conf = response.confidence.score
            confidences.append(conf)
            labels.append(correct)

            traces.append({
                "query_id": item.id,
                "question": item.question,
                "ground_truth": item.ground_truth,
                "predicted_answer": response.answer,
                "is_correct": correct,
                "strategy": response.strategy.value,
                "difficulty": response.difficulty.value,
            })

    # Compute calibration and failure summary
    report = compute_calibration(confidences, labels, num_bins=args.bins)
    analyzer = FailureAnalyzer()
    summary = analyzer.analyze_records(traces)

    print(render_ascii_reliability_diagram(report))
    print(render_failure_mode_summary(summary))

    if args.output:
        combined = {
            "calibration_report": report.model_dump(),
            "failure_analysis": summary.model_dump(),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved calibration & failure analysis summary to: {args.output}")


if __name__ == "__main__":
    main()
