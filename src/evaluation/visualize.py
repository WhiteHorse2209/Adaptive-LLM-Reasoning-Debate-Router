"""ASCII visualizer for multi-objective trade-offs, Pareto frontiers, and strategy distribution."""

from typing import Dict, List, Optional
from src.evaluation.optimization import OptimizationReport, ParetoCandidate


def render_pareto_scatter(
    candidates: List[ParetoCandidate],
    x_metric: str = "mean_calls",
    y_metric: str = "accuracy",
    width: int = 50,
    height: int = 12,
) -> str:
    """Renders an ASCII 2D scatter plot plotting candidates across two trade-off dimensions."""
    if not candidates:
        return "No candidates to visualize."

    x_vals = [getattr(c, x_metric) for c in candidates]
    y_vals = [getattr(c, y_metric) for c in candidates]

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    x_span = max_x - min_x if max_x > min_x else 1.0
    y_span = max_y - min_y if max_y > min_y else 1.0

    # Initialize blank grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Place candidates
    for c in candidates:
        xv = getattr(c, x_metric)
        yv = getattr(c, y_metric)

        col = int(((xv - min_x) / x_span) * (width - 1))
        # Row 0 is top (max_y)
        row = int(((max_y - yv) / y_span) * (height - 1))

        marker = "*" if c.is_pareto_optimal else "o"
        grid[row][col] = marker

    lines = []
    lines.append("\n" + "=" * (width + 16))
    lines.append(f"PARETO FRONTIER SCATTER ({y_metric.upper()} vs {x_metric.upper()})")
    lines.append(f"Legend: [*] Pareto Optimal Frontier | [o] Dominated Candidate")
    lines.append("=" * (width + 16))

    for r in range(height):
        y_val = max_y - (r / (height - 1)) * y_span if height > 1 else max_y
        row_str = "".join(grid[r])
        lines.append(f"{y_val:>6.2f} |{row_str}|")

    lines.append("       +" + "-" * width + "+")
    lines.append(f"       {min_x:<6.2f}" + " " * (width - 14) + f"{max_x:>6.2f}")
    lines.append(f"       {'':<20} {x_metric.upper()} --->")
    return "\n".join(lines)


def render_tradeoff_barchart(candidates: List[ParetoCandidate]) -> str:
    """Renders comparative horizontal ASCII bar charts for candidate strategies."""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("MULTI-OBJECTIVE TRADE-OFF COMPARISON")
    lines.append("=" * 90)

    for c in candidates:
        tag = "[PARETO]" if c.is_pareto_optimal else "[NON-OPT]"
        lines.append(f"\nConfiguration: {c.name:<25} {tag} (Fitness: {c.fitness_score:.4f})")
        
        # Accuracy bar (max 100%)
        acc_bar = "#" * int(round(c.accuracy * 30))
        lines.append(f"  Accuracy: [{acc_bar:<30}] {c.accuracy * 100:>5.1f}%")

        # Calls bar (scaled relative to 10 max)
        calls_bar = "=" * min(int(round(c.mean_calls * 3)), 30)
        lines.append(f"  Calls:    [{calls_bar:<30}] {c.mean_calls:>5.1f} calls/q")

        # Latency
        lines.append(f"  Latency:  {c.mean_latency:>5.2f}s | Tokens: {c.mean_tokens:>5.0f}")

    lines.append("=" * 90)
    return "\n".join(lines)


def render_strategy_distribution(strategy_counts: Dict[str, int]) -> str:
    """Visualizes the routing strategy activation distribution as an ASCII proportion bar."""
    total = sum(strategy_counts.values())
    if total == 0:
        return "No queries routed."

    direct_cnt = strategy_counts.get("DIRECT", 0)
    sc_cnt = strategy_counts.get("SELF_CONSISTENCY", 0)
    debate_cnt = strategy_counts.get("MULTI_AGENT_DEBATE", strategy_counts.get("DEBATE", 0))

    direct_pct = (direct_cnt / total) * 100
    sc_pct = (sc_cnt / total) * 100
    debate_pct = (debate_cnt / total) * 100

    # 40-character bar
    direct_len = int(round((direct_pct / 100) * 40))
    sc_len = int(round((sc_pct / 100) * 40))
    debate_len = 40 - (direct_len + sc_len)
    if debate_len < 0:
        debate_len = 0

    bar = f"[{'D' * direct_len}{'S' * sc_len}{'M' * debate_len}]"

    lines = [
        "\n==========================================================================================",
        "ADAPTIVE STRATEGY ROUTING DISTRIBUTION",
        "==========================================================================================",
        f"Visual Bar: {bar}",
        f"  [D] Direct Mode:           {direct_cnt:>4} queries ({direct_pct:>5.1f}%)",
        f"  [S] Self-Consistency:      {sc_cnt:>4} queries ({sc_pct:>5.1f}%)",
        f"  [M] Multi-Agent Debate:    {debate_cnt:>4} queries ({debate_pct:>5.1f}%)",
        f"  Total Queries Dispatched:  {total:>4}",
        "==========================================================================================",
    ]
    return "\n".join(lines)


def render_optimization_report(report: OptimizationReport) -> str:
    """Formats full optimization summary table with Pareto frontier and recommendations."""
    lines = []
    lines.append("\n==========================================================================================")
    lines.append(f"PARETO MULTI-OBJECTIVE OPTIMIZATION REPORT (Mode: {report.priority_mode.value})")
    lines.append("==========================================================================================")
    lines.append(f"{'RANK':<5} {'CONFIGURATION':<25} {'PARETO':<8} {'ACCURACY':<10} {'CALLS':<8} {'LATENCY':<10} {'TOKENS':<8} {'FITNESS'}")
    lines.append("-" * 90)

    for rank, c in enumerate(report.all_candidates, start=1):
        is_rec = ">>> " if c == report.recommended_candidate else "    "
        is_p = "YES" if c.is_pareto_optimal else "NO"
        lines.append(
            f"{is_rec}{rank:<2} {c.name:<25} {is_p:<8} {c.accuracy * 100:>5.1f}%     {c.mean_calls:<8.1f} {c.mean_latency:<10.2f} {c.mean_tokens:<8.0f} {c.fitness_score:<8.4f}"
        )

    lines.append("=" * 90)
    lines.append(f"RECOMMENDED CONFIGURATION: {report.recommended_candidate.name}")
    lines.append(f"  Accuracy: {report.recommended_candidate.accuracy * 100:.1f}% | Mean Calls: {report.recommended_candidate.mean_calls:.1f}")
    lines.append(f"  Mean Latency: {report.recommended_candidate.mean_latency:.2f}s | Mean Tokens: {report.recommended_candidate.mean_tokens:.0f}")
    lines.append(f"  Overall Fitness Score: {report.recommended_candidate.fitness_score:.4f}")
    lines.append("=" * 90)
    return "\n".join(lines)
