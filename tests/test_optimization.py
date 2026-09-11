"""Unit tests for Pareto optimization framework, candidate scoring, and ASCII visualizer."""

import pytest
from src.evaluation.optimization import (
    OptimizationPriority,
    OptimizationReport,
    ParetoCandidate,
    ParetoOptimizer,
)
from src.evaluation.visualize import (
    render_optimization_report,
    render_pareto_scatter,
    render_strategy_distribution,
    render_tradeoff_barchart,
)


@pytest.fixture
def sample_candidates():
    return [
        # Baseline Direct: Fast, cheap, moderate accuracy
        ParetoCandidate(
            name="Direct_Baseline",
            accuracy=0.60,
            mean_latency=1.5,
            mean_tokens=150.0,
            mean_calls=1.0,
        ),
        # Dominated Direct: Slower and worse accuracy than Baseline
        ParetoCandidate(
            name="Slow_Direct_Dominated",
            accuracy=0.55,
            mean_latency=3.0,
            mean_tokens=250.0,
            mean_calls=1.0,
        ),
        # Self-Consistency: Medium accuracy, medium cost
        ParetoCandidate(
            name="Self_Consistency_k3",
            accuracy=0.75,
            mean_latency=4.5,
            mean_tokens=450.0,
            mean_calls=3.0,
        ),
        # Multi-Agent Debate: Highest accuracy, highest cost
        ParetoCandidate(
            name="Debate_2Rounds",
            accuracy=0.90,
            mean_latency=12.0,
            mean_tokens=1200.0,
            mean_calls=6.0,
        ),
        # Adaptive Router: High accuracy close to debate, but lower average cost
        ParetoCandidate(
            name="Adaptive_Balanced",
            accuracy=0.85,
            mean_latency=4.0,
            mean_tokens=400.0,
            mean_calls=2.2,
        ),
    ]


def test_pareto_dominance(sample_candidates):
    optimizer = ParetoOptimizer()
    baseline = sample_candidates[0]
    dominated = sample_candidates[1]

    # Baseline (0.60 acc, 1.5s lat, 150 tok) dominates dominated (0.55 acc, 3.0s lat, 250 tok)
    assert optimizer.dominates(baseline, dominated) is True
    assert optimizer.dominates(dominated, baseline) is False


def test_pareto_frontier_computation(sample_candidates):
    optimizer = ParetoOptimizer()
    frontier = optimizer.compute_pareto_frontier(sample_candidates)

    frontier_names = {c.name for c in frontier}
    # "Slow_Direct_Dominated" must NOT be in frontier
    assert "Slow_Direct_Dominated" not in frontier_names
    assert "Direct_Baseline" in frontier_names
    assert "Debate_2Rounds" in frontier_names
    assert "Adaptive_Balanced" in frontier_names

    # Check flag consistency
    for c in sample_candidates:
        if c.name in frontier_names:
            assert c.is_pareto_optimal is True
        else:
            assert c.is_pareto_optimal is False


def test_optimization_priorities(sample_candidates):
    optimizer = ParetoOptimizer()

    # 1. Quality First should favor highest accuracy (Debate_2Rounds or Adaptive_Balanced)
    report_quality = optimizer.optimize(
        sample_candidates.copy(), priority=OptimizationPriority.QUALITY_FIRST
    )
    assert report_quality.recommended_candidate.accuracy >= 0.85

    # 2. Budget Constrained should favor fast, cheap configurations
    report_budget = optimizer.optimize(
        sample_candidates.copy(), priority=OptimizationPriority.BUDGET_CONSTRAINED
    )
    assert report_budget.recommended_candidate.mean_calls <= 2.5
    assert report_budget.recommended_candidate.mean_latency <= 4.0


def test_empty_candidates_raises():
    optimizer = ParetoOptimizer()
    with pytest.raises(ValueError, match="cannot be empty"):
        optimizer.optimize([])


def test_ascii_visualizers(sample_candidates):
    optimizer = ParetoOptimizer()
    report = optimizer.optimize(sample_candidates, priority=OptimizationPriority.BALANCED)

    # 1. Scatter Plot
    scatter = render_pareto_scatter(report.all_candidates, x_metric="mean_calls", y_metric="accuracy")
    assert "PARETO FRONTIER SCATTER" in scatter
    assert "Legend:" in scatter
    assert "*" in scatter

    # 2. Trade-off Bar Chart
    barchart = render_tradeoff_barchart(report.all_candidates)
    assert "MULTI-OBJECTIVE TRADE-OFF COMPARISON" in barchart
    assert "Direct_Baseline" in barchart
    assert "Accuracy:" in barchart

    # 3. Strategy Distribution Bar
    dist_bar = render_strategy_distribution({"DIRECT": 60, "SELF_CONSISTENCY": 25, "MULTI_AGENT_DEBATE": 15})
    assert "ADAPTIVE STRATEGY ROUTING DISTRIBUTION" in dist_bar
    assert "Direct Mode:             60" in dist_bar
    assert "60.0%" in dist_bar

    # 4. Optimization Report
    report_str = render_optimization_report(report)
    assert "PARETO MULTI-OBJECTIVE OPTIMIZATION REPORT" in report_str
    assert "RECOMMENDED CONFIGURATION:" in report_str
