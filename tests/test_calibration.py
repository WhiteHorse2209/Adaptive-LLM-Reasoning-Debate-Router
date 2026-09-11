"""Unit tests for Confidence Calibration (ECE) and 8-Category Failure Taxonomy."""

import pytest
from src.evaluation.calibration import compute_calibration, CalibrationReport
from src.evaluation.failure_analysis import (
    CATEGORY_DESCRIPTIONS,
    FailureAnalyzer,
    FailureCategory,
    FailureMode,
)
from src.evaluation.run_calibration import (
    render_ascii_reliability_diagram,
    render_failure_mode_summary,
)


def test_perfect_calibration():
    # 5 samples with 0.2 confidence (1 correct = 20% empirical acc)
    # 5 samples with 0.8 confidence (4 correct = 80% empirical acc)
    confidences = [0.2] * 5 + [0.8] * 5
    labels = [True, False, False, False, False] + [True, True, True, True, False]

    report = compute_calibration(confidences, labels, num_bins=5)
    assert isinstance(report, CalibrationReport)
    assert report.total_samples == 10
    assert report.overall_accuracy == 0.5
    assert report.mean_confidence == 0.5
    assert report.expected_calibration_error == 0.0
    assert report.maximum_calibration_error == 0.0


def test_overconfidence_and_ece():
    # High confidence, but all wrong
    confidences = [0.95, 0.90, 0.85]
    labels = [False, False, False]

    report = compute_calibration(confidences, labels, num_bins=5)
    assert report.overall_accuracy == 0.0
    assert report.is_overconfident is True
    assert report.expected_calibration_error > 0.8
    assert report.brier_score > 0.7


def test_calibration_input_validation():
    with pytest.raises(ValueError, match="identical length"):
        compute_calibration([0.5, 0.7], [True])

    with pytest.raises(ValueError, match="empty inputs"):
        compute_calibration([], [])


def test_all_8_failure_mode_classifications():
    analyzer = FailureAnalyzer()

    # 1. INITIAL_WRONG_DEBATE_CORRECT (Debate recovered from error)
    cat1 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=True,
        difficulty="HARD",
        agent1_correct=False,
        agent2_correct=True,
    )
    assert cat1 == FailureCategory.INITIAL_WRONG_DEBATE_CORRECT

    # 2. INITIAL_WRONG_DEBATE_WRONG (Intractable fallacy persisted)
    cat2 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=False,
        difficulty="HARD",
        agent1_correct=False,
        agent2_correct=False,
    )
    assert cat2 == FailureCategory.INITIAL_WRONG_DEBATE_WRONG

    # 3. INITIAL_CORRECT_DEBATE_CORRECT (Truth preserved)
    cat3 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=True,
        difficulty="HARD",
        agent1_correct=True,
        agent2_correct=True,
    )
    assert cat3 == FailureCategory.INITIAL_CORRECT_DEBATE_CORRECT

    # 4. INITIAL_CORRECT_DEBATE_WRONG (Harmful concession)
    cat4 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=False,
        difficulty="HARD",
        agent1_correct=True,
        agent2_correct=False,
        concession_to_error=True,
    )
    assert cat4 == FailureCategory.INITIAL_CORRECT_DEBATE_WRONG

    # 5. UNDER_ROUTING (Direct used on hard query and failed)
    cat5 = analyzer.classify(
        strategy="DIRECT",
        final_correct=False,
        difficulty="HARD",
    )
    assert cat5 == FailureCategory.UNDER_ROUTING

    # 6. OVER_ROUTING (Debate invoked on easy query)
    cat6 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=True,
        difficulty="EASY",
    )
    assert cat6 == FailureCategory.OVER_ROUTING

    # 7. JUDGE_SELECTION_ERROR (Agent had correct answer, judge chose wrong)
    cat7 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=False,
        difficulty="HARD",
        agent1_correct=True,
        agent2_correct=False,
        judge_selected_correct=False,
    )
    assert cat7 == FailureCategory.JUDGE_SELECTION_ERROR

    # 8. UNANIMOUS_HALLUCINATION (Both agents agreed on same incorrect premise)
    cat8 = analyzer.classify(
        strategy="MULTI_AGENT_DEBATE",
        final_correct=False,
        difficulty="HARD",
        unanimous_wrong_premise=True,
    )
    assert cat8 == FailureCategory.UNANIMOUS_HALLUCINATION


def test_failure_analyzer_summary_generation():
    analyzer = FailureAnalyzer()
    raw_traces = [
        {"query_id": "q1", "strategy": "DIRECT", "is_correct": False, "difficulty": "HARD"},
        {"query_id": "q2", "strategy": "MULTI_AGENT_DEBATE", "is_correct": True, "difficulty": "EASY"},
        {
            "query_id": "q3",
            "strategy": "MULTI_AGENT_DEBATE",
            "is_correct": False,
            "difficulty": "HARD",
            "unanimous_wrong_premise": True,
        },
    ]

    summary = analyzer.analyze_records(raw_traces)
    assert summary.total_analyzed == 3
    assert summary.category_counts[FailureCategory.UNDER_ROUTING.value] == 1
    assert summary.category_counts[FailureCategory.OVER_ROUTING.value] == 1
    assert summary.category_counts[FailureCategory.UNANIMOUS_HALLUCINATION.value] == 1
    assert len(summary.records) == 3


def test_ascii_renderers():
    confidences = [0.1, 0.9]
    labels = [False, True]
    report = compute_calibration(confidences, labels, num_bins=2)

    diagram = render_ascii_reliability_diagram(report)
    assert "CONFIDENCE RELIABILITY DIAGRAM" in diagram
    assert "ECE:" in diagram

    analyzer = FailureAnalyzer()
    summary = analyzer.analyze_records([
        {"query_id": "q1", "strategy": "DIRECT", "is_correct": False, "difficulty": "HARD"}
    ])
    summary_text = render_failure_mode_summary(summary)
    assert "8-CATEGORY FAILURE & BEHAVIOR TAXONOMY" in summary_text
    assert "UNDER_ROUTING" in summary_text
