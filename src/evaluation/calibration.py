"""Confidence calibration engine calculating Expected Calibration Error (ECE) and reliability bins."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CalibrationBin(BaseModel):
    """Metrics for a single confidence interval bin."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    correct_count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_error: float = Field(
        ..., description="Absolute difference |empirical_accuracy - mean_confidence|"
    )


class CalibrationReport(BaseModel):
    """Aggregate confidence calibration report including ECE and MCE."""
    bins: List[CalibrationBin]
    total_samples: int
    overall_accuracy: float
    mean_confidence: float
    expected_calibration_error: float = Field(
        ..., description="Weighted average calibration error across non-empty bins (ECE)"
    )
    maximum_calibration_error: float = Field(
        ..., description="Maximum calibration error observed in any single bin (MCE)"
    )
    brier_score: float = Field(
        ..., description="Mean squared error between confidence and binary correctness"
    )
    is_overconfident: bool = Field(
        ..., description="True if mean confidence exceeds overall accuracy"
    )


def compute_calibration(
    confidences: List[float],
    labels: List[bool],
    num_bins: int = 5,
) -> CalibrationReport:
    """Computes Expected Calibration Error (ECE) and reliability bin statistics.
    
    Args:
        confidences: List of predicted confidence scores in [0.0, 1.0].
        labels: List of boolean outcomes (True if predicted answer is correct).
        num_bins: Number of equal-width calibration bins (default 5: 0-0.2, 0.2-0.4, etc.).
    """
    if len(confidences) != len(labels):
        raise ValueError("Confidences and labels must have identical length.")
    if not confidences:
        raise ValueError("Cannot compute calibration on empty inputs.")

    total_samples = len(confidences)
    bin_width = 1.0 / num_bins
    bins_data: List[CalibrationBin] = []

    weighted_ece = 0.0
    max_ce = 0.0
    squared_errors = []

    for i in range(num_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width

        # In last bin, include upper bound 1.0 inclusive
        if i == num_bins - 1:
            indices = [
                idx for idx, c in enumerate(confidences) if lower <= c <= upper
            ]
        else:
            indices = [
                idx for idx, c in enumerate(confidences) if lower <= c < upper
            ]

        sample_count = len(indices)
        if sample_count > 0:
            correct_count = sum(1 for idx in indices if labels[idx])
            mean_conf = sum(confidences[idx] for idx in indices) / sample_count
            emp_acc = correct_count / sample_count
            cal_err = abs(emp_acc - mean_conf)
            weighted_ece += (sample_count / total_samples) * cal_err
            max_ce = max(max_ce, cal_err)
        else:
            correct_count = 0
            mean_conf = (lower + upper) / 2.0
            emp_acc = 0.0
            cal_err = 0.0

        bins_data.append(
            CalibrationBin(
                bin_index=i + 1,
                bin_lower=round(lower, 2),
                bin_upper=round(upper, 2),
                sample_count=sample_count,
                correct_count=correct_count,
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(emp_acc, 4),
                calibration_error=round(cal_err, 4),
            )
        )

    # Calculate overall metrics
    overall_correct = sum(1 for y in labels if y)
    overall_acc = overall_correct / total_samples
    overall_mean_conf = sum(confidences) / total_samples

    for c, y in zip(confidences, labels):
        target = 1.0 if y else 0.0
        squared_errors.append((c - target) ** 2)

    brier_score = sum(squared_errors) / total_samples

    return CalibrationReport(
        bins=bins_data,
        total_samples=total_samples,
        overall_accuracy=round(overall_acc, 4),
        mean_confidence=round(overall_mean_conf, 4),
        expected_calibration_error=round(weighted_ece, 4),
        maximum_calibration_error=round(max_ce, 4),
        brier_score=round(brier_score, 4),
        is_overconfident=(overall_mean_conf > overall_acc),
    )
