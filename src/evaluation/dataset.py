"""Curated reasoning benchmark dataset and objective verification utilities."""

import re
from typing import List, Optional
from pydantic import BaseModel, Field


class BenchmarkItem(BaseModel):
    """A benchmark reasoning query with objective ground truth."""
    id: str
    question: str
    ground_truth: str
    difficulty_label: str  # EASY, MEDIUM, HARD
    category: str          # arithmetic, multi_step_math, logic
    notes: Optional[str] = None


BENCHMARK_DATASET: List[BenchmarkItem] = [
    # EASY (Single-step arithmetic or direct logic)
    BenchmarkItem(
        id="gsm8k_easy_01",
        question="A bakery sells cupcakes for $3 each. If Sarah buys 7 cupcakes, how much does she spend in total?",
        ground_truth="21",
        difficulty_label="EASY",
        category="arithmetic",
        notes="Simple multiplication: 3 * 7 = 21",
    ),
    BenchmarkItem(
        id="gsm8k_easy_02",
        question="Tom had 45 marbles. He gave 18 marbles to his brother and 12 marbles to his friend. How many marbles does Tom have left?",
        ground_truth="15",
        difficulty_label="EASY",
        category="arithmetic",
        notes="45 - 18 - 12 = 15",
    ),
    BenchmarkItem(
        id="gsm8k_easy_03",
        question="If a train travels at a constant speed of 60 miles per hour, how many miles does it travel in 3.5 hours?",
        ground_truth="210",
        difficulty_label="EASY",
        category="arithmetic",
        notes="60 * 3.5 = 210",
    ),
    # MEDIUM (Multi-step math with trick conditions / self-consistency candidates)
    BenchmarkItem(
        id="gsm8k_med_01",
        question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
        ground_truth="5",
        difficulty_label="MEDIUM",
        category="multi_step_math",
        notes="x + (x + 1.00) = 1.10 -> 2x = 0.10 -> x = 0.05 dollars = 5 cents",
    ),
    BenchmarkItem(
        id="gsm8k_med_02",
        question="If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        ground_truth="5",
        difficulty_label="MEDIUM",
        category="multi_step_math",
        notes="1 machine takes 5 minutes to make 1 widget; 100 machines making 100 widgets take 5 minutes.",
    ),
    BenchmarkItem(
        id="gsm8k_med_03",
        question="A store offers a 20% discount on a $50 jacket. After the discount, an 8% sales tax is added to the discounted price. What is the final price in dollars?",
        ground_truth="43.20",
        difficulty_label="MEDIUM",
        category="multi_step_math",
        notes="Discounted price: 50 * 0.80 = 40. Tax: 40 * 1.08 = 43.20",
    ),
    # HARD (Multi-agent debate / deep multi-step constraints)
    BenchmarkItem(
        id="gsm8k_hard_01",
        question="In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?",
        ground_truth="47",
        difficulty_label="HARD",
        category="logic",
        notes="Since it doubles daily, on day 47 it covers exactly half.",
    ),
    BenchmarkItem(
        id="gsm8k_hard_02",
        question="Three friends (Alex, Blake, and Casey) split a dinner bill. Alex pays 1/3 of the total bill. Blake pays $15 more than Alex. Casey pays the remaining $25. What was the total dinner bill in dollars?",
        ground_truth="120",
        difficulty_label="HARD",
        category="multi_step_math",
        notes="T/3 + (T/3 + 15) + 25 = T -> 2T/3 + 40 = T -> T/3 = 40 -> T = 120",
    ),
    BenchmarkItem(
        id="gsm8k_hard_03",
        question="A car travels from Town A to Town B at an average speed of 30 mph, and returns along the exact same route from Town B to Town A at an average speed of 60 mph. What is the average speed of the round trip in mph?",
        ground_truth="40",
        difficulty_label="HARD",
        category="multi_step_math",
        notes="Harmonic mean: 2 * 30 * 60 / (30 + 60) = 3600 / 90 = 40 mph.",
    ),
    BenchmarkItem(
        id="gsm8k_hard_04",
        question="Farmer Brown has chickens and cows. Together, the animals have 35 heads and 94 feet. How many cows does Farmer Brown have?",
        ground_truth="12",
        difficulty_label="HARD",
        category="multi_step_math",
        notes="c + w = 35, 2c + 4w = 94 -> 2c + 2w = 70 -> 2w = 24 -> w = 12 cows.",
    ),
]


def extract_numerical_answer(text: str) -> Optional[float]:
    """Extracts the final numerical value from a reasoning text or answer string."""
    if not text:
        return None

    # Check for FINAL ANSWER: tag first
    match = re.search(r"FINAL ANSWER:\s*([^\n\r]+)", text, re.IGNORECASE)
    candidate_text = match.group(1).strip() if match else text

    # Strip currency signs, commas, and percentage signs
    candidate_text = candidate_text.replace("$", "").replace(",", "").replace("%", "").strip()

    # Search for numbers (decimals or integers)
    # Match numbers like 43.20, 120, -5, etc.
    numbers = re.findall(r"[-+]?\d*\.?\d+", candidate_text)
    if numbers:
        try:
            # Prefer the last number mentioned in the final answer section
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def is_answer_correct(predicted: str, ground_truth: str, tolerance: float = 1e-2) -> bool:
    """Evaluates if the predicted answer matches the ground truth canonically or numerically."""
    if not predicted or not ground_truth:
        return False

    pred_num = extract_numerical_answer(predicted)
    gt_num = extract_numerical_answer(ground_truth)

    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) <= tolerance

    # Fallback to normalized substring / exact string match
    norm_pred = re.sub(r"[^\w\s]", "", predicted.lower()).strip()
    norm_gt = re.sub(r"[^\w\s]", "", ground_truth.lower()).strip()

    if norm_gt in norm_pred:
        return True

    return False


def get_benchmark_dataset(
    limit: Optional[int] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[BenchmarkItem]:
    """Retrieves filtered subset of the benchmark dataset."""
    items = list(BENCHMARK_DATASET)
    if category:
        items = [item for item in items if item.category.lower() == category.lower()]
    if difficulty:
        items = [item for item in items if item.difficulty_label.upper() == difficulty.upper()]
    if limit is not None:
        items = items[:limit]
    return items
