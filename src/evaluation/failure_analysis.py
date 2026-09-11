"""Comprehensive 8-category failure mode and behavior taxonomy for reasoning systems."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """8-category diagnostic taxonomy classifying debate and routing behaviors."""
    INITIAL_WRONG_DEBATE_CORRECT = "INITIAL_WRONG_DEBATE_CORRECT"
    INITIAL_WRONG_DEBATE_WRONG = "INITIAL_WRONG_DEBATE_WRONG"
    INITIAL_CORRECT_DEBATE_CORRECT = "INITIAL_CORRECT_DEBATE_CORRECT"
    INITIAL_CORRECT_DEBATE_WRONG = "INITIAL_CORRECT_DEBATE_WRONG"
    UNDER_ROUTING = "UNDER_ROUTING"
    OVER_ROUTING = "OVER_ROUTING"
    JUDGE_SELECTION_ERROR = "JUDGE_SELECTION_ERROR"
    UNANIMOUS_HALLUCINATION = "UNANIMOUS_HALLUCINATION"


# Convenient alias
FailureMode = FailureCategory


CATEGORY_DESCRIPTIONS: Dict[FailureCategory, str] = {
    FailureCategory.INITIAL_WRONG_DEBATE_CORRECT: "Successful error recovery: agent began with flawed reasoning but corrected via debate cross-examination.",
    FailureCategory.INITIAL_WRONG_DEBATE_WRONG: "Intractable fallacy: flawed initial reasoning persisted across all debate rounds.",
    FailureCategory.INITIAL_CORRECT_DEBATE_CORRECT: "Truth preservation: sound initial reasoning withstood scrutiny and was confirmed by adjudication.",
    FailureCategory.INITIAL_CORRECT_DEBATE_WRONG: "Harmful concession: correct agent yielded under peer pressure to flawed counter-arguments.",
    FailureCategory.UNDER_ROUTING: "Under-routing: router selected Direct mode on a complex question, causing reasoning failure.",
    FailureCategory.OVER_ROUTING: "Over-routing: router invoked expensive multi-agent debate on an easy question, wasting compute.",
    FailureCategory.JUDGE_SELECTION_ERROR: "Adjudication failure: at least one agent offered the correct answer, but the Judge selected the flawed proposal.",
    FailureCategory.UNANIMOUS_HALLUCINATION: "Unanimous delusion: both agents independently hallucinated or agreed upon the identical incorrect premise.",
}


class FailureRecord(BaseModel):
    """Diagnostic record for a single classified query trace."""
    query_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    strategy: str
    difficulty: str
    category: FailureCategory
    description: str
    is_correct: bool


class FailureAnalysisSummary(BaseModel):
    """Aggregate distribution and breakdown across failure taxonomy categories."""
    total_analyzed: int
    category_counts: Dict[str, int]
    category_percentages: Dict[str, float]
    records: List[FailureRecord] = Field(default_factory=list)


class FailureAnalyzer:
    """Classifies reasoning traces into the 8-category taxonomy."""

    @staticmethod
    def classify(
        strategy: str,
        final_correct: bool,
        difficulty: str = "MEDIUM",
        agent1_correct: Optional[bool] = None,
        agent2_correct: Optional[bool] = None,
        judge_selected_correct: Optional[bool] = None,
        unanimous_wrong_premise: bool = False,
        concession_to_error: bool = False,
    ) -> FailureCategory:
        """Determines the diagnostic classification for a reasoning interaction.
        
        Args:
            strategy: 'DIRECT', 'SELF_CONSISTENCY', or 'MULTI_AGENT_DEBATE' / 'DEBATE'
            final_correct: Whether the final system answer matched ground truth
            difficulty: 'EASY', 'MEDIUM', 'HARD', or 'UNCERTAIN'
            agent1_correct: Initial correctness of Agent 1 (for debate)
            agent2_correct: Initial correctness of Agent 2 (for debate)
            judge_selected_correct: If judge favored the correct debater
            unanimous_wrong_premise: If both agents agreed on same incorrect logic
            concession_to_error: If initially correct agent surrendered to incorrect argument
        """
        strat_upper = strategy.upper()
        diff_upper = difficulty.upper()

        # Check for unanimous hallucination first
        if unanimous_wrong_premise and not final_correct:
            return FailureCategory.UNANIMOUS_HALLUCINATION

        # Non-debate routing failures
        if "DEBATE" not in strat_upper:
            if not final_correct:
                if diff_upper in ["HARD", "UNCERTAIN", "MEDIUM"]:
                    return FailureCategory.UNDER_ROUTING
                return FailureCategory.UNDER_ROUTING
            # If direct succeeded on hard, it's efficient; if on easy, expected
            return FailureCategory.INITIAL_CORRECT_DEBATE_CORRECT

        # Debate routing & interaction modes
        if diff_upper == "EASY" and final_correct:
            return FailureCategory.OVER_ROUTING

        # Debate with explicit agent states
        if agent1_correct is not None and agent2_correct is not None:
            both_wrong = (not agent1_correct) and (not agent2_correct)
            both_correct = agent1_correct and agent2_correct
            one_correct = (agent1_correct and not agent2_correct) or (agent2_correct and not agent1_correct)

            if both_wrong:
                if final_correct:
                    return FailureCategory.INITIAL_WRONG_DEBATE_CORRECT
                if unanimous_wrong_premise:
                    return FailureCategory.UNANIMOUS_HALLUCINATION
                return FailureCategory.INITIAL_WRONG_DEBATE_WRONG

            if one_correct:
                if final_correct:
                    return FailureCategory.INITIAL_WRONG_DEBATE_CORRECT
                else:
                    if concession_to_error:
                        return FailureCategory.INITIAL_CORRECT_DEBATE_WRONG
                    if judge_selected_correct is False:
                        return FailureCategory.JUDGE_SELECTION_ERROR
                    return FailureCategory.JUDGE_SELECTION_ERROR

            if both_correct:
                if final_correct:
                    return FailureCategory.INITIAL_CORRECT_DEBATE_CORRECT
                else:
                    return FailureCategory.INITIAL_CORRECT_DEBATE_WRONG

        # Fallbacks based on final accuracy
        if final_correct:
            return FailureCategory.INITIAL_CORRECT_DEBATE_CORRECT
        else:
            return FailureCategory.INITIAL_WRONG_DEBATE_WRONG

    def analyze_records(
        self,
        raw_traces: List[Dict[str, Any]],
    ) -> FailureAnalysisSummary:
        """Analyzes a list of trace dictionaries and generates an aggregate report."""
        records: List[FailureRecord] = []
        counts: Dict[str, int] = {cat.value: 0 for cat in FailureCategory}

        for trace in raw_traces:
            category = self.classify(
                strategy=trace.get("strategy", "DIRECT"),
                final_correct=trace.get("is_correct", False),
                difficulty=trace.get("difficulty", "MEDIUM"),
                agent1_correct=trace.get("agent1_correct"),
                agent2_correct=trace.get("agent2_correct"),
                judge_selected_correct=trace.get("judge_selected_correct"),
                unanimous_wrong_premise=trace.get("unanimous_wrong_premise", False),
                concession_to_error=trace.get("concession_to_error", False),
            )

            counts[category.value] += 1
            records.append(
                FailureRecord(
                    query_id=trace.get("query_id", "q_unknown"),
                    question=trace.get("question", ""),
                    ground_truth=trace.get("ground_truth", ""),
                    predicted_answer=trace.get("predicted_answer", ""),
                    strategy=trace.get("strategy", "DIRECT"),
                    difficulty=trace.get("difficulty", "MEDIUM"),
                    category=category,
                    description=CATEGORY_DESCRIPTIONS[category],
                    is_correct=trace.get("is_correct", False),
                )
            )

        total = len(records)
        percentages = {
            cat: round((count / total) * 100.0, 2) if total > 0 else 0.0
            for cat, count in counts.items()
        }

        return FailureAnalysisSummary(
            total_analyzed=total,
            category_counts=counts,
            category_percentages=percentages,
            records=records,
        )
