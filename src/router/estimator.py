import re
from typing import Optional, Tuple

from src.provider.base import LLMProvider
from src.provider.models import LLMRequest, LLMResponse
from src.router.models import ConfidenceAssessment, DifficultyLevel


class ConfidenceEstimator:
    """Estimates model confidence and difficulty for an incoming question."""

    SYSTEM_PROMPT = (
        "You are an analytical reasoning assistant. "
        "Analyze and solve the user's question. "
        "You must format your response strictly with the following sections:\n"
        "EXPLANATION: <step-by-step reasoning>\n"
        "FINAL ANSWER: <concise answer>\n"
        "DIFFICULTY: <EASY | UNCERTAIN | HARD>\n"
        "CONFIDENCE: <number between 0.0 and 1.0 representing your certainty in the answer>\n"
        "CONFIDENCE REASON: <brief one-sentence reason for this confidence rating>"
    )

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def assess_and_solve(
        self,
        question: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str, ConfidenceAssessment, LLMResponse]:
        """Runs the initial model pass to solve the question and assess confidence.

        Returns:
            Tuple of (answer, explanation, ConfidenceAssessment, raw_LLMResponse)
        """
        request = LLMRequest(
            prompt=f"Question: {question}",
            system_prompt=self.SYSTEM_PROMPT,
            temperature=temperature if temperature is not None else self.provider.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.provider.default_max_tokens,
            timeout=self.provider.default_timeout,
        )

        response = self.provider.generate(request)
        answer, explanation, assessment = self.parse_assessment(response.text)
        return answer, explanation, assessment, response

    def parse_assessment(self, text: str) -> Tuple[str, str, ConfidenceAssessment]:
        """Extracts answer, explanation, and confidence metrics from model output with resilient fallbacks."""
        # 1. Parse Confidence Score
        conf_match = re.search(
            r"CONFIDENCE:\s*(\d+(?:\.\d+)?%?)", text, re.IGNORECASE
        )
        if conf_match:
            raw_conf = conf_match.group(1).strip()
            try:
                if raw_conf.endswith("%"):
                    score = float(raw_conf[:-1]) / 100.0
                else:
                    val = float(raw_conf)
                    # If model outputs 0-100 scale (e.g. 95 instead of 0.95)
                    score = val / 100.0 if val > 1.0 else val
            except ValueError:
                score = 0.75
        else:
            # Fallback heuristic: check for uncertainty markers
            lower_text = text.lower()
            if any(w in lower_text for w in ["uncertain", "not sure", "possibly", "might be", "hard to say"]):
                score = 0.45
            elif any(w in lower_text for w in ["definitely", "certainly", "trivial", "clearly", "obviously"]):
                score = 0.95
            else:
                score = 0.75

        # Clamp between 0.0 and 1.0
        score = max(0.0, min(1.0, round(score, 3)))

        # 2. Parse Difficulty
        diff_match = re.search(
            r"DIFFICULTY:\s*(EASY|UNCERTAIN|HARD)", text, re.IGNORECASE
        )
        if diff_match:
            diff_str = diff_match.group(1).upper()
            try:
                difficulty = DifficultyLevel(diff_str)
            except ValueError:
                difficulty = self._infer_difficulty(score)
        else:
            difficulty = self._infer_difficulty(score)

        # 3. Categorical Confidence Level
        if score >= 0.80:
            level = "HIGH"
        elif score >= 0.50:
            level = "MEDIUM"
        else:
            level = "LOW"

        # 4. Parse Confidence Reason
        reason_match = re.search(
            r"CONFIDENCE REASON:\s*(.*)", text, re.IGNORECASE
        )
        justification = (
            reason_match.group(1).strip()
            if reason_match
            else f"Model self-assessed confidence at {score:.2f} based on problem clarity."
        )

        # 5. Parse Final Answer
        ans_match = re.search(
            r"FINAL ANSWER:\s*(.*?)(?=\n(?:DIFFICULTY|CONFIDENCE|CONFIDENCE REASON):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if ans_match:
            answer = ans_match.group(1).strip()
        else:
            # Fallback: take line before CONFIDENCE or last non-empty line
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            answer = lines[-1] if lines else text.strip()

        # 6. Parse Explanation
        exp_match = re.search(
            r"EXPLANATION:\s*(.*?)(?=\n(?:FINAL ANSWER|DIFFICULTY|CONFIDENCE):|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if exp_match:
            explanation = exp_match.group(1).strip()
        else:
            explanation = text.strip()

        assessment = ConfidenceAssessment(
            score=score,
            level=level,
            difficulty=difficulty,
            justification=justification,
        )

        return answer, explanation, assessment

    def _infer_difficulty(self, score: float) -> DifficultyLevel:
        """Infers difficulty level from the confidence score if not explicitly specified."""
        if score >= 0.80:
            return DifficultyLevel.EASY
        elif score >= 0.50:
            return DifficultyLevel.UNCERTAIN
        else:
            return DifficultyLevel.HARD
