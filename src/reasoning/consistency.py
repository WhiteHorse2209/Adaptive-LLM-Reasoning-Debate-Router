from collections import Counter
import re
from typing import Dict, List, Optional, Tuple

from src.provider.base import LLMProvider
from src.provider.models import LLMRequest, TokenUsage
from src.reasoning.models import CandidateSolution, SelfConsistencyResult


class SelfConsistencyReasoner:
    """Self-Consistency reasoning engine (Wang et al., 2022).
    
    Samples multiple independent reasoning paths at non-zero temperature,
    extracts candidate answers, and selects the consensus answer via majority voting.
    """

    SYSTEM_PROMPT = (
        "You are an analytical reasoning assistant. "
        "Solve the following question carefully and step-by-step. "
        "Provide your clear reasoning first. "
        "Then end your response with your exact final answer on its own line in the format: "
        "'FINAL ANSWER: <concise answer>'."
    )

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def sample_and_vote(
        self,
        question: str,
        num_samples: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> SelfConsistencyResult:
        """Generates multiple independent solutions and performs majority voting."""
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        candidates: List[CandidateSolution] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0

        for i in range(1, num_samples + 1):
            request = LLMRequest(
                prompt=f"Question: {question}",
                system_prompt=self.SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else self.provider.default_max_tokens,
                timeout=self.provider.default_timeout,
            )

            resp = self.provider.generate(request)
            raw_answer, explanation = self._parse_response(resp.text)

            cand = CandidateSolution(
                sample_id=i,
                answer=raw_answer,
                explanation=explanation,
                token_usage=resp.token_usage,
                latency_seconds=resp.latency_seconds,
            )
            candidates.append(cand)

            total_prompt_tokens += resp.token_usage.prompt_tokens
            total_completion_tokens += resp.token_usage.completion_tokens
            total_latency += resp.latency_seconds

        # Normalize answers for robust tallying
        normalized_answers = [self.normalize_answer(c.answer) for c in candidates]
        counts = Counter(normalized_answers)

        # Majority Vote
        winning_normalized, max_count = counts.most_common(1)[0]
        agreement_score = round(max_count / num_samples, 3)

        # Retrieve representative original answer and explanation for the winning consensus
        winning_candidate = next(
            c for c, norm in zip(candidates, normalized_answers) if norm == winning_normalized
        )
        final_answer = winning_candidate.answer
        final_explanation = winning_candidate.explanation

        # Format distribution for display
        distribution: Dict[str, int] = dict(counts)

        total_usage = TokenUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        cost = self.provider.calculate_cost(total_usage)

        return SelfConsistencyResult(
            final_answer=final_answer,
            final_explanation=final_explanation,
            agreement_score=agreement_score,
            agreement_distribution=distribution,
            candidates=candidates,
            num_samples=num_samples,
            model=str(getattr(self.provider, "default_model", "unknown") or "unknown"),
            provider=self.provider.provider_name,
            token_usage=total_usage,
            latency_seconds=round(total_latency, 4),
            external_api_cost=cost,
            metadata={
                "strategy": "SELF_CONSISTENCY",
                "temperature": temperature,
                "consensus_answer": final_answer,
                "vote_distribution": distribution,
            },
        )

    def normalize_answer(self, answer: str) -> str:
        """Normalizes candidate answer strings to enable fair clustering/voting."""
        ans = answer.strip().lower()

        # Remove trailing punctuation (periods, commas, exclamation marks)
        ans = re.sub(r"[.,!?;:]+$", "", ans).strip()

        # Remove leading currency symbols or words
        ans = re.sub(r"^\$\s*", "", ans).strip()

        # Normalize numeric representations (e.g., '100.0' -> '100')
        try:
            val = float(ans)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except ValueError:
            pass

        return ans

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """Extracts answer and explanation from model response text."""
        match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
            explanation = text[:match.start()].strip()
            if not explanation:
                explanation = text.strip()
            return final_answer, explanation

        # Fallback if marker is missing
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            return lines[-1], text.strip()
        return text.strip(), text.strip()
