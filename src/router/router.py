from typing import Any, Dict, Optional

from src.provider.base import LLMProvider
from src.router.estimator import ConfidenceEstimator
from src.router.models import (
    DifficultyLevel,
    ReasoningStrategy,
    RoutedResponse,
)


class AdaptiveRouter:
    """Adaptive router that dynamically evaluates query difficulty and routes to the optimal reasoning strategy."""

    def __init__(self, provider: LLMProvider, router_config: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.config = router_config or {}
        self.threshold_high = float(self.config.get("confidence_threshold_high", 0.80))
        self.threshold_low = float(self.config.get("confidence_threshold_low", 0.50))
        self.estimator = ConfidenceEstimator(provider)

    def route_and_solve(
        self,
        question: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RoutedResponse:
        """Evaluates query difficulty, routes accordingly, and returns structured response."""
        # 1. Initial pass: generate proposed answer and estimate confidence/difficulty
        answer, explanation, assessment, initial_resp = self.estimator.assess_and_solve(
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        score = assessment.score
        call_count = 1
        total_token_usage = initial_resp.token_usage
        total_latency = initial_resp.latency_seconds

        # 2. Dynamic Routing Decision based on Configurable Thresholds & Assessed Difficulty
        if assessment.difficulty == DifficultyLevel.HARD or score < self.threshold_low:
            # Mode 3: MULTI_AGENT_DEBATE (Foundation flagged for Phase 5)
            strategy = ReasoningStrategy.MULTI_AGENT_DEBATE
            difficulty = DifficultyLevel.HARD
            metadata = {
                "route_decision": "MULTI_AGENT_DEBATE",
                "route_reason": f"High difficulty or low confidence ({score:.2f} < {self.threshold_low:.2f}). Requires multi-agent debate.",
                "additional_agents_called": 0,
                "needs_further_reasoning": True,
                **initial_resp.metadata,
            }

        elif assessment.difficulty == DifficultyLevel.UNCERTAIN or score < self.threshold_high:
            # Mode 2: SELF_CONSISTENCY (Foundation flagged for Phase 4)
            strategy = ReasoningStrategy.SELF_CONSISTENCY
            difficulty = DifficultyLevel.UNCERTAIN
            metadata = {
                "route_decision": "SELF_CONSISTENCY",
                "route_reason": f"Uncertain problem or moderate confidence ({score:.2f} in [{self.threshold_low:.2f}, {self.threshold_high:.2f})). Recommend self-consistency sampling.",
                "additional_agents_called": 0,
                "needs_further_reasoning": True,
                **initial_resp.metadata,
            }

        else:
            # Mode 1: DIRECT (Easy query + High confidence)
            strategy = ReasoningStrategy.DIRECT
            difficulty = DifficultyLevel.EASY
            metadata = {
                "route_decision": "DIRECT",
                "route_reason": f"High confidence ({score:.2f} >= {self.threshold_high:.2f}) on EASY query. No additional compute needed.",
                "additional_agents_called": 0,
                **initial_resp.metadata,
            }

        cost = self.provider.calculate_cost(total_token_usage)

        return RoutedResponse(
            answer=answer,
            explanation=explanation,
            strategy=strategy,
            difficulty=difficulty,
            confidence=assessment,
            call_count=call_count,
            model=initial_resp.model,
            provider=self.provider.provider_name,
            token_usage=total_token_usage,
            latency_seconds=round(total_latency, 4),
            external_api_cost=cost,
            metadata=metadata,
        )
