from typing import Any, Dict, Optional

from src.provider.base import LLMProvider
from src.provider.models import TokenUsage
from src.reasoning.consistency import SelfConsistencyReasoner
from src.reasoning.judge import DebateWithJudgePipeline
from src.router.estimator import ConfidenceEstimator
from src.router.models import (
    DifficultyLevel,
    ReasoningStrategy,
    RoutedResponse,
)


class AdaptiveRouter:
    """Adaptive router that dynamically evaluates query difficulty and routes to the optimal reasoning strategy:
    - Mode 1 (EASY): Direct single-model response (1 call)
    - Mode 2 (UNCERTAIN): Self-consistency sampling and consensus voting (1 + N calls)
    - Mode 3 (HARD): Multi-agent debate cross-examination + Supreme Judge adjudication (1 + N*R + 1 calls)
    """

    def __init__(
        self,
        provider: LLMProvider,
        router_config: Optional[Dict[str, Any]] = None,
        debate_config: Optional[Dict[str, Any]] = None,
        judge_config: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.config = router_config or {}
        self.threshold_high = float(self.config.get("confidence_threshold_high", 0.80))
        self.threshold_low = float(self.config.get("confidence_threshold_low", 0.50))
        self.consistency_samples = int(self.config.get("consistency_samples", 3))
        self.sample_temperature = float(self.config.get("sample_temperature", 0.7))

        self.estimator = ConfidenceEstimator(provider)
        self.consistency_reasoner = SelfConsistencyReasoner(provider)
        self.debate_pipeline = DebateWithJudgePipeline(
            provider, debate_config=debate_config, judge_config=judge_config
        )

    def route_and_solve(
        self,
        question: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RoutedResponse:
        """Evaluates query difficulty, routes accordingly across the 3 modes, and returns structured response."""
        # 1. Initial pass: generate proposed answer and estimate confidence/difficulty
        initial_answer, initial_explanation, assessment, initial_resp = self.estimator.assess_and_solve(
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        score = assessment.score
        initial_usage = initial_resp.token_usage
        initial_latency = initial_resp.latency_seconds

        transcript = None
        verdict = None
        sc_result = None

        # 2. Dynamic Routing Decision across the 3 Modes
        if assessment.difficulty == DifficultyLevel.HARD or score < self.threshold_low:
            # Mode 3: MULTI_AGENT_DEBATE + SUPREME JUDGE
            strategy = ReasoningStrategy.MULTI_AGENT_DEBATE
            difficulty = DifficultyLevel.HARD

            pipeline_result = self.debate_pipeline.run(question)
            final_answer = pipeline_result.final_answer
            final_explanation = pipeline_result.explanation
            transcript = pipeline_result.transcript
            verdict = pipeline_result.verdict

            call_count = 1 + pipeline_result.total_calls
            total_usage = TokenUsage(
                prompt_tokens=initial_usage.prompt_tokens + pipeline_result.total_token_usage.prompt_tokens,
                completion_tokens=initial_usage.completion_tokens + pipeline_result.total_token_usage.completion_tokens,
                total_tokens=initial_usage.total_tokens + pipeline_result.total_token_usage.total_tokens,
            )
            total_latency = initial_latency + pipeline_result.total_latency_seconds

            metadata = {
                "route_decision": "MULTI_AGENT_DEBATE",
                "route_reason": f"High difficulty or low confidence ({score:.2f} < {self.threshold_low:.2f}). Executed full debate and judge pipeline.",
                "winning_agent": verdict.winning_agent,
                "confidence_in_verdict": verdict.confidence_in_verdict,
                "consensus_reached_in_debate": transcript.consensus_reached,
                "identified_flaws": verdict.identified_flaws,
                **initial_resp.metadata,
            }

        elif assessment.difficulty == DifficultyLevel.UNCERTAIN or score < self.threshold_high:
            # Mode 2: SELF_CONSISTENCY (Multi-path sampling & consensus voting)
            strategy = ReasoningStrategy.SELF_CONSISTENCY
            difficulty = DifficultyLevel.UNCERTAIN

            sc_result = self.consistency_reasoner.sample_and_vote(
                question=question,
                num_samples=self.consistency_samples,
                temperature=self.sample_temperature,
                max_tokens=max_tokens,
            )

            final_answer = sc_result.final_answer
            final_explanation = sc_result.final_explanation
            call_count = 1 + sc_result.num_samples

            total_usage = TokenUsage(
                prompt_tokens=initial_usage.prompt_tokens + sc_result.token_usage.prompt_tokens,
                completion_tokens=initial_usage.completion_tokens + sc_result.token_usage.completion_tokens,
                total_tokens=initial_usage.total_tokens + sc_result.token_usage.total_tokens,
            )
            total_latency = initial_latency + sc_result.latency_seconds

            metadata = {
                "route_decision": "SELF_CONSISTENCY",
                "route_reason": f"Uncertain query or moderate confidence ({score:.2f} in [{self.threshold_low:.2f}, {self.threshold_high:.2f})). Majority voting applied.",
                "num_samples": sc_result.num_samples,
                "agreement_score": sc_result.agreement_score,
                "agreement_distribution": sc_result.agreement_distribution,
                "candidate_answers": [c.answer for c in sc_result.candidates],
                **initial_resp.metadata,
            }

        else:
            # Mode 1: DIRECT (Easy query + High confidence -> 1 model call)
            strategy = ReasoningStrategy.DIRECT
            difficulty = DifficultyLevel.EASY
            final_answer = initial_answer
            final_explanation = initial_explanation
            call_count = 1
            total_usage = initial_usage
            total_latency = initial_latency

            metadata = {
                "route_decision": "DIRECT",
                "route_reason": f"High confidence ({score:.2f} >= {self.threshold_high:.2f}) on EASY query. No additional compute needed.",
                "additional_agents_called": 0,
                **initial_resp.metadata,
            }

        cost = self.provider.calculate_cost(total_usage)

        return RoutedResponse(
            answer=final_answer,
            explanation=final_explanation,
            strategy=strategy,
            difficulty=difficulty,
            confidence=assessment,
            call_count=call_count,
            model=initial_resp.model,
            provider=self.provider.provider_name,
            token_usage=total_usage,
            latency_seconds=round(total_latency, 4),
            external_api_cost=cost,
            transcript=transcript,
            verdict=verdict,
            self_consistency_result=sc_result,
            metadata=metadata,
        )
