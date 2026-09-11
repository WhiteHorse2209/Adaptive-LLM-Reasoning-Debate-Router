"""Automated benchmark evaluator comparing Direct, Self-Consistency, Always Debate, and Adaptive Router."""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.evaluation.dataset import BenchmarkItem, is_answer_correct
from src.provider.base import LLMProvider
from src.provider.models import TokenUsage
from src.reasoning.consistency import SelfConsistencyReasoner
from src.reasoning.direct import DirectReasoner
from src.reasoning.judge import DebateWithJudgePipeline
from src.router.models import ReasoningStrategy
from src.router.router import AdaptiveRouter
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class QueryEvaluationRecord(BaseModel):
    """Execution trace and outcome for an individual benchmark question."""
    query_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    is_correct: bool
    strategy: str
    calls: int
    token_usage: TokenUsage
    latency_seconds: float
    external_api_cost: float = 0.0


class StrategyEvaluationSummary(BaseModel):
    """Aggregate benchmark metrics for a single reasoning strategy."""
    strategy_name: str
    total_queries: int
    correct_count: int
    accuracy: float
    total_calls: int
    mean_calls_per_query: float
    total_tokens: int
    mean_tokens_per_query: float
    total_latency_seconds: float
    mean_latency_seconds: float
    external_api_cost: float = 0.0
    routing_breakdown: Dict[str, float] = Field(default_factory=dict)
    query_records: List[QueryEvaluationRecord] = Field(default_factory=list)


class BenchmarkEvaluator:
    """Orchestrates comprehensive benchmark evaluations across reasoning modes."""

    def __init__(
        self,
        provider: LLMProvider,
        router_config: Optional[Dict[str, Any]] = None,
        debate_config: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.router_config = router_config or {}
        self.debate_config = debate_config or {}

        self.direct_reasoner = DirectReasoner(provider)
        self.consistency_reasoner = SelfConsistencyReasoner(provider)
        self.debate_pipeline = DebateWithJudgePipeline(provider, debate_config=self.debate_config)
        self.router = AdaptiveRouter(
            provider,
            router_config=self.router_config,
            debate_config=self.debate_config,
        )

    def evaluate_direct(self, items: List[BenchmarkItem]) -> StrategyEvaluationSummary:
        """Evaluates all queries using Single LLM Direct mode."""
        records: List[QueryEvaluationRecord] = []
        logger.info(f"Running DIRECT evaluation on {len(items)} items...")

        for item in items:
            start_t = time.perf_counter()
            resp = self.direct_reasoner.answer(item.question)
            lat = time.perf_counter() - start_t
            correct = is_answer_correct(resp.answer, item.ground_truth)

            records.append(
                QueryEvaluationRecord(
                    query_id=item.id,
                    question=item.question,
                    ground_truth=item.ground_truth,
                    predicted_answer=resp.answer,
                    is_correct=correct,
                    strategy="DIRECT",
                    calls=1,
                    token_usage=resp.token_usage,
                    latency_seconds=round(lat, 4),
                    external_api_cost=resp.external_api_cost,
                )
            )

        return self._aggregate_summary("DIRECT (Single LLM)", records)

    def evaluate_self_consistency(
        self, items: List[BenchmarkItem], num_samples: int = 3
    ) -> StrategyEvaluationSummary:
        """Evaluates all queries using Self-Consistency consensus voting."""
        records: List[QueryEvaluationRecord] = []
        logger.info(f"Running SELF-CONSISTENCY evaluation (N={num_samples}) on {len(items)} items...")

        for item in items:
            start_t = time.perf_counter()
            sc_res = self.consistency_reasoner.sample_and_vote(
                item.question, num_samples=num_samples, temperature=0.7
            )
            lat = time.perf_counter() - start_t
            correct = is_answer_correct(sc_res.final_answer, item.ground_truth)

            records.append(
                QueryEvaluationRecord(
                    query_id=item.id,
                    question=item.question,
                    ground_truth=item.ground_truth,
                    predicted_answer=sc_res.final_answer,
                    is_correct=correct,
                    strategy="SELF_CONSISTENCY",
                    calls=sc_res.num_samples,
                    token_usage=sc_res.token_usage,
                    latency_seconds=round(lat, 4),
                    external_api_cost=self.provider.calculate_cost(sc_res.token_usage),
                )
            )

        return self._aggregate_summary(f"SELF_CONSISTENCY (N={num_samples})", records)

    def evaluate_debate(
        self, items: List[BenchmarkItem], num_rounds: int = 2
    ) -> StrategyEvaluationSummary:
        """Evaluates all queries using Always-On Multi-Agent Debate + Supreme Judge."""
        records: List[QueryEvaluationRecord] = []
        cfg = dict(self.debate_config)
        cfg["num_rounds"] = num_rounds
        pipeline = DebateWithJudgePipeline(self.provider, debate_config=cfg)

        logger.info(f"Running ALWAYS-ON DEBATE evaluation (Rounds={num_rounds}) on {len(items)} items...")

        for item in items:
            start_t = time.perf_counter()
            deb_res = pipeline.run(item.question)
            lat = time.perf_counter() - start_t
            correct = is_answer_correct(deb_res.final_answer, item.ground_truth)

            records.append(
                QueryEvaluationRecord(
                    query_id=item.id,
                    question=item.question,
                    ground_truth=item.ground_truth,
                    predicted_answer=deb_res.final_answer,
                    is_correct=correct,
                    strategy="MULTI_AGENT_DEBATE",
                    calls=deb_res.total_calls,
                    token_usage=deb_res.total_token_usage,
                    latency_seconds=round(lat, 4),
                    external_api_cost=deb_res.external_api_cost,
                )
            )

        return self._aggregate_summary(f"ALWAYS_DEBATE (Rounds={num_rounds})", records)

    def evaluate_adaptive(self, items: List[BenchmarkItem]) -> StrategyEvaluationSummary:
        """Evaluates all queries using the dynamic Adaptive Router."""
        records: List[QueryEvaluationRecord] = []
        logger.info(f"Running ADAPTIVE ROUTER evaluation on {len(items)} items...")

        strategy_counts = {
            ReasoningStrategy.DIRECT.value: 0,
            ReasoningStrategy.SELF_CONSISTENCY.value: 0,
            ReasoningStrategy.MULTI_AGENT_DEBATE.value: 0,
        }

        for item in items:
            start_t = time.perf_counter()
            routed = self.router.route_and_solve(item.question)
            lat = time.perf_counter() - start_t
            correct = is_answer_correct(routed.answer, item.ground_truth)

            strategy_val = routed.strategy.value
            strategy_counts[strategy_val] = strategy_counts.get(strategy_val, 0) + 1

            records.append(
                QueryEvaluationRecord(
                    query_id=item.id,
                    question=item.question,
                    ground_truth=item.ground_truth,
                    predicted_answer=routed.answer,
                    is_correct=correct,
                    strategy=strategy_val,
                    calls=routed.call_count,
                    token_usage=routed.token_usage,
                    latency_seconds=round(lat, 4),
                    external_api_cost=routed.external_api_cost,
                )
            )

        n = len(items) if items else 1
        breakdown = {
            "DIRECT": round(strategy_counts.get("DIRECT", 0) / n * 100, 1),
            "SELF_CONSISTENCY": round(strategy_counts.get("SELF_CONSISTENCY", 0) / n * 100, 1),
            "MULTI_AGENT_DEBATE": round(strategy_counts.get("MULTI_AGENT_DEBATE", 0) / n * 100, 1),
        }

        summary = self._aggregate_summary("ADAPTIVE_ROUTER", records)
        summary.routing_breakdown = breakdown
        return summary

    def run_full_comparison(self, items: List[BenchmarkItem]) -> Dict[str, StrategyEvaluationSummary]:
        """Runs comparative evaluation across all 4 modes on the benchmark dataset."""
        logger.info(f"Commencing full 4-paradigm comparative evaluation across {len(items)} queries.")
        return {
            "direct": self.evaluate_direct(items),
            "self_consistency": self.evaluate_self_consistency(items),
            "debate": self.evaluate_debate(items),
            "adaptive": self.evaluate_adaptive(items),
        }

    def _aggregate_summary(
        self, strategy_name: str, records: List[QueryEvaluationRecord]
    ) -> StrategyEvaluationSummary:
        """Aggregates individual query records into a summary."""
        total = len(records)
        correct = sum(1 for r in records if r.is_correct)
        accuracy = round(correct / total, 4) if total > 0 else 0.0

        total_calls = sum(r.calls for r in records)
        mean_calls = round(total_calls / total, 2) if total > 0 else 0.0

        total_tokens = sum(r.token_usage.total_tokens for r in records)
        mean_tokens = round(total_tokens / total, 1) if total > 0 else 0.0

        total_lat = sum(r.latency_seconds for r in records)
        mean_lat = round(total_lat / total, 2) if total > 0 else 0.0

        total_cost = sum(r.external_api_cost for r in records)

        return StrategyEvaluationSummary(
            strategy_name=strategy_name,
            total_queries=total,
            correct_count=correct,
            accuracy=accuracy,
            total_calls=total_calls,
            mean_calls_per_query=mean_calls,
            total_tokens=total_tokens,
            mean_tokens_per_query=mean_tokens,
            total_latency_seconds=round(total_lat, 2),
            mean_latency_seconds=mean_lat,
            external_api_cost=total_cost,
            query_records=records,
        )
