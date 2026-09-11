"""Ablation studies framework analyzing thresholds, agent counts, rounds, and arbitration."""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.evaluation.dataset import BenchmarkItem, is_answer_correct
from src.provider.base import LLMProvider
from src.provider.models import TokenUsage
from src.reasoning.consistency import SelfConsistencyReasoner
from src.reasoning.debate import MultiAgentDebateEngine
from src.reasoning.debate_models import AgentConfig
from src.reasoning.judge import DebateJudge, DebateWithJudgePipeline
from src.router.models import DifficultyLevel, ReasoningStrategy, RoutedResponse
from src.router.router import AdaptiveRouter
from src.utils.logger import get_logger

logger = get_logger("ablation")


class AblationConfig(BaseModel):
    """Configuration for a specific experimental or ablation condition."""
    experiment_id: str
    name: str
    description: str
    confidence_threshold_high: float = 0.80
    confidence_threshold_low: float = 0.50
    consistency_samples: int = 3
    debate_num_agents: int = 2
    debate_num_rounds: int = 2
    adjudication_type: str = Field(default="judge", description="'judge' or 'majority_voting'")
    model: Optional[str] = None


class AblationRunResult(BaseModel):
    """Aggregate outcomes and efficiency metrics for an ablation configuration."""
    experiment_id: str
    name: str
    accuracy: float
    mean_calls_per_query: float
    mean_tokens_per_query: float
    mean_latency_seconds: float
    external_api_cost: float
    routing_breakdown: Dict[str, float] = Field(default_factory=dict)
    efficiency_index: float = Field(
        ..., description="Normalized Pareto score: Accuracy / (mean_calls * mean_latency)"
    )
    config: AblationConfig


def get_standard_ablation_matrix() -> List[AblationConfig]:
    """Generates the canonical suite of experimental configurations across all dimensions."""
    return [
        # Baseline
        AblationConfig(
            experiment_id="exp_01_baseline",
            name="Standard Baseline",
            description="Balanced default router thresholds (0.80 / 0.50), 2 agents, 2 rounds, judge",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
            consistency_samples=3,
            debate_num_agents=2,
            debate_num_rounds=2,
            adjudication_type="judge",
        ),
        # 1. Threshold Variations
        AblationConfig(
            experiment_id="exp_02_conservative_router",
            name="Conservative Quality-First Router",
            description="Elevated thresholds (0.90 / 0.70) routing aggressively to Self-Consistency and Debate",
            confidence_threshold_high=0.90,
            confidence_threshold_low=0.70,
            consistency_samples=3,
            debate_num_agents=2,
            debate_num_rounds=2,
            adjudication_type="judge",
        ),
        AblationConfig(
            experiment_id="exp_03_aggressive_cost_saver",
            name="Cost-Saver Efficiency Router",
            description="Low thresholds (0.70 / 0.30) favoring Direct fast zero-shot reasoning",
            confidence_threshold_high=0.70,
            confidence_threshold_low=0.30,
            consistency_samples=3,
            debate_num_agents=2,
            debate_num_rounds=2,
            adjudication_type="judge",
        ),
        # 2. Debate Round Variations
        AblationConfig(
            experiment_id="exp_04_debate_1_round",
            name="Shallow Debate (1 Round)",
            description="Only independent exploration round; eliminates mutual cross-examination",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
            consistency_samples=3,
            debate_num_agents=2,
            debate_num_rounds=1,
            adjudication_type="judge",
        ),
        AblationConfig(
            experiment_id="exp_05_debate_3_rounds",
            name="Deep Debate (3 Rounds)",
            description="Extended cross-examination and multi-turn rebuttal convergence",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
            consistency_samples=3,
            debate_num_agents=2,
            debate_num_rounds=3,
            adjudication_type="judge",
        ),
        # 3. Agent Count Variations
        AblationConfig(
            experiment_id="exp_06_debate_3_agents",
            name="Tripartite Debate (3 Agents)",
            description="3 distinct agent personas (Logician, Empirical Critic, Domain Specialist)",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
            consistency_samples=3,
            debate_num_agents=3,
            debate_num_rounds=2,
            adjudication_type="judge",
        ),
        # 4. Adjudication Ablation (Judge vs Majority Voting)
        AblationConfig(
            experiment_id="exp_07_no_judge_majority_vote",
            name="No-Judge Majority Voting",
            description="Eliminates the impartial Supreme Judge; selects final answer via naive agent vote",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
            consistency_samples=3,
            debate_num_agents=2,
            debate_num_rounds=2,
            adjudication_type="majority_voting",
        ),
        # 5. Self-Consistency Sampling Variations
        AblationConfig(
            experiment_id="exp_08_consistency_n5",
            name="High-Sample Consistency (N=5)",
            description="Self-consistency sampling expanded to N=5 candidates",
            confidence_threshold_high=0.80,
            confidence_threshold_low=0.50,
            consistency_samples=5,
            debate_num_agents=2,
            debate_num_rounds=2,
            adjudication_type="judge",
        ),
    ]


class AblationEngine:
    """Orchestrates parameterized ablation runs across reasoning configurations."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _build_debate_config(self, cfg: AblationConfig) -> Dict[str, Any]:
        """Constructs debate configuration with appropriate agent count and personas."""
        agents = [
            {
                "name": "Agent_A",
                "persona": "Analytical logician focused on rigorous first-principles derivation.",
            },
            {
                "name": "Agent_B",
                "persona": "Critical empirical thinker focused on edge cases and counterexamples.",
            },
        ]
        if cfg.debate_num_agents >= 3:
            agents.append({
                "name": "Agent_C",
                "persona": "Pragmatic domain expert verifying practical feasibility and boundary sanity.",
            })

        return {
            "num_agents": cfg.debate_num_agents,
            "num_rounds": cfg.debate_num_rounds,
            "agent_temperature": 0.7,
            "agents": agents,
        }

    def run_single_ablation(
        self, cfg: AblationConfig, items: List[BenchmarkItem]
    ) -> AblationRunResult:
        """Executes a complete benchmark evaluation under a specific ablation config."""
        logger.info(f"Executing ablation run '{cfg.name}' ({cfg.experiment_id}) on {len(items)} queries...")

        router_cfg = {
            "confidence_threshold_high": cfg.confidence_threshold_high,
            "confidence_threshold_low": cfg.confidence_threshold_low,
            "consistency_samples": cfg.consistency_samples,
            "sample_temperature": 0.7,
        }
        debate_cfg = self._build_debate_config(cfg)

        router = AdaptiveRouter(
            self.provider,
            router_config=router_cfg,
            debate_config=debate_cfg,
        )

        strategy_counts = {"DIRECT": 0, "SELF_CONSISTENCY": 0, "MULTI_AGENT_DEBATE": 0}
        total_correct = 0
        total_calls = 0
        total_tokens = 0
        total_lat = 0.0
        total_cost = 0.0

        for item in items:
            start_t = time.perf_counter()
            routed = router.route_and_solve(item.question)
            lat = time.perf_counter() - start_t

            ans = routed.answer

            # If adjudication is ablated to majority voting during debate
            if routed.strategy == ReasoningStrategy.MULTI_AGENT_DEBATE and cfg.adjudication_type == "majority_voting":
                if routed.transcript and routed.transcript.final_agent_answers:
                    # Select majority answer from debating agents
                    from collections import Counter
                    answers = list(routed.transcript.final_agent_answers.values())
                    vote_counts = Counter(answers)
                    ans = vote_counts.most_common(1)[0][0]
                    # Subtract the 1 judge call and judge tokens
                    routed.call_count = max(1, routed.call_count - 1)

            correct = is_answer_correct(ans, item.ground_truth)
            if correct:
                total_correct += 1

            strat = routed.strategy.value
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

            total_calls += routed.call_count
            total_tokens += routed.token_usage.total_tokens
            total_lat += lat
            total_cost += routed.external_api_cost

        n = len(items) if items else 1
        accuracy = round(total_correct / n, 4)
        mean_calls = round(total_calls / n, 2)
        mean_tokens = round(total_tokens / n, 1)
        mean_lat = round(total_lat / n, 2)

        # Efficiency index: Accuracy / (mean_calls * mean_latency + epsilon)
        eff_index = round(accuracy / (max(0.1, mean_calls * max(0.1, mean_lat))), 4)

        breakdown = {
            "DIRECT": round(strategy_counts.get("DIRECT", 0) / n * 100, 1),
            "SELF_CONSISTENCY": round(strategy_counts.get("SELF_CONSISTENCY", 0) / n * 100, 1),
            "MULTI_AGENT_DEBATE": round(strategy_counts.get("MULTI_AGENT_DEBATE", 0) / n * 100, 1),
        }

        return AblationRunResult(
            experiment_id=cfg.experiment_id,
            name=cfg.name,
            accuracy=accuracy,
            mean_calls_per_query=mean_calls,
            mean_tokens_per_query=mean_tokens,
            mean_latency_seconds=mean_lat,
            external_api_cost=total_cost,
            routing_breakdown=breakdown,
            efficiency_index=eff_index,
            config=cfg,
        )

    def run_ablation_grid(
        self, configs: List[AblationConfig], items: List[BenchmarkItem]
    ) -> List[AblationRunResult]:
        """Runs comparative evaluations across a matrix of ablation configurations."""
        results: List[AblationRunResult] = []
        for cfg in configs:
            res = self.run_single_ablation(cfg, items)
            results.append(res)
        return results

    @staticmethod
    def export_results(
        results: List[AblationRunResult],
        json_path: str = "experiments/ablation_results.json",
        csv_path: str = "experiments/ablation_summary.csv",
    ):
        """Exports ablation experiment results to both JSON and CSV files."""
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        # 1. Export JSON
        data = [r.model_dump() for r in results]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported ablation experiment data to {json_path}")

        # 2. Export CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Experiment_ID",
                "Name",
                "Accuracy",
                "Mean_Calls",
                "Mean_Tokens",
                "Mean_Latency_s",
                "Efficiency_Index",
                "Direct_Pct",
                "SC_Pct",
                "Debate_Pct",
                "Threshold_High",
                "Threshold_Low",
                "Debate_Agents",
                "Debate_Rounds",
                "Adjudication",
            ])
            for r in results:
                writer.writerow([
                    r.experiment_id,
                    r.name,
                    f"{r.accuracy:.4f}",
                    r.mean_calls_per_query,
                    r.mean_tokens_per_query,
                    r.mean_latency_seconds,
                    r.efficiency_index,
                    r.routing_breakdown.get("DIRECT", 0.0),
                    r.routing_breakdown.get("SELF_CONSISTENCY", 0.0),
                    r.routing_breakdown.get("MULTI_AGENT_DEBATE", 0.0),
                    r.config.confidence_threshold_high,
                    r.config.confidence_threshold_low,
                    r.config.debate_num_agents,
                    r.config.debate_num_rounds,
                    r.config.adjudication_type,
                ])
        logger.info(f"Exported ablation summary table to {csv_path}")
