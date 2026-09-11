"""Multi-objective Pareto optimization framework analyzing Accuracy vs Compute vs Latency trade-offs."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class OptimizationPriority(str, Enum):
    """Preset weighting priorities for multi-objective optimization."""
    QUALITY_FIRST = "QUALITY_FIRST"          # 80% accuracy, 10% latency, 10% tokens
    BALANCED = "BALANCED"                    # 50% accuracy, 25% latency, 25% tokens
    BUDGET_CONSTRAINED = "BUDGET_CONSTRAINED" # 25% accuracy, 35% latency, 40% tokens


WEIGHT_PRESETS: Dict[OptimizationPriority, Dict[str, float]] = {
    OptimizationPriority.QUALITY_FIRST: {"acc": 0.80, "lat": 0.10, "tok": 0.10},
    OptimizationPriority.BALANCED: {"acc": 0.50, "lat": 0.25, "tok": 0.25},
    OptimizationPriority.BUDGET_CONSTRAINED: {"acc": 0.25, "lat": 0.35, "tok": 0.40},
}


class ParetoCandidate(BaseModel):
    """A single configuration candidate or strategy evaluated on benchmarks."""
    name: str
    threshold_high: Optional[float] = None
    threshold_low: Optional[float] = None
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Empirical benchmark accuracy")
    mean_latency: float = Field(..., ge=0.0, description="Mean latency in seconds per query")
    mean_tokens: float = Field(..., ge=0.0, description="Mean tokens consumed per query")
    mean_calls: float = Field(..., ge=0.0, description="Mean model calls per query")
    is_pareto_optimal: bool = False
    fitness_score: float = 0.0


class OptimizationReport(BaseModel):
    """Aggregate report comparing candidate configurations and identifying the Pareto frontier."""
    all_candidates: List[ParetoCandidate]
    pareto_frontier: List[ParetoCandidate]
    recommended_candidate: ParetoCandidate
    priority_mode: OptimizationPriority


class ParetoOptimizer:
    """Calculates non-dominated Pareto frontiers and scores candidates against multi-objective fitness functions."""

    @staticmethod
    def dominates(a: ParetoCandidate, b: ParetoCandidate) -> bool:
        """Returns True if candidate `a` Pareto-dominates candidate `b`.
        
        `a` dominates `b` if `a` is no worse than `b` in all 3 objectives
        (Accuracy higher, Latency lower, Tokens lower) and strictly better in at least one.
        """
        no_worse = (
            a.accuracy >= b.accuracy
            and a.mean_latency <= b.mean_latency
            and a.mean_tokens <= b.mean_tokens
        )
        strictly_better = (
            a.accuracy > b.accuracy
            or a.mean_latency < b.mean_latency
            or a.mean_tokens < b.mean_tokens
        )
        return no_worse and strictly_better

    def compute_pareto_frontier(self, candidates: List[ParetoCandidate]) -> List[ParetoCandidate]:
        """Identifies all non-dominated candidates belonging to the Pareto frontier."""
        if not candidates:
            return []

        frontier: List[ParetoCandidate] = []
        for candidate in candidates:
            is_dominated = False
            for other in candidates:
                if other != candidate and self.dominates(other, candidate):
                    is_dominated = True
                    break
            if not is_dominated:
                candidate.is_pareto_optimal = True
                frontier.append(candidate)
            else:
                candidate.is_pareto_optimal = False

        # Sort frontier by accuracy descending
        frontier.sort(key=lambda c: c.accuracy, reverse=True)
        return frontier

    def score_candidates(
        self,
        candidates: List[ParetoCandidate],
        priority: OptimizationPriority = OptimizationPriority.BALANCED,
    ) -> List[ParetoCandidate]:
        """Calculates normalized fitness scores across candidates according to weighting priorities.
        
        Higher fitness score is better (scaled 0.0 to 1.0).
        """
        if not candidates:
            return []

        weights = WEIGHT_PRESETS[priority]
        w_acc = weights["acc"]
        w_lat = weights["lat"]
        w_tok = weights["tok"]

        min_acc = min(c.accuracy for c in candidates)
        max_acc = max(c.accuracy for c in candidates)
        min_lat = min(c.mean_latency for c in candidates)
        max_lat = max(c.mean_latency for c in candidates)
        min_tok = min(c.mean_tokens for c in candidates)
        max_tok = max(c.mean_tokens for c in candidates)

        acc_span = max_acc - min_acc if max_acc > min_acc else 1.0
        lat_span = max_lat - min_lat if max_lat > min_lat else 1.0
        tok_span = max_tok - min_tok if max_tok > min_tok else 1.0

        for c in candidates:
            # Normalized metrics in [0.0, 1.0] where 1.0 is always the best
            norm_acc = (c.accuracy - min_acc) / acc_span
            norm_lat = (max_lat - c.mean_latency) / lat_span
            norm_tok = (max_tok - c.mean_tokens) / tok_span

            fitness = (w_acc * norm_acc) + (w_lat * norm_lat) + (w_tok * norm_tok)
            c.fitness_score = round(fitness, 4)

        # Sort all candidates by fitness score descending
        candidates.sort(key=lambda c: c.fitness_score, reverse=True)
        return candidates

    def optimize(
        self,
        candidates: List[ParetoCandidate],
        priority: OptimizationPriority = OptimizationPriority.BALANCED,
    ) -> OptimizationReport:
        """Performs end-to-end Pareto identification, fitness scoring, and recommendation."""
        if not candidates:
            raise ValueError("Candidate list cannot be empty for optimization.")

        frontier = self.compute_pareto_frontier(candidates)
        scored = self.score_candidates(candidates, priority=priority)

        # Recommend top-scoring candidate from the Pareto frontier, or overall top
        pareto_recommendations = [c for c in scored if c.is_pareto_optimal]
        recommended = pareto_recommendations[0] if pareto_recommendations else scored[0]

        return OptimizationReport(
            all_candidates=scored,
            pareto_frontier=frontier,
            recommended_candidate=recommended,
            priority_mode=priority,
        )
