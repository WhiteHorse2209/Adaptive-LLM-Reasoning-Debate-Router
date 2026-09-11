from typing import Any, Dict, List
from pydantic import BaseModel, Field

from src.provider.models import TokenUsage


class CandidateSolution(BaseModel):
    """An individual candidate reasoning path and answer from a sampling call."""
    sample_id: int
    answer: str
    explanation: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0


class SelfConsistencyResult(BaseModel):
    """Aggregated result of multi-path self-consistency sampling and majority voting."""
    final_answer: str
    final_explanation: str
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="Fraction of samples agreeing on the final answer")
    agreement_distribution: Dict[str, int] = Field(default_factory=dict, description="Vote counts per unique answer")
    candidates: List[CandidateSolution] = Field(default_factory=list)
    num_samples: int
    model: str
    provider: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    external_api_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
