from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.provider.models import TokenUsage


class DifficultyLevel(str, Enum):
    EASY = "EASY"
    UNCERTAIN = "UNCERTAIN"
    HARD = "HARD"


class ReasoningStrategy(str, Enum):
    DIRECT = "DIRECT"
    SELF_CONSISTENCY = "SELF_CONSISTENCY"
    MULTI_AGENT_DEBATE = "MULTI_AGENT_DEBATE"


class ConfidenceAssessment(BaseModel):
    """Detailed confidence and difficulty assessment for a query."""
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score bounded between 0.0 and 1.0")
    level: str = Field(..., description="Categorical confidence: HIGH, MEDIUM, LOW")
    difficulty: DifficultyLevel = Field(..., description="Estimated difficulty: EASY, UNCERTAIN, HARD")
    justification: str = Field(default="", description="Reasoning behind the confidence estimation")


class RoutedResponse(BaseModel):
    """Output response from the adaptive router containing final answer and detailed routing telemetry."""
    answer: str
    explanation: str
    strategy: ReasoningStrategy
    difficulty: DifficultyLevel
    confidence: ConfidenceAssessment
    call_count: int = 1
    model: str
    provider: str
    token_usage: TokenUsage
    latency_seconds: float
    external_api_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
