from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.provider.models import TokenUsage
from src.reasoning.debate_models import DebateTranscript


class JudgeVerdict(BaseModel):
    """The authoritative evaluation and decision issued by the debate judge."""
    verdict_answer: str = Field(..., description="The final answer determined to be correct by the judge")
    evaluation_summary: str = Field(..., description="Detailed analytical justification assessing each agent's arguments")
    winning_agent: Optional[str] = Field(default=None, description="Name of the winning agent, or 'SYNTHESIS' if synthesized")
    confidence_in_verdict: float = Field(default=1.0, ge=0.0, le=1.0, description="Judge's confidence in the verdict")
    identified_flaws: List[str] = Field(default_factory=list, description="Logical fallacies or calculation errors identified in agent arguments")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DebatePipelineResult(BaseModel):
    """End-to-end outcome of a multi-agent debate followed by judge adjudication."""
    question: str
    final_answer: str
    explanation: str
    transcript: DebateTranscript
    verdict: JudgeVerdict
    total_calls: int
    total_token_usage: TokenUsage
    total_latency_seconds: float
    external_api_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
