from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.provider.models import TokenUsage


class AgentConfig(BaseModel):
    """Configuration for an individual debating agent."""
    name: str
    persona: str = Field(
        default="Analytical reasoning agent focused on rigorous logical derivation.",
        description="Persona and behavioral prompt instructions for the debater"
    )
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = 300


class DebateTurn(BaseModel):
    """A single agent's argument and answer during a debate round."""
    round_number: int
    agent_name: str
    argument: str
    current_answer: str
    revised: bool = Field(default=False, description="Whether the agent modified its answer compared to its previous turn")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0


class DebateRound(BaseModel):
    """A round of debate where all participating agents present their arguments."""
    round_number: int
    turns: List[DebateTurn] = Field(default_factory=list)


class DebateTranscript(BaseModel):
    """Complete transcript and audit log of a multi-round, multi-agent debate."""
    question: str
    rounds: List[DebateRound] = Field(default_factory=list)
    final_agent_answers: Dict[str, str] = Field(default_factory=dict, description="Final answer proposed by each agent")
    consensus_reached: bool = Field(default=False, description="True if all agents agree on the exact same answer")
    consensus_answer: Optional[str] = Field(default=None, description="The agreed answer if consensus was reached")
    total_calls: int = 0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_latency_seconds: float = 0.0
    external_api_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
