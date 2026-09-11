"""Pydantic schemas for the FastAPI REST interface."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReasonApiRequest(BaseModel):
    """Payload for POST /reason endpoint."""
    question: str = Field(..., description="The query, logic problem, or question to reason through.")
    profile: Optional[str] = Field(None, description="Optional model profile override from config.json.")
    mode: Optional[str] = Field("adaptive", description="Execution mode: 'adaptive', 'direct', 'self_consistency', or 'debate'.")
    rounds: Optional[int] = Field(2, ge=1, le=5, description="Number of debate rounds if debate mode is triggered.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Optional sampling temperature override.")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum token generation limit.")


class HealthApiResponse(BaseModel):
    """Payload for GET /health endpoint."""
    status: str = Field(..., description="System status ('healthy' or 'degraded').")
    provider: str = Field(..., description="Active LLM provider name.")
    default_model: str = Field(..., description="Default model configured.")
    provider_healthy: bool = Field(..., description="True if local Ollama daemon is reachable.")
    active_profile: str = Field(..., description="Name of the active model profile.")


class ReasonApiResponse(BaseModel):
    """Payload returned by POST /reason endpoint."""
    request_id: str
    question: str
    answer: str
    explanation: str
    strategy: str
    difficulty: str
    confidence_score: float
    model: str
    provider: str
    call_count: int
    token_usage: Dict[str, int]
    latency_seconds: float
    external_api_cost: float
    debate_info: Optional[Dict[str, Any]] = None
    self_consistency_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
