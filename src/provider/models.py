from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Tracks token consumption for an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(BaseModel):
    """Standardized input request for any LLM provider."""
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: Optional[int] = 60
    stop: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Standardized output response from any LLM provider."""
    text: str
    model: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None


class StructuredQAResponse(BaseModel):
    """Structured QA output containing the answer, explanation, and telemetry."""
    answer: str
    explanation: str
    model: str
    provider: str
    token_usage: TokenUsage
    latency_seconds: float
    external_api_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
