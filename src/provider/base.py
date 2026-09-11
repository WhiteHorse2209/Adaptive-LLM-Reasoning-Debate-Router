from abc import ABC, abstractmethod
from typing import Any, Dict

from src.provider.models import LLMRequest, LLMResponse, TokenUsage


class LLMProvider(ABC):
    """Abstract Base Class for all LLM providers (Ollama, and future cloud providers)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = config.get("provider", "unknown")
        self.default_model = config.get("model", "")
        self.default_temperature = config.get("temperature", 0.1)
        self.default_max_tokens = config.get("max_tokens")
        self.default_timeout = config.get("timeout", 60)

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Executes a single prompt completion against the provider.
        
        Args:
            request: The standardized LLMRequest.
            
        Returns:
            The standardized LLMResponse with text, token usage, and latency.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Checks whether the provider endpoint is healthy and reachable.
        
        Returns:
            True if healthy, False otherwise.
        """
        pass

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculates external API cost in USD.
        
        For local inference (e.g. Ollama), external API cost is always 0.0.
        Computation is performed entirely on local hardware.
        """
        return 0.0
