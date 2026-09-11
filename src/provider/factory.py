from typing import Any, Dict

from src.provider.base import LLMProvider
from src.provider.ollama import OllamaProvider


def get_provider(profile_config: Dict[str, Any]) -> LLMProvider:
    """Factory function to instantiate an LLMProvider based on configuration.
    
    Args:
        profile_config: Dictionary containing provider, model, timeout, api_base, etc.
        
    Returns:
        An instantiated LLMProvider subclass.
        
    Raises:
        ValueError: If provider is unknown or unsupported.
    """
    provider_type = profile_config.get("provider", "").lower()
    
    if provider_type == "ollama":
        return OllamaProvider(profile_config)
    else:
        raise ValueError(
            f"Unsupported provider: '{provider_type}'. Supported providers in Phase 2: 'ollama'."
        )
