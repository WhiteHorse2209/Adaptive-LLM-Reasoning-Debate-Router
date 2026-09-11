"""Provider domain exceptions for robust LLM engineering."""


class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""
    pass


class OllamaConnectionError(LLMProviderError, ConnectionError):
    """Raised when unable to establish a connection to the Ollama daemon."""
    pass


class OllamaModelNotFoundError(LLMProviderError, FileNotFoundError):
    """Raised when the specified model is not pulled or available in Ollama."""
    pass


class OllamaTimeoutError(LLMProviderError, TimeoutError):
    """Raised when an inference generation request to Ollama times out."""
    pass


class MalformedOutputError(LLMProviderError, ValueError):
    """Raised when an LLM returns an empty, corrupted, or unparseable output structure."""
    pass
