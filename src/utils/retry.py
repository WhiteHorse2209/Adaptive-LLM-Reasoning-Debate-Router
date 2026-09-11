"""Exponential backoff retry utility for resilient LLM inference."""

import functools
import random
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Type

import requests

from src.provider.exceptions import (
    LLMProviderError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from src.utils.logger import get_logger

logger = get_logger("retry")

DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    OllamaConnectionError,
    OllamaTimeoutError,
)

NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    OllamaModelNotFoundError,
    ValueError,
    KeyError,
)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Sequence[Type[Exception]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    non_retryable_exceptions: Sequence[Type[Exception]] = NON_RETRYABLE_EXCEPTIONS,
    logger_instance: Optional[Any] = None,
) -> Callable:
    """Decorator to retry a callable with exponential backoff and jitter."""
    log = logger_instance or logger

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    attempt += 1

                    # Fast-path exit on non-retryable exceptions
                    if isinstance(exc, tuple(non_retryable_exceptions)):
                        log.error(f"Encountered non-retryable exception: {exc}. Aborting immediately.")
                        raise

                    # Fast-path exit if exception is not within retryable exceptions
                    if not isinstance(exc, tuple(retryable_exceptions)):
                        log.error(f"Encountered unhandled exception: {exc}. Aborting.")
                        raise

                    func_name = getattr(func, "__name__", str(func))
                    if attempt > max_retries:
                        log.error(
                            f"Exceeded max retries ({max_retries}) for {func_name}. Last error: {exc}"
                        )
                        raise

                    sleep_time = delay * (1.0 + (random.uniform(0, 0.25) if jitter else 0.0))
                    log.warning(
                        f"Attempt {attempt}/{max_retries} failed for {func_name}: {exc}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff_factor

        return wrapper

    return decorator
