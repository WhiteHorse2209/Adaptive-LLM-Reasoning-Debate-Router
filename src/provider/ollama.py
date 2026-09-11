"""Ollama provider with structured logging, retries, and error resilience."""

import time
from typing import Any, Dict, Optional
import requests

from src.provider.base import LLMProvider
from src.provider.exceptions import (
    LLMProviderError,
    MalformedOutputError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from src.provider.models import LLMRequest, LLMResponse, TokenUsage
from src.utils.logger import get_logger
from src.utils.retry import retry_with_exponential_backoff


class OllamaProvider(LLMProvider):
    """Implementation of LLMProvider targeting a local Ollama instance with production hardening."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "http://localhost:11434").rstrip("/")
        self.generate_url = f"{self.api_base}/api/generate"
        self.tags_url = f"{self.api_base}/api/tags"
        self.max_retries = config.get("max_retries", 2)
        self.retry_delay = config.get("retry_delay", 0.5)
        self.logger = get_logger("ollama_provider")

    def health_check(self) -> bool:
        """Checks if the Ollama server is running and reachable."""
        try:
            resp = requests.get(self.tags_url, timeout=5)
            if resp.status_code == 200:
                self.logger.debug("Ollama daemon health check passed.")
                return True
            self.logger.warning(f"Ollama daemon returned status code {resp.status_code}")
            return False
        except Exception as e:
            self.logger.warning(f"Ollama daemon unreachable at {self.tags_url}: {e}")
            return False

    def _execute_post(self, payload: Dict[str, Any], timeout: float, model: str) -> Dict[str, Any]:
        """Executes the raw HTTP POST request with specific exception translation."""
        try:
            resp = requests.post(self.generate_url, json=payload, timeout=timeout)
            if resp.status_code == 404:
                error_detail = resp.text
                raise OllamaModelNotFoundError(
                    f"Model '{model}' not found in Ollama (HTTP 404): {error_detail}. Run 'ollama pull {model}'."
                )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise MalformedOutputError(f"Ollama returned non-dict response: {type(data)}")
            return data
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError(
                f"Ollama generation timed out after {timeout}s for model '{model}': {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.api_base}. Ensure Ollama is running (`ollama serve`): {e}"
            ) from e
        except (OllamaModelNotFoundError, OllamaTimeoutError, OllamaConnectionError, MalformedOutputError):
            raise
        except requests.exceptions.HTTPError as e:
            resp_text = resp.text if 'resp' in locals() else str(e)
            if "not found" in resp_text.lower():
                raise OllamaModelNotFoundError(
                    f"Model '{model}' not found in Ollama: {resp_text}. Run 'ollama pull {model}'."
                ) from e
            raise LLMProviderError(f"Ollama returned HTTP error {resp.status_code}: {resp_text}") from e
        except Exception as e:
            raise LLMProviderError(f"Unexpected error communicating with Ollama: {e}") from e

    def generate(self, request: LLMRequest, request_id: Optional[str] = None) -> LLMResponse:
        """Invokes the Ollama local model via REST API with exponential backoff retries."""
        model = request.model or self.default_model
        if not model:
            raise ValueError("No model specified in request or provider configuration.")

        temperature = request.temperature if request.temperature is not None else self.default_temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.default_max_tokens
        timeout = request.timeout or self.default_timeout

        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if request.stop:
            options["stop"] = request.stop

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
            "options": options,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        req_log_extra = {"request_id": request_id} if request_id else {}
        self.logger.debug(
            f"Dispatching inference request to model '{model}' (timeout={timeout}s)",
            extra=req_log_extra,
        )

        # Apply exponential backoff retries
        runner = retry_with_exponential_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_delay,
            backoff_factor=2.0,
            jitter=False if self.retry_delay < 0.1 else True,
            logger_instance=self.logger,
        )(self._execute_post)

        start_time = time.perf_counter()
        data = runner(payload, timeout, model)
        duration = time.perf_counter() - start_time

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        metadata = {
            "provider": "ollama",
            "request_id": request_id,
            "done": data.get("done", True),
            "done_reason": data.get("done_reason", "stop"),
            "total_duration_ns": data.get("total_duration", 0),
            "load_duration_ns": data.get("load_duration", 0),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
            "eval_duration_ns": data.get("eval_duration", 0),
            "external_api_cost": 0.0,
        }

        self.logger.debug(
            f"Inference completed in {duration:.2f}s (tokens={total_tokens})",
            extra=req_log_extra,
        )

        return LLMResponse(
            text=data.get("response", "").strip(),
            model=model,
            token_usage=usage,
            latency_seconds=round(duration, 4),
            metadata=metadata,
            raw_response=data,
        )
