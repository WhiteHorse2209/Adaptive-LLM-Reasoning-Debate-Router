import time
from typing import Any, Dict, Optional
import requests

from src.provider.base import LLMProvider
from src.provider.models import LLMRequest, LLMResponse, TokenUsage


class OllamaProvider(LLMProvider):
    """Implementation of LLMProvider targeting a local Ollama instance."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "http://localhost:11434").rstrip("/")
        self.generate_url = f"{self.api_base}/api/generate"
        self.tags_url = f"{self.api_base}/api/tags"

    def health_check(self) -> bool:
        """Checks if the Ollama server is running and reachable."""
        try:
            resp = requests.get(self.tags_url, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Invokes the Ollama local model via REST API."""
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

        start_time = time.perf_counter()
        try:
            resp = requests.post(self.generate_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Ollama generation timed out after {timeout}s for model '{model}': {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.api_base}. Ensure Ollama is running (`ollama serve`): {e}"
            ) from e
        except requests.exceptions.HTTPError as e:
            error_detail = resp.text if 'resp' in locals() else str(e)
            raise RuntimeError(f"Ollama returned HTTP error {resp.status_code}: {error_detail}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error communicating with Ollama: {e}") from e

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
            "done": data.get("done", True),
            "done_reason": data.get("done_reason", "stop"),
            "total_duration_ns": data.get("total_duration", 0),
            "load_duration_ns": data.get("load_duration", 0),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
            "eval_duration_ns": data.get("eval_duration", 0),
            "external_api_cost": 0.0,  # Zero external cost for local models
        }

        return LLMResponse(
            text=data.get("response", "").strip(),
            model=model,
            token_usage=usage,
            latency_seconds=round(duration, 4),
            metadata=metadata,
            raw_response=data,
        )
