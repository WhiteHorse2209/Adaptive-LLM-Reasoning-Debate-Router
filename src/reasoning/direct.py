import re
from typing import Optional

from src.provider.base import LLMProvider
from src.provider.models import LLMRequest, StructuredQAResponse


class DirectReasoner:
    """Direct single-model reasoning engine."""

    SYSTEM_PROMPT = (
        "You are an analytical and concise reasoning assistant. "
        "Solve the user's question directly. "
        "Provide a clear, brief explanation of your reasoning first. "
        "Then end your response with the exact final answer on its own line in the format: "
        "'FINAL ANSWER: <your answer>'."
    )

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def answer(
        self,
        question: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> StructuredQAResponse:
        """Executes direct single-model reasoning for a given question."""
        request = LLMRequest(
            prompt=f"Question: {question}",
            system_prompt=self.SYSTEM_PROMPT,
            temperature=temperature if temperature is not None else self.provider.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.provider.default_max_tokens,
            timeout=self.provider.default_timeout,
        )

        response = self.provider.generate(request)
        answer_text, explanation_text = self._parse_response(response.text)

        cost = self.provider.calculate_cost(response.token_usage)

        return StructuredQAResponse(
            answer=answer_text,
            explanation=explanation_text,
            model=response.model,
            provider=self.provider.provider_name,
            token_usage=response.token_usage,
            latency_seconds=response.latency_seconds,
            external_api_cost=cost,
            metadata=response.metadata,
        )

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Extracts the final answer and explanation from the model output."""
        match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
            explanation = text[:match.start()].strip()
            # If explanation is empty, use text
            if not explanation:
                explanation = text
            return final_answer, explanation

        # Fallback if marker is missing
        lines = text.strip().split("\n")
        if len(lines) > 1:
            return lines[-1].strip(), text.strip()
        return text.strip(), text.strip()
