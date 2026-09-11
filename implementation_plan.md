# Phase 2 Implementation Plan: Ollama Provider + Single LLM

## Goal Description
Implement the Ollama local LLM provider adhering to the provider abstraction interface, and deliver a working Single LLM question-answering system that runs completely locally with zero external API costs. Output structured results containing answer, explanation, model, token usage, latency, and metadata.

## Findings & Environment Check
- **Git Remote**: `https://github.com/WhiteHorse2209/Adaptive-LLM-Reasoning-Debate-Router.git` is configured and tracking branch `main`. Phase 1 was pushed successfully.
- **Ollama Status**: Ollama server is running locally on `http://localhost:11434`.
- **Available Models**:
  - `llama3.2:latest` (2.0 GB)
  - `llama3.2:3b` (2.0 GB)
  - `qwen3:1.7b` (1.4 GB)
- We configure `config.json` to utilize these existing local models (`llama3.2:latest` as `local_fast` and `qwen3:1.7b` as `local_reasoning`).

## Proposed Changes

### 1. Dependencies
#### [MODIFY] `requirements.txt`
- Add `requests>=2.31.0` for clean HTTP communication with Ollama REST API (`/api/generate` and `/api/tags`).

### 2. Configuration Enhancement
#### [MODIFY] `config.json`
- Update model names to match the user's locally installed models (`llama3.2:latest`, `qwen3:1.7b`).
- Add `api_base` (`http://localhost:11434`) and `timeout` (seconds) to profile options.

### 3. Provider Abstraction Layer
#### [NEW] `src/provider/models.py`
- Pydantic models:
  - `TokenUsage`: `prompt_tokens`, `completion_tokens`, `total_tokens`.
  - `LLMRequest`: `prompt`, `system_prompt`, `model`, `temperature`, `max_tokens`, `timeout`.
  - `LLMResponse`: `text`, `model`, `token_usage`, `latency_seconds`, `metadata`, `raw_response`.
  - `StructuredQAResponse`: `answer`, `explanation`, `model`, `token_usage`, `latency_seconds`, `metadata`.

#### [NEW] `src/provider/base.py`
- Abstract base class `LLMProvider(ABC)`:
  - `generate(request: LLMRequest) -> LLMResponse`
  - `health_check() -> bool`

#### [NEW] `src/provider/ollama.py`
- `OllamaProvider(LLMProvider)`:
  - Connects to Ollama REST API (`/api/generate`).
  - Measures wall-clock latency with high-resolution timer (`time.perf_counter()`).
  - Extracts native Ollama token counts: `prompt_eval_count`, `eval_count`.
  - Gracefully handles connection timeouts and HTTP errors.

#### [NEW] `src/provider/factory.py`
- `get_provider(profile_config: dict) -> LLMProvider`:
  - Factory function instantiating `OllamaProvider` (and prepared for future cloud providers).

### 4. Single LLM Question-Answering Service & CLI
#### [NEW] `src/reasoning/direct.py`
- `DirectReasoner`:
  - Formats prompt to elicit clean structured answer and reasoning/explanation.
  - Returns `StructuredQAResponse`.

#### [NEW] `src/main.py`
- CLI entrypoint allowing a user to run:
  `python -m src.main --question "What is the capital of France?" --profile local_fast`
- Prints structured output as formatted JSON or readable summary.

### 5. Automated Tests
#### [NEW] `tests/test_ollama_provider.py`
- Unit tests mocking Ollama responses (testing prompt token count extraction, latency calculation, error handling).
- Integration test checking live Ollama connection and generation against installed `llama3.2:latest`.

### 6. Documentation & Git Sync
#### [MODIFY] `README.md`
- Add Phase 2 documentation, architecture updates, and instructions for running the Single LLM QA.
- Commit with meaningful message: `feat: implement Phase 2 Ollama provider and single LLM reasoning`
- Push to GitHub remote `origin/main`.

## Verification Plan
### Automated Tests
- Run `.\venv\Scripts\python.exe -m pytest tests/`
  - Verifies unit tests pass with mocked responses.
  - Verifies live integration test against local Ollama.
### Manual Verification
- Run CLI:
  `.\venv\Scripts\python.exe -m src.main --question "What is 25 * 4?" --profile local_fast`
- Verify JSON output contains:
  - `answer`
  - `explanation`
  - `model`
  - `token_usage` (prompt, completion, total)
  - `latency_seconds`
  - `metadata` (provider, external_api_cost: 0)
