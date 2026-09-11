# Dual-Phase Implementation Plan: Phase 8 (Robust LLM Engineering) & Phase 9 (FastAPI + Streamlit)

## Goal Description
Implement the next two major phases to transition the Adaptive LLM Reasoning & Debate Router from a functional prototype into a production-grade local AI engineering platform:
1. **Phase 8 — Robust LLM Engineering**: Production-ready reliability layer featuring structured logging, request ID tracking, exponential backoff retries, specific Ollama failure classifications (model missing, connection refused, generation timeout, malformed output), structured output validation, and graceful fallbacks.
2. **Phase 9 — FastAPI + Streamlit Interface**: A unified serving and visualization layer exposing:
   - **FastAPI**: Clean RESTful endpoints (`GET /health`, `POST /reason`) with Pydantic request/response validation.
   - **Streamlit**: An interactive web dashboard with specialized visualization for each reasoning mode (Direct answer card, Self-Consistency candidate distribution & agreement score, and Multi-Agent Debate round-by-round cross-examination cards with Judge adjudication).

As instructed, both phases will be implemented in this session with **individual git commits per phase**.

---

## Phase 8: Robust LLM Engineering

### 1. Error Classification & Provider Exceptions
#### [NEW] `src/provider/exceptions.py`
- Hierarchical domain exceptions:
  - `LLMProviderError`: Base exception for all provider operations.
  - `OllamaConnectionError`: Ollama server unreachable (`http://localhost:11434` down/refused).
  - `OllamaModelNotFoundError`: Requested model is not pulled or available in Ollama (404).
  - `OllamaTimeoutError`: Model inference exceeded wall-clock timeout.
  - `MalformedOutputError`: LLM returned empty, unparseable, or corrupt structure.

### 2. Structured Logging & Request IDs
#### [NEW] `src/utils/logger.py`
- `get_logger(name: str)`: Provides structured logging with timestamps, component name, log levels, and contextual `request_id`.
- Formats logs cleanly with machine-readable metadata and readable terminal output.

### 3. Exponential Backoff Retries
#### [NEW] `src/utils/retry.py`
- `retry_with_exponential_backoff`: Decorator and utility function:
  - Configurable `max_retries` (default 3), `initial_delay` (default 1.0s), `backoff_factor` (default 2.0), `jitter` (True).
  - Selective retry policy: retries transient connection errors and timeouts, but immediately fails on non-retryable errors (e.g. `OllamaModelNotFoundError` / 404).
  - Never retries indefinitely.

### 4. Provider & Router Integration
#### [MODIFY] `src/provider/ollama.py`
- Wrap HTTP requests with exception classification and exponential backoff retry.
- Attach `request_id` to log entries and request traces.
#### [MODIFY] `src/router/router.py`
- Generate a unique `request_id` for every query.
- Add graceful degradation: if Mode 3 (debate) or Mode 2 (self-consistency) suffers an unexpected provider exception, cleanly fall back to Mode 1 (direct) with warning metadata rather than unhandled crash.

### 5. Unit Tests & Commit for Phase 8
#### [NEW] `tests/test_robustness.py`
- Tests exponential backoff retries, exception translation (404 -> ModelNotFound, ConnectionError -> OllamaConnectionError), non-retryable error fast-path, and router fallback.
- Git commit message: `feat(phase-8): implement robust LLM engineering with structured logging, retries, and error resilience`.

---

## Phase 9: FastAPI + Streamlit

### 1. Requirements & Dependencies
#### [MODIFY] `requirements.txt`
- Add `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `httpx>=0.24.0`, `streamlit>=1.25.0`.

### 2. FastAPI REST Service
#### [NEW] `src/api/models.py`
- `ReasonApiRequest`:
  - `question: str`
  - `profile: Optional[str] = None`
  - `mode: Optional[str] = "adaptive"` ("adaptive", "direct", "debate", "self_consistency")
  - `rounds: Optional[int] = 2`
- `ReasonApiResponse`:
  - `request_id: str`
  - `question: str`
  - `answer: str`
  - `explanation: str`
  - `strategy: str`
  - `difficulty: str`
  - `confidence: float`
  - `model: str`
  - `provider: str`
  - `total_calls: int`
  - `total_tokens: TokenUsage`
  - `latency_seconds: float`
  - `external_api_cost: float`
  - `metadata: Dict[str, Any]`
  - `debate_transcript: Optional[Dict[str, Any]]`
  - `judge_verdict: Optional[Dict[str, Any]]`
  - `self_consistency: Optional[Dict[str, Any]]`
- `HealthApiResponse`:
  - `status: str`
  - `provider: str`
  - `default_model: str`
  - `provider_healthy: bool`
#### [NEW] `src/api/app.py`
- Expose `GET /health` and `POST /reason`.
- Include request validation, CORS, and HTTP exception handling.

### 3. Streamlit Interactive Dashboard
#### [NEW] `src/ui/app.py`
- Modern, visually compelling UI (dark-mode aesthetic, custom styling):
  - Sidebar: Profile selector, Mode selection (Adaptive / Direct / Self-Consistency / Debate), threshold adjusters, round counters.
  - Main view: Query prompt input, sample prompt buttons (e.g. Easy math, Trick question, Ethical dilemma).
  - Live execution with spinner and telemetry ribbon (Strategy badge, difficulty, confidence, call count, token usage, latency, $0.00 cost).
  - Mode-specific tabs/visualizers:
    - **DIRECT**: Formatted answer card, step-by-step reasoning.
    - **SELF-CONSISTENCY**: Candidate voting chart, agreement score, individual generation cards.
    - **MULTI-AGENT DEBATE**: Round-by-round tabs showing Agent A & Agent B arguments, revision indicators, and Judge adjudication card with flaw analysis and winning agent.

### 4. Integration Tests & Commit for Phase 9
#### [NEW] `tests/test_api.py`
- Test `/health` endpoint and `/reason` endpoint using FastAPI `TestClient`.
- Git commit message: `feat(phase-9): implement FastAPI REST endpoints and Streamlit interactive dashboard`.

---

## Verification Plan

### Automated Tests
- `pytest tests/test_robustness.py -v`: Verify retries, backoff, exception handling, and router fallback.
- `pytest tests/test_api.py -v`: Verify FastAPI `/health` and `/reason` across all strategies.
- `pytest tests/ -v`: Verify all 45+ tests pass across all test suites.

### Manual Verification
1. **Phase 8 Robustness**:
   - Query with an invalid model name to verify graceful handling without unhandled crash.
   - Verify structured log output with `request_id` and latency measurements.
2. **Phase 9 FastAPI**:
   - Start FastAPI server: `python -m uvicorn src.api.app:app --port 8000`.
   - Send HTTP request: `curl http://localhost:8000/health` and `POST /reason`.
3. **Phase 9 Streamlit**:
   - Launch Streamlit: `python -m streamlit run src/ui/app.py --server.headless true`.
