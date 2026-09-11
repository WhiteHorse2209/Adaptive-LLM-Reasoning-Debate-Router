# Adaptive LLM Reasoning & Debate Router

## Overview
This project builds an adaptive reasoning system that dynamically determines how much computational effort to spend on a user query. It intelligently routes questions across three modes:
1. **Direct Mode:** Fast, zero-shot answer for easy/high-confidence queries.
2. **Self-Consistency:** Evaluates multiple parallel candidate generations for uncertain queries.
3. **Multi-Agent Debate:** Engages multiple agents in a structured, multi-round peer-review debate for hard queries.

The primary initial design focuses entirely on local-first LLM inference using **Ollama**, ensuring **zero external API costs** during development and evaluation, with computation performed on the user's local hardware.

---

## Architecture

```
                    USER QUESTION
                          │
                          ▼
                  DIRECT / ROUTER
                          │
                          ▼
                 LLM PROVIDER LAYER (Abstract)
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
    OLLAMA PROVIDER              (Optional Cloud)
  (Local GPU / CPU)
           │
           ▼
    STRUCTURED OUTPUT
  (Answer, Explanation, Telemetry)
```

---

## Phase 2: Ollama Provider & Single LLM Reasoning

Phase 2 implements the provider abstraction layer and the direct single-model reasoning pipeline.

### Core Modules
- **`src/provider/models.py`**: Pydantic data schemas (`LLMRequest`, `LLMResponse`, `TokenUsage`, `StructuredQAResponse`).
- **`src/provider/base.py`**: Abstract base class `LLMProvider` defining uniform `generate()`, `health_check()`, and cost calculation interfaces.
- **`src/provider/ollama.py`**: Full Ollama REST API integration (`/api/generate`) with automatic token count extraction (`prompt_eval_count`, `eval_count`), high-resolution wall-clock latency measurement, and robust error handling.
- **`src/provider/factory.py`**: Provider factory isolating model selection from application logic.
- **`src/reasoning/direct.py`**: `DirectReasoner` handling single-model inference and parsing answers/explanations into structured formats.
- **`src/main.py`**: CLI entry point for executing questions against configured model profiles.

---

## Quickstart & Installation

### 1. Prerequisites
- Install [Ollama](https://ollama.com/) locally.
- Start the Ollama server:
  ```bash
  ollama serve
  ```
- Pull local models (for example, `llama3.2` or `qwen3`):
  ```bash
  ollama pull llama3.2:latest
  ```

### 2. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Unit & Integration Tests
```bash
python -m pytest tests/ -v
```

### 4. Run Single LLM Question Answering

**Human-readable format:**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast --format text
```

**Structured JSON format:**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast --format json
```

Example JSON Output:
```json
{
  "answer": "100",
  "explanation": "To calculate 25 * 4, multiply 25 by 4: 25 + 25 + 25 + 25 = 100.",
  "model": "llama3.2:latest",
  "provider": "ollama",
  "token_usage": {
    "prompt_tokens": 58,
    "completion_tokens": 34,
    "total_tokens": 92
  },
  "latency_seconds": 1.4821,
  "external_api_cost": 0.0,
  "metadata": {
    "provider": "ollama",
    "done": true,
    "done_reason": "stop"
  }
}
```

---

## Configuration (`config.json`)

Model profiles and parameters are configured independently of application code:
```json
{
  "active_profile": "local_fast",
  "profiles": {
    "local_fast": {
      "provider": "ollama",
      "model": "llama3.2:latest",
      "temperature": 0.1,
      "max_tokens": 500,
      "timeout": 60,
      "api_base": "http://localhost:11434"
    },
    "local_reasoning": {
      "provider": "ollama",
      "model": "qwen3:1.7b",
      "temperature": 0.4,
      "max_tokens": 2000,
      "timeout": 120,
      "api_base": "http://localhost:11434"
    }
  },
  "router": {
    "confidence_threshold_high": 0.9,
    "confidence_threshold_low": 0.5,
    "consistency_samples": 5
  }
}
```
