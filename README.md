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
                 CONFIDENCE ESTIMATOR
              (Score: 0.0-1.0, Difficulty)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      [EASY]         [UNCERTAIN]       [HARD]
   Score >= 0.80    0.50 <= S < 0.80  Score < 0.50
          │               │               │
          ▼               ▼               ▼
       MODE 1          MODE 2          MODE 3
       DIRECT     SELF-CONSISTENCY     DEBATE
    (1 LLM Call)   (N Gen Samples)   (Multi-Round)
          │               │               │
          └───────────────┼───────────────┘
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
    STRUCTURED ROUTED OUTPUT
  (Answer, Strategy, Confidence, Call Count, Tokens, Latency)
```

---

## Core Components

### Phase 2: Provider Abstraction & Single LLM Direct Reasoning
- **`src/provider/models.py`**: Pydantic data schemas (`LLMRequest`, `LLMResponse`, `TokenUsage`, `StructuredQAResponse`).
- **`src/provider/base.py`**: Abstract base class `LLMProvider` defining uniform `generate()`, `health_check()`, and cost calculation interfaces.
- **`src/provider/ollama.py`**: Full Ollama REST API integration (`/api/generate`) with automatic token count extraction (`prompt_eval_count`, `eval_count`), high-resolution wall-clock latency measurement, and robust error handling.
- **`src/provider/factory.py`**: Provider factory isolating model selection from application logic.
- **`src/reasoning/direct.py`**: `DirectReasoner` handling single-model inference and parsing answers/explanations into structured formats.

### Phase 3: Confidence Estimation & Adaptive Direct Routing
- **`src/router/models.py`**: Enums for `DifficultyLevel` (`EASY`, `UNCERTAIN`, `HARD`), `ReasoningStrategy` (`DIRECT`, `SELF_CONSISTENCY`, `MULTI_AGENT_DEBATE`), `ConfidenceAssessment`, and `RoutedResponse`.
- **`src/router/estimator.py`**: `ConfidenceEstimator` providing single-pass difficulty classification, verbalized confidence extraction (0.0 to 1.0), and justification notes with resilient parsing and fallback heuristics.
- **`src/router/router.py`**: `AdaptiveRouter` comparing confidence against configurable thresholds. Easy queries immediately route to `DIRECT` mode (1 call, 0 extra agents), intelligently preventing unnecessary compute.
- **`src/main.py`**: CLI entry point supporting `--mode {adaptive, direct}` and `--format {text, json}`.

---

## Quickstart & Installation

### 1. Prerequisites
- Install [Ollama](https://ollama.com/) locally.
- Start the Ollama server:
  ```bash
  ollama serve
  ```
- Pull local models (e.g., `llama3.2` or `qwen3`):
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

### 4. Run Question Answering & Adaptive Routing

**Adaptive Routing Mode (Default - dynamically routes based on confidence):**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast --mode adaptive
```

Example Output for Easy Query (Routed to DIRECT):
```
=================================================================
QUESTION:
  What is 25 * 4?

EXPLANATION:
  To find the product of 25 and 4, we multiply these two numbers together.

FINAL ANSWER:
  100

ROUTING & CONFIDENCE TELEMETRY:
  Strategy:         DIRECT
  Difficulty:       EASY
  Confidence:       1.00 (HIGH)
  Confidence Note:  This is a basic arithmetic operation that can be solved with ease.
  Calls Made:       1
  Model:            llama3.2:latest (ollama)
  Latency:          1.42s
  Tokens:           205 (prompt: 136, completion: 69)
  External Cost:    $0.0000 (Zero API Cost - Local Hardware)
=================================================================
```

**JSON Output Format:**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast --format json
```

---

## Configuration (`config.json`)

Model profiles and router thresholds are configured independently of application code:
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
    "confidence_threshold_high": 0.80,
    "confidence_threshold_low": 0.50,
    "estimation_mode": "single_pass",
    "consistency_samples": 5
  }
}
```
