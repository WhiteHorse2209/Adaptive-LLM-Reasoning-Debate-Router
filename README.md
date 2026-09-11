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

### Phase 4: Self-Consistency Reasoning
- **`src/reasoning/models.py`**: `CandidateSolution` and `SelfConsistencyResult` schemas tracking each candidate reasoning chain, sample token usage, individual latency, and overall consensus.
- **`src/reasoning/consistency.py`**: `SelfConsistencyReasoner` executing $N$ independent generation paths with non-zero temperature ($T=0.7$), canonicalizing candidate outputs, and selecting the consensus answer via majority voting:
  $$\text{Agreement} = \frac{\max_a \text{count}(a)}{N}$$
- **`src/router/router.py`**: Seamlessly executes Self-Consistency for queries identified as `UNCERTAIN`, tracking all $1 + N$ calls, accumulated tokens, latency, and sample distributions.
- **`src/main.py`**: CLI displays candidate vote distributions, agreement percentages, and individual sample solutions.

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

**Easy Query (Automatically routed to DIRECT - 1 model call):**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast
```
Output:
```
=================================================================
QUESTION:
  What is 25 * 4?

FINAL ANSWER:
  100

ROUTING & CONFIDENCE TELEMETRY:
  Strategy:         DIRECT
  Difficulty:       EASY
  Confidence:       1.00 (HIGH)
  Calls Made:       1
  Model:            llama3.2:latest (ollama)
  Tokens:           209 (prompt: 136, completion: 73)
  External Cost:    $0.0000 (Zero API Cost - Local Hardware)
=================================================================
```

**Uncertain Query (Automatically routed to SELF-CONSISTENCY - 1 + N calls):**
```bash
python -m src.main --question "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?" --profile local_fast
```
Output:
```
=================================================================
ROUTING & CONFIDENCE TELEMETRY:
  Strategy:         SELF_CONSISTENCY
  Difficulty:       UNCERTAIN
  Confidence:       0.65 (MEDIUM)
  Calls Made:       4
  Agreement Score:  66.7%
  Vote Counts:      {'5 cents': 2, '10 cents': 1}
  Candidate Samples:
    Sample 1:       5 cents
    Sample 2:       10 cents
    Sample 3:       5 cents
  Model:            llama3.2:latest (ollama)
  External Cost:    $0.0000 (Zero API Cost - Local Hardware)
=================================================================
```

---

## Configuration (`config.json`)

Model profiles, router thresholds, and self-consistency parameters are configured independently of application code:
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
    "consistency_samples": 3,
    "sample_temperature": 0.7
  }
}
```
