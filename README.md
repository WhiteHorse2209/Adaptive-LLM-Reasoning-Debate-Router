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
       DIRECT     SELF-CONSISTENCY  MULTI-AGENT DEBATE
    (1 LLM Call)   (N Gen Samples)   (Agent A ↔ Agent B)
          │               │               │
          │               │         Round 1 (Independent)
          │               │         Round 2 (Cross-Exam)
          │               │         Round 3 (Rebuttal)
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

### Phase 5: Multi-Agent Debate Foundation
- **`src/reasoning/debate_models.py`**: Data schemas including `AgentConfig`, `DebateTurn`, `DebateRound`, and `DebateTranscript`.
- **`src/reasoning/debate.py`**:
  - `DebateAgent`: Individual debater driven by distinct personas, evaluating opponent arguments, and actively defending or revising its conclusions.
  - `MultiAgentDebateEngine`: Genuine multi-round peer debate (NOT a sequential Solver -> Critic -> Verifier pipeline).
    - **Round 1 (Parallel Independent Discovery)**: Agents solve the question in isolation to eliminate anchoring bias.
    - **Round 2 (Mutual Cross-Examination)**: Agents review opponents' reasoning, challenge assumptions, defend valid deductions, or concede errors.
    - **Round 3 (Optional Convergence)**: Deepens debate until consensus or principled divergence.
- **`src/main.py`**: Standalone execution via `--mode debate` displaying round-by-round arguments, revisions, and consensus telemetry.

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

### 4. Running the Modes

**Mode 1: Direct Answer (Easy Query):**
```bash
python -m src.main --question "What is 25 * 4?" --profile local_fast
```

**Mode 2: Self-Consistency (Uncertain Query):**
```bash
python -m src.main --question "Which number is larger: 9.11 or 9.9?" --profile local_fast
```

**Mode 3: Multi-Agent Debate (Complex / Contested Query):**
```bash
python -m src.main --question "Should autonomous vehicles prioritize passenger safety or pedestrian safety in an unavoidable accident?" --mode debate --rounds 2 --profile local_fast
```

Example Debate Output:
```
======================================================================
GENUINE MULTI-AGENT DEBATE TRANSCRIPT
Question: Should autonomous vehicles prioritize passenger safety or pedestrian safety?
======================================================================

>>> ROUND 1 <<<
[Agent_A]:
Argument:
  From a utilitarian perspective, minimizing total casualties is paramount...
Proposed Answer: Prioritize pedestrian safety to minimize total harm.

[Agent_B]:
Argument:
  From a contractual perspective, manufacturers owe primary duty to passengers...
Proposed Answer: Prioritize passenger safety based on fiduciary duty.

>>> ROUND 2 <<<
[Agent_A]:
Argument:
  Agent B's argument overlooks the moral asymmetry between passengers...
Proposed Answer: Prioritize pedestrian safety.

[Agent_B] [REVISED ANSWER!]:
Argument:
  Agent A raises a valid point regarding the moral duty to unconsenting pedestrians...
Proposed Answer: Prioritize pedestrian safety with minimial harm routing.

======================================================================
DEBATE TELEMETRY & CONSENSUS:
  Consensus Reached:   YES
  Consensus Answer:    Prioritize pedestrian safety
  Total Agent Calls:   4
  Total Latency:       62.4s
  Total Tokens:        840
  External Cost:       $0.0000 (Zero API Cost - Local Hardware)
======================================================================
```

---

## Configuration (`config.json`)

Model profiles, router thresholds, and debate hyperparameters are configured independently of application code:
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
  },
  "debate": {
    "num_agents": 2,
    "num_rounds": 2,
    "agent_temperature": 0.7,
    "agents": [
      {
        "name": "Agent_A",
        "persona": "Analytical logician focused on rigorous first-principles derivation."
      },
      {
        "name": "Agent_B",
        "persona": "Critical empirical thinker focused on edge cases and counterexamples."
      }
    ]
  }
}
```
