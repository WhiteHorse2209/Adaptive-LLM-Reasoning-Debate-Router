# Adaptive LLM Reasoning & Debate Router

## Overview
This project builds an adaptive reasoning system that dynamically determines how much computational effort to spend on a user query. It intelligently routes questions across three modes:
1. **Direct Mode:** Fast, zero-shot answer for easy/high-confidence queries.
2. **Self-Consistency:** Evaluates multiple parallel candidate generations for uncertain queries.
3. **Multi-Agent Debate:** Engages multiple agents in a structured, multi-round peer-review debate for hard queries.

The primary initial design focuses entirely on local-first LLM inference using **Ollama**, ensuring zero external API costs during the foundational development and benchmark phases. 

## Phase 1 Foundation
This initial project setup establishes the structure and basic configuration mechanism to interact with the LLM backend.

### Project Structure
- `src/config`: JSON-based configuration management mapping local model profiles.
- `src/provider`: Contains the generic LLM abstraction (and the Ollama-specific implementation).
- `src/reasoning`: Reasoning strategies (Direct, Consistency, Debate).
- `src/router`: Adaptive routing logic based on difficulty.

### Installation
1. Install [Ollama](https://ollama.com/) locally.
2. Pull your desired models (e.g., `ollama pull llama3.2:1b`).
3. Set up the Python virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
