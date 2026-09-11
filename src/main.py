import argparse
import json
import sys

from src.config.config import load_config
from src.provider.factory import get_provider
from src.reasoning.direct import DirectReasoner
from src.router.router import AdaptiveRouter


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive LLM Reasoning & Debate Router - Dynamic Mode & Direct Routing (Phase 3)"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What is 25 * 4?",
        help="The question to answer.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        default=None,
        help="Model profile name to use (defaults to active_profile in config.json).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["adaptive", "direct"],
        default="adaptive",
        help="Execution mode: 'adaptive' (routes based on difficulty/confidence) or 'direct' (unconditional direct reasoning).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (json or text).",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Select profile
    profile_name = args.profile or config.get("active_profile")
    if not profile_name or profile_name not in config.get("profiles", {}):
        print(
            f"Error: Profile '{profile_name}' not found in configuration profiles: "
            f"{list(config.get('profiles', {}).keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    profile_config = config["profiles"][profile_name]
    router_config = config.get("router", {})

    # Instantiate Provider via Abstraction Factory
    try:
        provider = get_provider(profile_config)
    except Exception as e:
        print(f"Error creating provider: {e}", file=sys.stderr)
        sys.exit(1)

    # Health Check
    if not provider.health_check():
        print(
            f"Warning: Provider '{provider.provider_name}' failed health check. "
            f"Ensure Ollama is running (`ollama serve`). Attempting generation anyway...",
            file=sys.stderr,
        )

    # Execute Reasoning Pipeline
    try:
        if args.mode == "adaptive":
            router = AdaptiveRouter(provider, router_config=router_config)
            response = router.route_and_solve(args.question)
        else:
            reasoner = DirectReasoner(provider)
            response = reasoner.answer(args.question)
    except Exception as e:
        print(f"Error during reasoning execution: {e}", file=sys.stderr)
        sys.exit(1)

    # Render Output
    if args.format == "json":
        print(json.dumps(response.model_dump(), indent=2))
    else:
        print("\n" + "=" * 65)
        print("QUESTION:")
        print(f"  {args.question}")
        print("\nEXPLANATION:")
        print(f"  {response.explanation}")
        print("\nFINAL ANSWER:")
        print(f"  {response.answer}")
        print("\nROUTING & CONFIDENCE TELEMETRY:")
        if hasattr(response, "strategy"):
            print(f"  Strategy:         {response.strategy.value}")
            print(f"  Difficulty:       {response.difficulty.value}")
            print(f"  Confidence:       {response.confidence.score:.2f} ({response.confidence.level})")
            print(f"  Confidence Note:  {response.confidence.justification}")
            print(f"  Calls Made:       {response.call_count}")
        print(f"  Model:            {response.model} ({response.provider})")
        print(f"  Latency:          {response.latency_seconds}s")
        print(
            f"  Tokens:           {response.token_usage.total_tokens} "
            f"(prompt: {response.token_usage.prompt_tokens}, completion: {response.token_usage.completion_tokens})"
        )
        print(f"  External Cost:    ${response.external_api_cost:.4f} (Zero API Cost - Local Hardware)")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
