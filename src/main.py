import argparse
import json
import sys

from src.config.config import load_config, get_active_profile
from src.provider.factory import get_provider
from src.reasoning.direct import DirectReasoner


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive LLM Reasoning & Debate Router - Single LLM Direct Reasoning (Phase 2)"
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
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
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

    # Run Direct Reasoner
    reasoner = DirectReasoner(provider)
    try:
        response = reasoner.answer(args.question)
    except Exception as e:
        print(f"Error during reasoning: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(response.model_dump(), indent=2))
    else:
        print("\n" + "=" * 60)
        print("QUESTION:")
        print(f"  {args.question}")
        print("\nEXPLANATION:")
        print(f"  {response.explanation}")
        print("\nFINAL ANSWER:")
        print(f"  {response.answer}")
        print("\nTELEMETRY:")
        print(f"  Model:            {response.model} ({response.provider})")
        print(f"  Latency:          {response.latency_seconds}s")
        print(f"  Tokens:           {response.token_usage.total_tokens} (prompt: {response.token_usage.prompt_tokens}, completion: {response.token_usage.completion_tokens})")
        print(f"  External Cost:    ${response.external_api_cost:.4f} (Local Hardware)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
