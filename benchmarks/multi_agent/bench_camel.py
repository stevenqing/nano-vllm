"""CAMEL Role-Playing Benchmark using the actual CAMEL framework.

Starts either nano-vllm or vLLM as an OpenAI-compatible API server,
then runs CAMEL's RolePlaying society through it.

Usage:
  # Start nano-vllm server first:
  #   python nano_api_server.py --model ~/huggingface/Qwen3-8B/ --port 8100
  # Or start vLLM server:
  #   python -m vllm.entrypoints.openai.api_server --model ~/huggingface/Qwen3-8B/ --port 8200

  # Then run benchmark:
  python bench_camel.py --url http://localhost:8100/v1 --model nano-vllm --label nano
  python bench_camel.py --url http://localhost:8200/v1 --model Qwen3-8B --label vllm
"""
import argparse
import time
import os

os.environ["OPENAI_API_KEY"] = "not-needed"


def run_camel_roleplay(api_url, model_name, num_turns=5):
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_name,
        model_config_dict={"temperature": 0, "max_tokens": 150},
        url=api_url,
        api_key="not-needed",
    )

    tasks = [
        ("Python Programmer", "CTO", "Design a microservice architecture for an e-commerce platform."),
        ("Data Scientist", "ML Engineer", "Build a recommendation system using collaborative filtering."),
        ("Product Manager", "Senior Engineer", "Plan a real-time notification system for a social media app."),
    ]

    total_tokens = 0
    t_start = time.time()

    for user_role, asst_role, task_desc in tasks:
        user_sys = f"You are a {user_role}. Discuss with your partner to solve: {task_desc}. Keep responses to 2-3 sentences."
        asst_sys = f"You are a {asst_role}. Discuss with your partner to solve: {task_desc}. Keep responses to 2-3 sentences."

        user_agent = ChatAgent(system_message=user_sys, model=model)
        asst_agent = ChatAgent(system_message=asst_sys, model=model)

        # User starts
        user_msg = user_agent.step(f"Let's discuss: {task_desc}. What's your initial approach?")
        user_text = user_msg.msgs[0].content if user_msg.msgs else ""
        total_tokens += len(user_text.split())  # approx

        for turn in range(num_turns - 1):
            # Assistant responds
            asst_msg = asst_agent.step(user_text)
            asst_text = asst_msg.msgs[0].content if asst_msg.msgs else ""
            total_tokens += len(asst_text.split())

            # User responds
            user_msg = user_agent.step(asst_text)
            user_text = user_msg.msgs[0].content if user_msg.msgs else ""
            total_tokens += len(user_text.split())

    elapsed = time.time() - t_start
    return total_tokens, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model name for the API")
    parser.add_argument("--label", default="engine", help="Label for output")
    parser.add_argument("--turns", type=int, default=5, help="Turns per task")
    args = parser.parse_args()

    print(f"CAMEL Role-Playing Benchmark: {args.label}")
    print(f"  API: {args.url}")
    print(f"  Model: {args.model}")
    print(f"  Turns: {args.turns}")
    print()

    tokens, elapsed = run_camel_roleplay(args.url, args.model, args.turns)
    print(f"[{args.label}] ~{tokens} words, {elapsed:.2f}s, ~{tokens/elapsed:.1f} words/s")


if __name__ == "__main__":
    main()
