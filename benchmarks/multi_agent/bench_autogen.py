"""AutoGen Multi-Agent Benchmark using OpenAI-compatible API.

Usage:
  python bench_autogen.py --url http://localhost:8100/v1 --model nano-vllm --label nano
"""
import argparse
import asyncio
import time
import os

os.environ["OPENAI_API_KEY"] = "not-needed"


async def run_autogen(api_url, model_name, num_turns=5):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    model_client = OpenAIChatCompletionClient(
        model=model_name,
        base_url=api_url,
        api_key="not-needed",
        model_info={"vision": False, "function_calling": False, "json_output": False, "family": "unknown"},
    )

    tasks = [
        ("Design a URL shortener with analytics", "Software Architect", "Backend Engineer"),
        ("Build a task management API", "Product Manager", "Full-stack Developer"),
        ("Create a chat system with encryption", "Security Engineer", "Systems Architect"),
    ]

    total_tokens = 0
    t_start = time.time()

    for task, role1, role2 in tasks:
        name1 = role1.lower().replace(" ", "_").replace("-", "_")
        name2 = role2.lower().replace(" ", "_").replace("-", "_")
        agent1 = AssistantAgent(
            name1,
            model_client=model_client,
            system_message=f"You are a {role1}. Discuss: {task}. Keep responses to 2-3 sentences.",
        )
        agent2 = AssistantAgent(
            name2,
            model_client=model_client,
            system_message=f"You are a {role2}. Discuss: {task}. Keep responses to 2-3 sentences.",
        )

        msg = f"Let's discuss: {task}. What's your approach?"
        for turn in range(num_turns):
            result1 = await agent1.run(task=msg)
            text1 = result1.messages[-1].content if result1.messages else ""
            total_tokens += len(str(text1).split())

            result2 = await agent2.run(task=str(text1))
            text2 = result2.messages[-1].content if result2.messages else ""
            total_tokens += len(str(text2).split())
            msg = str(text2)

    elapsed = time.time() - t_start
    await model_client.close()
    return total_tokens, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="engine")
    parser.add_argument("--turns", type=int, default=5)
    args = parser.parse_args()

    print(f"AutoGen Benchmark: {args.label}")
    print(f"  API: {args.url}, Model: {args.model}, Turns: {args.turns}")

    tokens, elapsed = asyncio.run(run_autogen(args.url, args.model, args.turns))
    print(f"[{args.label}] ~{tokens} words, {elapsed:.2f}s, ~{tokens/elapsed:.1f} words/s")


if __name__ == "__main__":
    main()
