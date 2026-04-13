"""MetaGPT-style Pipeline Benchmark using OpenAI-compatible API.

Simulates the PM → Architect → Engineer → QA pipeline.

Usage:
  python bench_metagpt.py --url http://localhost:8100/v1 --model nano-vllm --label nano
"""
import argparse
import time
import os
import requests
import json

os.environ["OPENAI_API_KEY"] = "not-needed"

AGENTS = [
    {"role": "Product Manager", "prompt": "Analyze the following requirement and write 3 user stories:"},
    {"role": "Software Architect", "prompt": "Based on these user stories, design the system architecture:"},
    {"role": "Senior Engineer", "prompt": "Based on this architecture, write the key implementation code:"},
    {"role": "QA Engineer", "prompt": "Review the following code and list potential issues:"},
]

TASKS = [
    "Build a URL shortener service with click analytics",
    "Create a task management API with user authentication",
    "Design a real-time chat system with message persistence",
]


def call_api(url, model, system, user_msg, temperature=0, max_tokens=200):
    """Call OpenAI-compatible chat completion API."""
    resp = requests.post(
        f"{url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage.get("completion_tokens", len(content.split()))


def run_pipeline(api_url, model_name, tasks):
    total_tokens = 0
    t_start = time.time()

    for task in tasks:
        prev_output = task
        for agent in AGENTS:
            system = f"You are a {agent['role']}. Be concise and specific."
            user_msg = f"{agent['prompt']}\n\n{prev_output}"
            content, n_tokens = call_api(api_url, model_name, system, user_msg)
            total_tokens += n_tokens
            prev_output = content

    elapsed = time.time() - t_start
    return total_tokens, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="engine")
    args = parser.parse_args()

    print(f"MetaGPT Pipeline Benchmark: {args.label}")
    print(f"  API: {args.url}, Model: {args.model}")
    print(f"  Tasks: {len(TASKS)}, Agents: {len(AGENTS)}")

    tokens, elapsed = run_pipeline(args.url, args.model, TASKS)
    print(f"[{args.label}] {tokens} tok, {elapsed:.2f}s, {tokens/elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
