#!/usr/bin/env python3
"""
Scalable Multi-Agent Benchmark: varies number of parallel agents.

Tests how inference throughput scales with agent count — the key differentiator
for multi-agent inference engines. More concurrent agents = higher batch size
= better GPU utilization.

Usage:
    python bench_scale_agents.py --url http://localhost:8100/v1 --model Qwen3-0.6B --label nano-vllm
"""
import argparse
import asyncio
import json
import time

import aiohttp

SYSTEM_PROMPTS = [
    "You are a software architect. Design systems concisely in 2-3 sentences.",
    "You are a backend engineer. Implement solutions concisely in 2-3 sentences.",
    "You are a QA engineer. Find issues concisely in 2-3 sentences.",
    "You are a DevOps engineer. Plan deployments concisely in 2-3 sentences.",
    "You are a security engineer. Identify risks concisely in 2-3 sentences.",
    "You are a data scientist. Analyze data concisely in 2-3 sentences.",
    "You are a product manager. Define requirements concisely in 2-3 sentences.",
    "You are a UX designer. Propose designs concisely in 2-3 sentences.",
    "You are a frontend engineer. Build interfaces concisely in 2-3 sentences.",
    "You are a database admin. Optimize queries concisely in 2-3 sentences.",
    "You are a ML engineer. Build models concisely in 2-3 sentences.",
    "You are a tech lead. Review code concisely in 2-3 sentences.",
    "You are a CTO. Make decisions concisely in 2-3 sentences.",
    "You are a solutions architect. Integrate systems concisely in 2-3 sentences.",
    "You are a performance engineer. Optimize systems concisely in 2-3 sentences.",
    "You are a cloud architect. Design infrastructure concisely in 2-3 sentences.",
]

TASK = "Design and implement a real-time analytics dashboard for monitoring microservice health metrics."


async def call_agent(session, url, model, system_prompt, user_msg, max_tokens=150):
    """Single agent API call."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    async with session.post(f"{url}/chat/completions", json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=300)) as resp:
        data = await resp.json()
    latency = time.perf_counter() - t0
    content = data["choices"][0]["message"]["content"]
    output_tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))
    return content, latency, output_tokens


async def run_parallel_agents(url, model, num_agents, num_rounds=3):
    """Run num_agents in parallel for num_rounds, measure throughput."""
    prompts = [SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)] for i in range(num_agents)]

    async with aiohttp.ClientSession() as session:
        # Warmup
        await call_agent(session, url, model, "You are helpful.", "Hi", max_tokens=5)

        total_tokens = 0
        total_time = 0

        for round_idx in range(num_rounds):
            msg = TASK if round_idx == 0 else f"Continue the discussion. Round {round_idx+1}."

            t0 = time.perf_counter()
            tasks = [call_agent(session, url, model, p, msg) for p in prompts]
            results = await asyncio.gather(*tasks)
            round_time = time.perf_counter() - t0

            round_tokens = sum(r[2] for r in results)
            total_tokens += round_tokens
            total_time += round_time

        return total_tokens, total_time


async def benchmark(url, model, label, agent_counts, num_rounds=3):
    """Run benchmark across different agent counts."""
    print(f"=== Scale Agent Benchmark: {label} ===")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Rounds per config: {num_rounds}")
    print()

    results = []
    print(f"  {'Agents':>6} | {'Tokens':>8} | {'Time':>8} | {'tok/s':>8} | {'Latency':>8}")
    print("  " + "-" * 52)

    for n_agents in agent_counts:
        tokens, elapsed = await run_parallel_agents(url, model, n_agents, num_rounds)
        tps = tokens / elapsed
        avg_latency = elapsed / (n_agents * num_rounds) * 1000
        print(f"  {n_agents:>6} | {tokens:>8} | {elapsed:>7.1f}s | {tps:>7.0f} | {avg_latency:>6.0f}ms")
        results.append({
            "num_agents": n_agents, "total_tokens": tokens,
            "total_time_s": elapsed, "throughput_tok_s": tps,
            "avg_latency_ms": avg_latency,
        })

    print()
    return {"label": label, "model": model, "num_rounds": num_rounds, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="engine")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    agent_counts = [1, 2, 4, 8]
    if args.max_agents >= 16:
        agent_counts.append(16)

    result = asyncio.run(benchmark(args.url, args.model, args.label,
                                    agent_counts, args.rounds))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
