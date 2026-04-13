#!/usr/bin/env python3
"""
Mixture-of-Agents (MoA) Benchmark for local inference engines.

Adapted from togethercomputer/MoA to use OpenAI-compatible local API endpoints.
Measures total latency and throughput for multi-layer multi-agent inference.

Usage:
    python bench_moa.py --url http://localhost:8100/v1 --model Qwen3-0.6B --label nano-vllm
"""
import argparse
import asyncio
import json
import time


AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

# Diverse prompts to test different capabilities
EVAL_PROMPTS = [
    "What are the main differences between Python and Rust programming languages?",
    "Explain the concept of attention mechanism in transformers in simple terms.",
    "Write a short story about a robot discovering emotions for the first time.",
    "What are three practical strategies for reducing carbon emissions in cities?",
    "Explain the mathematical concept of eigenvalues and their applications.",
]


async def call_llm(session, url, model, messages, temperature=0.7, max_tokens=256):
    """Make an async OpenAI-compatible chat completion call."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    
    async with session.post(f"{url}/chat/completions", json=payload, headers=headers, timeout=300) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def run_moa_layer(session, url, model, user_prompt, num_agents, prev_responses=None):
    """Run one MoA layer: multiple agents process the same prompt in parallel."""
    tasks = []
    for i in range(num_agents):
        if prev_responses:
            # Layer 2+: include previous responses in system prompt
            sys_prompt = AGGREGATOR_SYSTEM_PROMPT + "\n" + "\n".join(
                [f"{j+1}. {r}" for j, r in enumerate(prev_responses)]
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            # Layer 1: just the user prompt
            messages = [{"role": "user", "content": user_prompt}]
        tasks.append(call_llm(session, url, model, messages))
    
    results = await asyncio.gather(*tasks)
    return results


async def run_moa_full(session, url, model, user_prompt, num_layers, num_agents):
    """Run the full MoA pipeline: multiple layers of parallel agents."""
    responses = None
    for layer in range(num_layers - 1):
        responses = await run_moa_layer(session, url, model, user_prompt, num_agents, responses)
    
    # Final aggregation (single agent)
    sys_prompt = AGGREGATOR_SYSTEM_PROMPT + "\n" + "\n".join(
        [f"{i+1}. {r}" for i, r in enumerate(responses)]
    )
    final = await call_llm(session, url, model, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ], temperature=0, max_tokens=512)
    
    return final


async def benchmark(url, model, label, num_layers, num_agents, prompts):
    """Run the full MoA benchmark."""
    import aiohttp
    
    print(f"=== MoA Benchmark: {label} ===")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Layers: {num_layers}, Agents/layer: {num_agents}")
    print(f"  Prompts: {len(prompts)}")
    print()
    
    total_time = 0
    total_calls = 0
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("  Warmup...")
        await call_llm(session, url, model, [{"role": "user", "content": "Hi"}], max_tokens=8)
        
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            result = await run_moa_full(session, url, model, prompt, num_layers, num_agents)
            elapsed = time.perf_counter() - t0
            
            # Count API calls: (num_layers-1) * num_agents + 1 aggregation
            calls = (num_layers - 1) * num_agents + 1
            total_calls += calls
            total_time += elapsed
            
            print(f"  Prompt {i+1}: {elapsed:.2f}s ({calls} API calls)")
            print(f"    Q: {prompt[:60]}...")
            print(f"    A: {result[:80]}...")
            print()
    
    avg_time = total_time / len(prompts)
    calls_per_sec = total_calls / total_time
    
    print(f"=== MoA Results: {label} ===")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg per prompt: {avg_time:.2f}s")
    print(f"  Total API calls: {total_calls}")
    print(f"  API calls/sec: {calls_per_sec:.2f}")
    print()
    
    return {
        "label": label,
        "model": model,
        "num_layers": num_layers,
        "num_agents": num_agents,
        "num_prompts": len(prompts),
        "total_time_s": total_time,
        "avg_time_per_prompt_s": avg_time,
        "total_api_calls": total_calls,
        "api_calls_per_sec": calls_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="MoA Benchmark")
    parser.add_argument("--url", required=True, help="OpenAI-compatible API base URL (e.g. http://localhost:8100/v1)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--label", default="engine", help="Label for this run")
    parser.add_argument("--layers", type=int, default=2, help="Number of MoA layers")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents per layer")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    result = asyncio.run(benchmark(
        args.url, args.model, args.label,
        args.layers, args.agents, EVAL_PROMPTS,
    ))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
