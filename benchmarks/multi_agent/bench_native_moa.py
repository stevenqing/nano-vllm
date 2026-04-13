#!/usr/bin/env python3
"""
Benchmark: Native MoA API vs HTTP MoA vs vLLM HTTP MoA.

Compares throughput for Mixture-of-Agents workloads:
  1. nano-vllm native run_moa() — direct engine API, multi-prompt batch
  2. nano-vllm HTTP — async API server (bench_moa.py style)
  3. vLLM HTTP — vLLM server (bench_moa.py style)

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_native_moa.py --model ~/huggingface/Qwen3-0.6B/
"""
import argparse
import time
import sys
import os

PROMPTS = [
    "What are the main differences between Python and Rust programming languages?",
    "Explain the concept of attention mechanism in transformers in simple terms.",
    "Write a short story about a robot discovering emotions for the first time.",
    "What are three practical strategies for reducing carbon emissions in cities?",
    "Explain the mathematical concept of eigenvalues and their applications.",
]


def bench_native_moa(model_path, prompts, num_agents, num_layers, max_tokens):
    """Native MoA: all prompts batched, no HTTP overhead."""
    from nanovllm import LLM, SamplingParams

    llm = LLM(model_path)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    # Warmup
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=4))

    # Single prompt
    t0 = time.perf_counter()
    r1 = llm.run_moa(prompts[:1], num_agents=num_agents, num_layers=num_layers, sampling_params=sp)
    t_single = time.perf_counter() - t0

    # All prompts batched
    t0 = time.perf_counter()
    r_all = llm.run_moa(prompts, num_agents=num_agents, num_layers=num_layers, sampling_params=sp)
    t_batch = time.perf_counter() - t0

    # Sequential (1 at a time, for comparison)
    t0 = time.perf_counter()
    for p in prompts:
        llm.run_moa([p], num_agents=num_agents, num_layers=num_layers, sampling_params=sp)
    t_sequential = time.perf_counter() - t0

    llm.exit()
    return t_single, t_batch, t_sequential, r_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    n = len(PROMPTS)
    agents = args.agents
    layers = args.layers

    print(f"{'='*60}")
    print(f"  Native MoA Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Config: {layers} layers × {agents} agents, {args.max_tokens} max_tokens")
    print(f"  Prompts: {n}")
    print(f"{'='*60}")

    t_single, t_batch, t_sequential, results = bench_native_moa(
        args.model, PROMPTS, agents, layers, args.max_tokens)

    calls_per_prompt = (layers - 1) * agents + 1  # L1: agents, L2: 1

    print(f"\n  {'Mode':<30} | {'Time':>8} | {'Per prompt':>10} | {'Speedup':>8}")
    print(f"  {'-'*65}")
    print(f"  {'Sequential (1 at a time)':<30} | {t_sequential:>7.2f}s | {t_sequential/n:>9.2f}s | {'1.00x':>8}")
    print(f"  {'Single prompt':<30} | {t_single:>7.2f}s | {t_single:>9.2f}s | {t_sequential/n/t_single:>7.2f}x")
    print(f"  {'Batch ({n} prompts)':<30} | {t_batch:>7.2f}s | {t_batch/n:>9.2f}s | {t_sequential/n/(t_batch/n):>7.2f}x")

    print(f"\n  Key metrics:")
    print(f"    Sequential throughput:  {n/t_sequential:.2f} prompts/s")
    print(f"    Batch throughput:       {n/t_batch:.2f} prompts/s")
    print(f"    Batch speedup:          {t_sequential/t_batch:.2f}x")

    # Show sample output
    print(f"\n  Sample output (prompt 1):")
    print(f"    {results[0]['text'][:100]}...")


if __name__ == "__main__":
    main()
