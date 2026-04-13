#!/usr/bin/env python3
"""
Native Pipeline Benchmarks for all multi-agent patterns.

Uses generate_pipeline() to batch across tasks — each task's agents
at the same stage decode simultaneously, maximizing GPU utilization.

Patterns:
  - CAMEL-style debate: 2 agents alternate (serial chain, batched across tasks)
  - AutoGen-style: same as CAMEL debate
  - MoA: fan-out → aggregate (already supported)

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_pipeline_all.py --model ~/huggingface/Qwen3-0.6B/
"""
import argparse
import time
import sys
import os

TASKS = [
    ("Python Programmer", "CTO", "Design a microservice architecture for an e-commerce platform."),
    ("Data Scientist", "ML Engineer", "Build a recommendation system using collaborative filtering."),
    ("Product Manager", "Senior Engineer", "Plan a real-time notification system for a social media app."),
]

MOA_PROMPTS = [
    "What are the main differences between Python and Rust programming languages?",
    "Explain the concept of attention mechanism in transformers in simple terms.",
    "Write a short story about a robot discovering emotions for the first time.",
    "What are three practical strategies for reducing carbon emissions in cities?",
    "Explain the mathematical concept of eigenvalues and their applications.",
]


def bench_debate_native(llm, tasks, num_turns=5, max_tokens=150):
    """CAMEL/AutoGen-style debate using run_debate.
    All tasks batched — BS = len(tasks) per turn."""
    from nanovllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    t0 = time.perf_counter()
    transcripts = llm.run_debate(tasks, num_turns=num_turns, sampling_params=sp)
    elapsed = time.perf_counter() - t0

    total_words = sum(len(t.split()) for conv in transcripts for t in conv)
    return total_words, elapsed


def bench_debate_sequential(llm, tasks, num_turns=5, max_tokens=150):
    """Sequential debate: one task at a time (like CAMEL/AutoGen via HTTP)."""
    from nanovllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    total_words = 0
    t0 = time.perf_counter()

    for user_role, asst_role, task_desc in tasks:
        user_sys = f"You are a {user_role}. Discuss: {task_desc}. Keep responses to 2-3 sentences."
        asst_sys = f"You are a {asst_role}. Discuss: {task_desc}. Keep responses to 2-3 sentences."

        def make_msgs(sys_msg, content):
            text = llm.tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True)
            return llm.tokenizer.encode(text)

        # Turn 1: user
        ids = make_msgs(user_sys, f"Let's discuss: {task_desc}. What's your initial approach?")
        r = llm.generate([ids], sp, use_tqdm=False)
        text = r[0]["text"]
        total_words += len(text.split())

        for turn in range(num_turns - 1):
            # Assistant turn
            ids = make_msgs(asst_sys, text)
            r = llm.generate([ids], sp, use_tqdm=False)
            text = r[0]["text"]
            total_words += len(text.split())

            # User turn
            ids = make_msgs(user_sys, text)
            r = llm.generate([ids], sp, use_tqdm=False)
            text = r[0]["text"]
            total_words += len(text.split())

    elapsed = time.perf_counter() - t0
    return total_words, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    print(f"{'='*65}")
    print(f"  Native Pipeline Benchmark: All Patterns")
    print(f"  Model: {args.model}")
    print(f"  Debate: {len(TASKS)} tasks × {args.turns} turns")
    print(f"  MoA: {len(MOA_PROMPTS)} prompts × 3 agents × 2 layers")
    print(f"{'='*65}")

    llm = LLM(args.model)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Warmup
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=4))

    # 1. Debate sequential (like CAMEL/AutoGen via HTTP, BS=1)
    w1, t1 = bench_debate_sequential(llm, TASKS, args.turns, args.max_tokens)

    # 2. Debate pipeline (all tasks batched, BS=3)
    w2, t2 = bench_debate_native(llm, TASKS, args.turns, args.max_tokens)

    # 3. MoA sequential
    t3_start = time.perf_counter()
    for p in MOA_PROMPTS:
        llm.run_moa([p], num_agents=3, num_layers=2, sampling_params=sp)
    t3 = time.perf_counter() - t3_start

    # 4. MoA batch
    t4_start = time.perf_counter()
    llm.run_moa(MOA_PROMPTS, num_agents=3, num_layers=2, sampling_params=sp)
    t4 = time.perf_counter() - t4_start

    print(f"\n  {'Pattern':<35} | {'Time':>8} | {'Per task':>10} | {'Speedup':>8}")
    print(f"  {'-'*70}")
    print(f"  {'Debate sequential (BS=1)':<35} | {t1:>7.2f}s | {t1/len(TASKS):>9.2f}s | {'1.00x':>8}")
    print(f"  {'Debate pipeline (BS=3)':<35} | {t2:>7.2f}s | {t2/len(TASKS):>9.2f}s | {t1/t2:>7.2f}x")
    print(f"  {'MoA sequential (BS=3+1)':<35} | {t3:>7.2f}s | {t3/len(MOA_PROMPTS):>9.2f}s | {'1.00x':>8}")
    print(f"  {'MoA batch-5 (BS=15+5)':<35} | {t4:>7.2f}s | {t4/len(MOA_PROMPTS):>9.2f}s | {t3/t4:>7.2f}x")

    print(f"\n  Total words generated:")
    print(f"    Debate seq: ~{w1}, Debate pipeline: ~{w2}")

    llm.exit()


if __name__ == "__main__":
    main()
