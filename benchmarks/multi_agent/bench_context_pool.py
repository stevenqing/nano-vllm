"""Benchmark: Context Pool + SCA L1 for Heterogeneous Multi-Agent.

Compares:
1. Baseline: each agent prefills [Context + System + Task] independently
2. Context Pool: shared context KV fork + only prefill suffix
3. Context Pool + L1 reorder: shared blocks sorted for L2 locality
"""
import os
import time
from random import randint, seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from nanovllm import LLM, SamplingParams


def build_agents(num_agents, context_len, system_len, task_len):
    """Generate random agent data: shared context + per-agent system+task."""
    seed(42)
    context = [randint(0, 10000) for _ in range(context_len)]
    agents = []
    for i in range(num_agents):
        system = [randint(0, 10000) for _ in range(system_len)]
        task = [randint(0, 10000) for _ in range(task_len)]
        agents.append({"system": system, "task": task})
    return context, agents


def bench_baseline(llm, context, agents, output_len):
    """Each agent prefills [System + Context + Task] independently (no prefix sharing)."""
    prompts = [a["system"] + context + a["task"] for a in agents]
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
    t = time.time()
    results = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.time() - t
    total_tokens = sum(len(r["token_ids"]) for r in results)
    return total_tokens, elapsed


def bench_reorder(llm, context, agents, output_len):
    """Prompt Reorder: [Context + System + Task] — prefix cache auto-shares context."""
    prompts = [context + a["system"] + a["task"] for a in agents]
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
    t = time.time()
    results = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.time() - t
    total_tokens = sum(len(r["token_ids"]) for r in results)
    return total_tokens, elapsed


def bench_context_pool(llm, context, agents, output_len):
    """Shared context KV fork + only prefill suffix per agent."""
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
    t = time.time()
    context_id = llm.cache_context(context)
    suffixes = [a["system"] + a["task"] for a in agents]
    results = llm.generate_with_context(suffixes, sp, context_id=context_id)
    llm.release_context(context_id)
    elapsed = time.time() - t
    total_tokens = sum(len(r["token_ids"]) for r in results)
    return total_tokens, elapsed


CONFIGS = [
    # (num_agents, context_len, system_len, task_len, output_len)
    (3,   500,  100, 50, 64),
    (3,  1000,  200, 50, 64),
    (5,  1000,  200, 50, 64),
    (10, 1000,  200, 50, 64),
    (3,  2000,  200, 50, 64),
    (5,  2000,  100, 50, 128),
]


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # Warmup
    llm.generate(["Warmup"], SamplingParams(max_tokens=8))

    print(f"{'Config':<40s} | {'Baseline':>12s} | {'Reorder':>12s} | {'CtxPool':>12s} | {'Pool/Base':>9s}")
    print("-" * 95)

    for cfg in CONFIGS:
        n_agents, ctx_len, sys_len, task_len, out_len = cfg
        config_key = f"A{n_agents}_C{ctx_len}_S{sys_len}_T{task_len}_O{out_len}"
        context, agents = build_agents(n_agents, ctx_len, sys_len, task_len)

        # Baseline: [System + Context + Task] — no prefix sharing
        tok_base, t_base = bench_baseline(llm, context, agents, out_len)

        # Reorder: [Context + System + Task] — prefix cache sharing
        tok_reord, t_reord = bench_reorder(llm, context, agents, out_len)

        # Context Pool: explicit KV fork
        tok_pool, t_pool = bench_context_pool(llm, context, agents, out_len)

        speedup = t_base / t_pool if t_pool > 0 else 0
        print(f"  {config_key:<38s} | {tok_base/t_base:>9.1f}t/s | {tok_reord/t_reord:>9.1f}t/s | {tok_pool/t_pool:>9.1f}t/s | {speedup:>8.2f}x")

    print()


if __name__ == "__main__":
    main()
