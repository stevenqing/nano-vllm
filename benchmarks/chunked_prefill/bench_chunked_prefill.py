#!/usr/bin/env python3
"""
Chunked Prefill Benchmark: Nano-vLLM vs vLLM

Tests the key chunked prefill scenario: new requests arrive while the engine
is actively decoding existing requests (continuous serving pattern).

Metrics:
  - Throughput (tok/s) with staggered arrivals
  - Time-to-first-token (TTFT) for late-arriving requests
  - Total completion time for a multi-agent pipeline

Scenarios:
  1. Staggered Arrivals: N requests arrive at different times
  2. Multi-Agent Fan-out: 3 agents process shared context in parallel
  3. Multi-Agent Pipeline: A → B → C serial dependency
"""
import os
import sys
import gc
import json
import time
import subprocess
import platform
from random import randint, seed
from dataclasses import dataclass, asdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch


def get_env_report():
    import flash_attn, triton, transformers, xxhash
    try:
        import vllm; vllm_ver = vllm.__version__
    except: vllm_ver = "N/A"
    try:
        import flashinfer; flashinfer_ver = flashinfer.__version__
    except: flashinfer_ver = "N/A"
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_attn.__version__,
        "triton": triton.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm_ver,
        "flashinfer": flashinfer_ver,
        "gpu_name": gpu.name,
        "gpu_memory_gb": f"{gpu.total_memory / 1024**3:.1f}",
    }


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_prompts(n, prompt_len, random_seed=42):
    seed(random_seed)
    return [[randint(0, 10000) for _ in range(prompt_len)] for _ in range(n)]


# ========== Scenario 1: Staggered Arrivals ==========
# Requests arrive in waves. Tests how well the engine handles prefill
# of new requests while decoding existing ones.

def bench_nanovllm_staggered(llm, prompts, output_len, arrival_gap):
    """Submit requests in waves, measuring total time and per-request TTFT."""
    from nanovllm import SamplingParams
    from nanovllm.engine.sequence import Sequence, SequenceStatus

    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    seqs = []
    ttfts = [None] * len(prompts)
    finish_times = [None] * len(prompts)
    next_arrival = 0
    arrived = 0

    t_start = time.perf_counter()

    # Submit first batch
    seq = Sequence(prompts[0], sp)
    llm.scheduler.add(seq)
    seqs.append(seq)
    arrived = 1

    steps = 0
    mixed_steps = 0
    while True:
        step_seqs, num_pf = llm.scheduler.schedule()
        for s, d in llm.scheduler.block_manager.get_pending_copies():
            llm.model_runner.call("copy_block_kv", s, d)
        token_ids = llm.model_runner.call("run", step_seqs, num_pf)
        llm.scheduler.postprocess(step_seqs, token_ids, num_pf)
        steps += 1

        if num_pf > 0 and len(step_seqs) > num_pf:
            mixed_steps += 1

        # Record TTFT
        now = time.perf_counter()
        for i, seq in enumerate(seqs):
            if ttfts[i] is None and seq.num_completion_tokens > 0:
                ttfts[i] = now - t_start

        # Staggered arrival: submit next request every `arrival_gap` decode steps
        if arrived < len(prompts) and steps % arrival_gap == 0:
            new_seq = Sequence(prompts[arrived], sp)
            llm.scheduler.add(new_seq)
            seqs.append(new_seq)
            arrived += 1

        # Record finish times
        for i, seq in enumerate(seqs):
            if finish_times[i] is None and seq.is_finished:
                finish_times[i] = now - t_start

        if all(ft is not None for ft in finish_times[:arrived]) and arrived == len(prompts):
            break

    elapsed = time.perf_counter() - t_start
    total_tokens = sum(s.num_completion_tokens for s in seqs)
    return {
        "total_time_ms": elapsed * 1000,
        "throughput_tok_s": total_tokens / elapsed,
        "ttfts_ms": [t * 1000 for t in ttfts],
        "avg_ttft_ms": sum(ttfts) / len(ttfts) * 1000,
        "mixed_steps": mixed_steps,
        "total_steps": steps,
        "total_tokens": total_tokens,
    }


def bench_vllm_staggered(llm, prompts, output_len, arrival_gap):
    """vLLM version: use internal engine step loop for staggered arrivals."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)

    t_start = time.perf_counter()

    # vLLM doesn't easily expose step-level control in offline mode.
    # For fairness, we use the generate() API with all prompts at once
    # (vLLM's internal scheduler handles chunked prefill + continuous batching)
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        sp, use_tqdm=False)

    elapsed = time.perf_counter() - t_start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "total_time_ms": elapsed * 1000,
        "throughput_tok_s": total_tokens / elapsed,
        "total_tokens": total_tokens,
    }


# ========== Scenario 2: Online-style generate() ==========
# Submit all prompts at once via the high-level generate() API.
# Both engines handle scheduling internally.

def bench_nanovllm_batch(llm, prompts, output_len):
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    t = time.perf_counter()
    results = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.perf_counter() - t
    total_tokens = sum(len(r["token_ids"]) for r in results)
    return {
        "total_time_ms": elapsed * 1000,
        "throughput_tok_s": total_tokens / elapsed,
        "total_tokens": total_tokens,
    }


def bench_vllm_batch(llm, prompts, output_len):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    t = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    elapsed = time.perf_counter() - t
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "total_time_ms": elapsed * 1000,
        "throughput_tok_s": total_tokens / elapsed,
        "total_tokens": total_tokens,
    }


# ========== Scenario 3: Multi-agent pipeline ==========
# A → B → C serial: each agent's output feeds into the next.

def bench_nanovllm_pipeline(llm, base_prompts, output_len, num_agents=3):
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    total_tokens = 0
    t = time.perf_counter()

    for prompt in base_prompts:
        context = list(prompt)
        for agent_idx in range(num_agents):
            result = llm.generate([context], sp, use_tqdm=False)[0]
            total_tokens += len(result["token_ids"])
            context = context + result["token_ids"] + [0] * 20  # separator

    elapsed = time.perf_counter() - t
    return {
        "total_time_ms": elapsed * 1000,
        "throughput_tok_s": total_tokens / elapsed,
        "total_tokens": total_tokens,
    }


def bench_vllm_pipeline(llm, base_prompts, output_len, num_agents=3):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    total_tokens = 0
    t = time.perf_counter()

    for prompt in base_prompts:
        context = list(prompt)
        for agent_idx in range(num_agents):
            outputs = llm.generate(
                [{"prompt_token_ids": context}], sp, use_tqdm=False)
            result_tokens = list(outputs[0].outputs[0].token_ids)
            total_tokens += len(result_tokens)
            context = context + result_tokens + [0] * 20

    elapsed = time.perf_counter() - t
    return {
        "total_time_ms": elapsed * 1000,
        "throughput_tok_s": total_tokens / elapsed,
        "total_tokens": total_tokens,
    }


# ========== Sweep Configs ==========

CONFIGS = [
    # (num_prompts, prompt_len, output_len, arrival_gap)
    # Small prompts, many requests
    (1,   100, 200, 10),
    (4,   100, 200, 10),
    (8,   100, 200, 10),
    (16,  100, 200, 10),
    (32,  100, 200, 10),
    # Longer prompts
    (4,   500, 200, 10),
    (8,   500, 200, 10),
    (16,  500, 200, 10),
    # Long prompts + short outputs (prefill-heavy)
    (4,  1000, 50,  10),
    (8,  1000, 50,  10),
    (16, 1000, 50,  10),
    # Short prompts + long outputs (decode-heavy)
    (4,   50,  500, 10),
    (8,   50,  500, 10),
    (16,  50,  500, 10),
]

PIPELINE_CONFIGS = [
    # (num_conversations, initial_prompt_len, output_per_agent, num_agents)
    (1,  100, 100, 3),
    (2,  100, 100, 3),
    (4,  100, 100, 3),
    (1,  500, 200, 3),
    (2,  500, 200, 3),
]


def run_benchmark(model_path):
    env = get_env_report()
    print("=" * 70)
    print("CHUNKED PREFILL BENCHMARK: Nano-vLLM vs vLLM")
    print("=" * 70)
    for k, v in env.items():
        print(f"  {k:20s}: {v}")
    print()

    results = {"env": env, "model": model_path, "batch": [], "pipeline": []}

    # ===== Nano-vLLM =====
    print("Loading Nano-vLLM...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from nanovllm import LLM as NanoLLM
    nano_llm = NanoLLM(model_path)
    # Warmup
    from nanovllm import SamplingParams as NanoSP
    nano_llm.generate([[0] * 10], NanoSP(temperature=0, max_tokens=5, ignore_eos=True), use_tqdm=False)

    print("\n--- Nano-vLLM: Batch Throughput ---")
    for num_prompts, prompt_len, output_len, arrival_gap in CONFIGS:
        prompts = generate_prompts(num_prompts, prompt_len)
        r = bench_nanovllm_batch(nano_llm, prompts, output_len)
        print(f"  N={num_prompts:3d} P={prompt_len:5d} O={output_len:4d} | "
              f"{r['throughput_tok_s']:8.0f} tok/s  {r['total_time_ms']:8.0f}ms")
        results["batch"].append({
            "engine": "nano-vllm", "num_prompts": num_prompts,
            "prompt_len": prompt_len, "output_len": output_len, **r
        })

    print("\n--- Nano-vLLM: Pipeline (A→B→C) ---")
    for num_conv, prompt_len, output_len, num_agents in PIPELINE_CONFIGS:
        prompts = generate_prompts(num_conv, prompt_len)
        r = bench_nanovllm_pipeline(nano_llm, prompts, output_len, num_agents)
        print(f"  C={num_conv} P={prompt_len:5d} O={output_len:4d} A={num_agents} | "
              f"{r['throughput_tok_s']:8.0f} tok/s  {r['total_time_ms']:8.0f}ms")
        results["pipeline"].append({
            "engine": "nano-vllm", "num_conv": num_conv,
            "prompt_len": prompt_len, "output_len": output_len,
            "num_agents": num_agents, **r
        })

    # Cleanup
    nano_llm.exit()
    del nano_llm
    clear_gpu()
    time.sleep(3)

    # ===== vLLM =====
    print("\nLoading vLLM...")
    from vllm import LLM as VLLM, SamplingParams as VllmSP
    vllm_llm = VLLM(
        model=model_path,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=False,
    )
    # Warmup
    vllm_llm.generate([{"prompt_token_ids": [0] * 10}],
                       VllmSP(temperature=0, max_tokens=5, ignore_eos=True), use_tqdm=False)

    print("\n--- vLLM: Batch Throughput ---")
    for num_prompts, prompt_len, output_len, arrival_gap in CONFIGS:
        prompts = generate_prompts(num_prompts, prompt_len)
        r = bench_vllm_batch(vllm_llm, prompts, output_len)
        print(f"  N={num_prompts:3d} P={prompt_len:5d} O={output_len:4d} | "
              f"{r['throughput_tok_s']:8.0f} tok/s  {r['total_time_ms']:8.0f}ms")
        results["batch"].append({
            "engine": "vllm", "num_prompts": num_prompts,
            "prompt_len": prompt_len, "output_len": output_len, **r
        })

    print("\n--- vLLM: Pipeline (A→B→C) ---")
    for num_conv, prompt_len, output_len, num_agents in PIPELINE_CONFIGS:
        prompts = generate_prompts(num_conv, prompt_len)
        r = bench_vllm_pipeline(vllm_llm, prompts, output_len, num_agents)
        print(f"  C={num_conv} P={prompt_len:5d} O={output_len:4d} A={num_agents} | "
              f"{r['throughput_tok_s']:8.0f} tok/s  {r['total_time_ms']:8.0f}ms")
        results["pipeline"].append({
            "engine": "vllm", "num_conv": num_conv,
            "prompt_len": prompt_len, "output_len": output_len,
            "num_agents": num_agents, **r
        })

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY: Batch Throughput Comparison")
    print("=" * 70)
    print(f"{'Config':>30s} | {'Nano-vLLM':>12s} | {'vLLM':>12s} | {'Ratio':>8s}")
    print("-" * 70)
    nano_results = [r for r in results["batch"] if r["engine"] == "nano-vllm"]
    vllm_results = [r for r in results["batch"] if r["engine"] == "vllm"]
    for nr, vr in zip(nano_results, vllm_results):
        cfg = f"N={nr['num_prompts']} P={nr['prompt_len']} O={nr['output_len']}"
        ratio = nr["throughput_tok_s"] / max(vr["throughput_tok_s"], 1)
        marker = "✓" if ratio >= 1.0 else " "
        print(f"{cfg:>30s} | {nr['throughput_tok_s']:>10.0f}  | {vr['throughput_tok_s']:>10.0f}  | {ratio:>6.2f}x {marker}")

    print()
    print("SUMMARY: Pipeline (A→B→C) Comparison")
    print("=" * 70)
    print(f"{'Config':>30s} | {'Nano-vLLM':>12s} | {'vLLM':>12s} | {'Ratio':>8s}")
    print("-" * 70)
    nano_pip = [r for r in results["pipeline"] if r["engine"] == "nano-vllm"]
    vllm_pip = [r for r in results["pipeline"] if r["engine"] == "vllm"]
    for nr, vr in zip(nano_pip, vllm_pip):
        cfg = f"C={nr['num_conv']} P={nr['prompt_len']} O={nr['output_len']} A={nr['num_agents']}"
        ratio = nr["throughput_tok_s"] / max(vr["throughput_tok_s"], 1)
        marker = "✓" if ratio >= 1.0 else " "
        print(f"{cfg:>30s} | {nr['throughput_tok_s']:>10.0f}  | {vr['throughput_tok_s']:>10.0f}  | {ratio:>6.2f}x {marker}")

    # Save results
    outfile = os.path.join(os.path.dirname(__file__), "results_chunked_prefill.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B")
    run_benchmark(model)
