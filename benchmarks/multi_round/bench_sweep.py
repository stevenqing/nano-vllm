#!/usr/bin/env python3
"""
Comprehensive Multi-Round Benchmark Sweep: Nano-vLLM vs vLLM

Ensures fair comparison by:
- Using the EXACT same GPU (CUDA_VISIBLE_DEVICES=0)
- Using the EXACT same random seeds for identical workloads
- Clearing GPU cache between runs
- Measuring GPU memory usage during inference
- Sweeping multiple configurations
- Running each config multiple times for statistical robustness
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

# Force single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch


# ===================== Environment Report =====================

def get_env_report():
    import flash_attn, triton, transformers, xxhash
    try:
        import vllm
        vllm_ver = vllm.__version__
    except:
        vllm_ver = "N/A"
    try:
        import flashinfer
        flashinfer_ver = flashinfer.__version__
    except:
        flashinfer_ver = "N/A"

    gpu = torch.cuda.get_device_properties(0)
    mem_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
        capture_output=True, text=True
    ).stdout.strip()

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "flash_attn": flash_attn.__version__,
        "triton": triton.__version__,
        "transformers": transformers.__version__,
        "xxhash": xxhash.VERSION,
        "vllm": vllm_ver,
        "flashinfer": flashinfer_ver,
        "gpu_name": gpu.name,
        "gpu_memory_gb": f"{gpu.total_memory / 1024**3:.1f}",
        "gpu_sm": f"{gpu.major}.{gpu.minor}",
        "gpu_memory_mib": mem_info,
    }
    return report


def print_env(report):
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    for k, v in report.items():
        print(f"  {k:20s}: {v}")
    print()


# ===================== Workload Generation =====================

def generate_workload(num_conversations, num_rounds, initial_prompt_len,
                      new_tokens_per_round, random_seed=42):
    seed(random_seed)
    conversations = [
        [randint(0, 10000) for _ in range(initial_prompt_len)]
        for _ in range(num_conversations)
    ]
    rounds_data = [
        [
            [randint(0, 10000) for _ in range(new_tokens_per_round)]
            for _ in range(num_rounds)
        ]
        for _ in range(num_conversations)
    ]
    return conversations, rounds_data


# ===================== Nano-vLLM Benchmarks =====================

def bench_nanovllm_no_reuse(llm, conversations, rounds_data, output_len):
    """Nano-vLLM: each round re-prefills full context."""
    from nanovllm import SamplingParams
    total_output_tokens = 0
    t = time.time()

    for conv_id in range(len(conversations)):
        context = list(conversations[conv_id])
        sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        outputs = llm.generate([context], sp, use_tqdm=False)
        result = outputs[0]
        total_output_tokens += len(result["token_ids"])

        for r in range(len(rounds_data[conv_id])):
            new_tokens = rounds_data[conv_id][r]
            context = context + list(result["token_ids"]) + new_tokens
            sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
            outputs = llm.generate([context], sp, use_tqdm=False)
            result = outputs[0]
            total_output_tokens += len(result["token_ids"])

    elapsed = time.time() - t
    return total_output_tokens, elapsed


def bench_nanovllm_with_reuse(llm, conversations, rounds_data, output_len):
    """Nano-vLLM: multi-round KV cache reuse via chat() API."""
    from nanovllm import SamplingParams
    total_output_tokens = 0
    t = time.time()

    for conv_id in range(len(conversations)):
        prompt_ids = conversations[conv_id]
        sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        result = llm.chat(prompt_ids, sp)
        seq_id = result["seq_id"]
        total_output_tokens += len(result["token_ids"])

        for r in range(len(rounds_data[conv_id])):
            new_tokens = rounds_data[conv_id][r]
            sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
            result = llm.chat(new_tokens, sp, seq_id=seq_id)
            total_output_tokens += len(result["token_ids"])

        llm.release_session(seq_id)

    elapsed = time.time() - t
    return total_output_tokens, elapsed


# ===================== vLLM Benchmark =====================

def bench_vllm_prefix_caching(llm, conversations, rounds_data, output_len):
    """vLLM: each round sends full context, prefix caching auto-reuses KV."""
    from vllm import SamplingParams
    total_output_tokens = 0
    t = time.time()

    for conv_id in range(len(conversations)):
        context = list(conversations[conv_id])
        sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        outputs = llm.generate([{"prompt_token_ids": context}], sp, use_tqdm=False)
        result_tokens = list(outputs[0].outputs[0].token_ids)
        total_output_tokens += len(result_tokens)

        for r in range(len(rounds_data[conv_id])):
            new_tokens = rounds_data[conv_id][r]
            context = context + result_tokens + new_tokens
            sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
            outputs = llm.generate([{"prompt_token_ids": context}], sp, use_tqdm=False)
            result_tokens = list(outputs[0].outputs[0].token_ids)
            total_output_tokens += len(result_tokens)

    elapsed = time.time() - t
    return total_output_tokens, elapsed


# ===================== GPU Memory Measurement =====================

def get_gpu_memory_used_mb():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
        capture_output=True, text=True
    )
    return int(result.stdout.strip().split("\n")[0])


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ===================== Sweep Configs =====================

SWEEP_CONFIGS = [
    # (num_conversations, num_rounds, initial_prompt_len, new_tokens_per_round, output_len)
    # Vary rounds
    (16, 2,  200, 50, 64),
    (16, 5,  200, 50, 64),
    (16, 10, 200, 50, 64),
    # Vary initial prompt length
    (16, 5,  100, 50, 64),
    (16, 5,  500, 50, 64),
    (16, 5, 1000, 50, 64),
    # Vary new tokens per round
    (16, 5, 200, 100, 64),
    (16, 5, 200, 200, 64),
    # Vary output length
    (16, 5, 200, 50,  32),
    (16, 5, 200, 50, 128),
    # Vary num conversations
    ( 4, 5, 200, 50, 64),
    (32, 5, 200, 50, 64),
]

NUM_REPEATS = 2  # Run each config twice for stability


# ===================== Main =====================

def run_single_engine(engine_name, engine_type, model_path):
    """Run all sweep configs for a single engine. Returns results list."""
    results = []

    if engine_type == "nanovllm":
        from nanovllm import LLM, SamplingParams
        clear_gpu_cache()
        mem_before = get_gpu_memory_used_mb()
        llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
        # Warmup
        llm.generate(["Warmup"], SamplingParams(max_tokens=8))
        mem_after_init = get_gpu_memory_used_mb()

        for cfg in SWEEP_CONFIGS:
            n_conv, n_rounds, prompt_len, new_tok, out_len = cfg
            config_key = f"C{n_conv}_R{n_rounds}_P{prompt_len}_N{new_tok}_O{out_len}"

            for repeat in range(NUM_REPEATS):
                conversations, rounds_data = generate_workload(
                    n_conv, n_rounds, prompt_len, new_tok, random_seed=42 + repeat
                )

                # No reuse
                clear_gpu_cache()
                tokens_nr, time_nr = bench_nanovllm_no_reuse(
                    llm, conversations, rounds_data, out_len
                )
                mem_nr = get_gpu_memory_used_mb()

                # With reuse
                clear_gpu_cache()
                tokens_wr, time_wr = bench_nanovllm_with_reuse(
                    llm, conversations, rounds_data, out_len
                )
                mem_wr = get_gpu_memory_used_mb()

                results.append({
                    "engine": "nano-vllm",
                    "mode": "no_reuse",
                    "config": config_key,
                    "repeat": repeat,
                    "output_tokens": tokens_nr,
                    "time_s": round(time_nr, 3),
                    "throughput_tok_s": round(tokens_nr / time_nr, 2),
                    "gpu_mem_mb": mem_nr,
                })
                results.append({
                    "engine": "nano-vllm",
                    "mode": "kv_reuse",
                    "config": config_key,
                    "repeat": repeat,
                    "output_tokens": tokens_wr,
                    "time_s": round(time_wr, 3),
                    "throughput_tok_s": round(tokens_wr / time_wr, 2),
                    "gpu_mem_mb": mem_wr,
                })

                print(f"  [{config_key}] r{repeat}: no_reuse={tokens_nr/time_nr:.1f} tok/s, "
                      f"kv_reuse={tokens_wr/time_wr:.1f} tok/s, "
                      f"speedup={time_nr/time_wr:.2f}x", flush=True)

        # Cleanup nano-vllm
        del llm
        clear_gpu_cache()

    elif engine_type == "vllm":
        from vllm import LLM, SamplingParams
        clear_gpu_cache()
        mem_before = get_gpu_memory_used_mb()
        llm = LLM(model_path, enforce_eager=False, max_model_len=4096,
                   gpu_memory_utilization=0.9, enable_prefix_caching=True)
        # Warmup
        llm.generate([{"prompt_token_ids": [0] * 10}], SamplingParams(max_tokens=8))
        mem_after_init = get_gpu_memory_used_mb()

        for cfg in SWEEP_CONFIGS:
            n_conv, n_rounds, prompt_len, new_tok, out_len = cfg
            config_key = f"C{n_conv}_R{n_rounds}_P{prompt_len}_N{new_tok}_O{out_len}"

            for repeat in range(NUM_REPEATS):
                conversations, rounds_data = generate_workload(
                    n_conv, n_rounds, prompt_len, new_tok, random_seed=42 + repeat
                )

                tokens, elapsed = bench_vllm_prefix_caching(
                    llm, conversations, rounds_data, out_len
                )
                mem_peak = get_gpu_memory_used_mb()

                results.append({
                    "engine": "vllm",
                    "mode": "prefix_caching",
                    "config": config_key,
                    "repeat": repeat,
                    "output_tokens": tokens,
                    "time_s": round(elapsed, 3),
                    "throughput_tok_s": round(tokens / elapsed, 2),
                    "gpu_mem_mb": mem_peak,
                })

                print(f"  [{config_key}] r{repeat}: prefix_caching={tokens/elapsed:.1f} tok/s", flush=True)

        del llm
        clear_gpu_cache()

    return results, mem_before, mem_after_init


def print_summary(all_results):
    """Print a formatted comparison table."""
    from collections import defaultdict

    # Group by config
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    print("\n" + "=" * 110)
    print(f"{'Config':<32s} | {'nano-vllm no_reuse':>18s} | {'nano-vllm kv_reuse':>18s} | "
          f"{'vLLM prefix_cache':>18s} | {'Speedup vs vLLM':>15s}")
    print("-" * 110)

    for config_key in dict.fromkeys(r["config"] for r in all_results):
        entries = by_config[config_key]

        # Average throughput per mode
        modes = {}
        for e in entries:
            key = (e["engine"], e["mode"])
            if key not in modes:
                modes[key] = []
            modes[key].append(e["throughput_tok_s"])

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        nr = avg(modes.get(("nano-vllm", "no_reuse"), []))
        wr = avg(modes.get(("nano-vllm", "kv_reuse"), []))
        vc = avg(modes.get(("vllm", "prefix_caching"), []))

        speedup = f"{wr/vc:.2f}x" if vc > 0 else "N/A"
        print(f"  {config_key:<30s} | {nr:>15.1f}t/s | {wr:>15.1f}t/s | "
              f"{vc:>15.1f}t/s | {speedup:>15s}")

    print("=" * 110)


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    env_report = get_env_report()
    print_env(env_report)

    all_results = []

    # --- Nano-vLLM ---
    print("=" * 70)
    print("RUNNING: Nano-vLLM")
    print("=" * 70)
    nano_results, nano_mem_before, nano_mem_init = run_single_engine(
        "nano-vllm", "nanovllm", model_path
    )
    all_results.extend(nano_results)
    print(f"\n  GPU mem: before={nano_mem_before}MB, after_init={nano_mem_init}MB\n", flush=True)

    # Give GPU time to release
    time.sleep(3)
    clear_gpu_cache()

    # --- vLLM ---
    print("=" * 70)
    print("RUNNING: vLLM (prefix caching ON)")
    print("=" * 70)
    vllm_results, vllm_mem_before, vllm_mem_init = run_single_engine(
        "vllm", "vllm", model_path
    )
    all_results.extend(vllm_results)
    print(f"\n  GPU mem: before={vllm_mem_before}MB, after_init={vllm_mem_init}MB\n", flush=True)

    # --- Summary ---
    print_summary(all_results)

    # Save raw results
    output = {
        "environment": env_report,
        "gpu_memory": {
            "nano_vllm": {"before": nano_mem_before, "after_init": nano_mem_init},
            "vllm": {"before": vllm_mem_before, "after_init": vllm_mem_init},
        },
        "results": all_results,
    }
    with open("bench_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRaw results saved to bench_sweep_results.json")


if __name__ == "__main__":
    main()
