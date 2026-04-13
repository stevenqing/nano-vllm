#!/usr/bin/env python3
"""
vLLM-only Multi-Round Benchmark Sweep.
Same configs/workloads as the nano-vllm sweep for fair comparison.
"""
import os
import sys
import gc
import json
import time
import subprocess
import platform
from random import randint, seed

import torch


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

    gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu = torch.cuda.get_device_properties(0)
    mem_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits", "-i", gpu_idx],
        capture_output=True, text=True
    ).stdout.strip()

    return {
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
        "CUDA_VISIBLE_DEVICES": gpu_idx,
    }


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


def bench_vllm_prefix_caching(llm, conversations, rounds_data, output_len):
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


def get_gpu_memory_used_mb():
    gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", gpu_idx],
        capture_output=True, text=True
    )
    return int(result.stdout.strip().split("\n")[0])


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Same configs as nano-vllm sweep
SWEEP_CONFIGS = [
    (16, 2,  200, 50, 64),
    (16, 5,  200, 50, 64),
    (16, 10, 200, 50, 64),
    (16, 5,  100, 50, 64),
    (16, 5,  500, 50, 64),
    (16, 5, 1000, 50, 64),
    (16, 5, 200, 100, 64),
    (16, 5, 200, 200, 64),
    (16, 5, 200, 50,  32),
    (16, 5, 200, 50, 128),
    ( 4, 5, 200, 50, 64),
    (32, 5, 200, 50, 64),
]

NUM_REPEATS = 2


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    env_report = get_env_report()
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    for k, v in env_report.items():
        print(f"  {k:20s}: {v}")
    print()

    from vllm import LLM, SamplingParams

    clear_gpu_cache()
    mem_before = get_gpu_memory_used_mb()

    print("=" * 70)
    print("RUNNING: vLLM (prefix caching ON)")
    print("=" * 70)

    llm = LLM(model_path, enforce_eager=False, max_model_len=4096,
              gpu_memory_utilization=0.9, enable_prefix_caching=True,
              disable_log_stats=True)

    # Warmup
    llm.generate([{"prompt_token_ids": [0] * 10}], SamplingParams(max_tokens=8))
    mem_after_init = get_gpu_memory_used_mb()

    results = []
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
                "config": config_key,
                "repeat": repeat,
                "output_tokens": tokens,
                "time_s": round(elapsed, 3),
                "throughput_tok_s": round(tokens / elapsed, 2),
                "gpu_mem_mb": mem_peak,
            })

            print(f"  [{config_key}] r{repeat}: prefix_caching={tokens/elapsed:.1f} tok/s")

    del llm
    clear_gpu_cache()

    print(f"\n  GPU mem: before={mem_before}MB, after_init={mem_after_init}MB\n")

    # Save results
    output = {
        "engine": "vllm",
        "environment": env_report,
        "gpu_memory": {"before_mb": mem_before, "after_init_mb": mem_after_init},
        "results": results,
    }
    with open("results_vllm.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to results_vllm.json")


if __name__ == "__main__":
    main()
