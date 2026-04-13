"""Benchmark: vLLM multi-round inference with prefix caching.

Same workload as bench_multiround.py for fair comparison.
vLLM automatically reuses cached KV blocks via hash-based prefix caching.
Each round sends full context but only new tokens are actually computed.
"""
import os
import time
from random import randint, seed

from vllm import LLM, SamplingParams


def bench_vllm(llm, conversations, rounds_data, output_len):
    """Multi-round without KV cache reuse — re-prefill full context each round."""
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


def main():
    seed(42)
    num_conversations = 16
    num_rounds = 5
    initial_prompt_len = 200
    new_tokens_per_round = 50
    output_len = 64

    path = os.path.expanduser("~/huggingface/Qwen3-8B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096, gpu_memory_utilization=0.6,
              enable_prefix_caching=True)

    # Warmup
    llm.generate([{"prompt_token_ids": [0] * 10}], SamplingParams(max_tokens=8))

    # Generate same test data as nano-vllm benchmark
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

    print(f"Config: {num_conversations} conversations x {num_rounds+1} rounds")
    print(f"  Initial prompt: {initial_prompt_len} tokens")
    print(f"  New tokens/round: {new_tokens_per_round}")
    print(f"  Output tokens/round: {output_len}")
    print()

    seed(42)
    tokens, elapsed = bench_vllm(llm, conversations, rounds_data, output_len)
    print(f"[vLLM - With Prefix Caching]")
    print(f"  Output tokens: {tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {tokens / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
