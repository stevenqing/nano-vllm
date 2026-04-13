"""Benchmark: Multi-round inference with vs without KV cache reuse.

Simulates multi-agent multi-round conversations:
- N conversations, each with R rounds
- Each round appends new_tokens to the growing context
- Compares:
  1. nano-vllm with multi-round KV reuse (chat API)
  2. nano-vllm without reuse (generate API, re-prefill each round)
"""
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def bench_with_reuse(llm, conversations, rounds_data, output_len):
    """Multi-round with KV cache reuse via chat()."""
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
            # Only send new user tokens — assistant output is already in the session
            sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
            result = llm.chat(new_tokens, sp, seq_id=seq_id)
            total_output_tokens += len(result["token_ids"])

        llm.release_session(seq_id)

    elapsed = time.time() - t
    return total_output_tokens, elapsed


def bench_without_reuse(llm, conversations, rounds_data, output_len):
    """Multi-round without KV cache reuse — re-prefill full context each round."""
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


def main():
    seed(42)
    num_conversations = 16
    num_rounds = 5
    initial_prompt_len = 200
    new_tokens_per_round = 50
    output_len = 64

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # Warmup
    llm.generate(["Warmup"], SamplingParams(max_tokens=8))

    # Generate test data
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

    # Bench without reuse (baseline)
    seed(42)
    tokens_no_reuse, time_no_reuse = bench_without_reuse(
        llm, conversations, rounds_data, output_len
    )
    print(f"[Without KV Reuse]")
    print(f"  Output tokens: {tokens_no_reuse}")
    print(f"  Time: {time_no_reuse:.2f}s")
    print(f"  Throughput: {tokens_no_reuse / time_no_reuse:.2f} tok/s")
    print()

    # Bench with reuse
    seed(42)
    tokens_reuse, time_reuse = bench_with_reuse(
        llm, conversations, rounds_data, output_len
    )
    print(f"[With KV Reuse (chat API)]")
    print(f"  Output tokens: {tokens_reuse}")
    print(f"  Time: {time_reuse:.2f}s")
    print(f"  Throughput: {tokens_reuse / time_reuse:.2f} tok/s")
    print()

    speedup = time_no_reuse / time_reuse
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
