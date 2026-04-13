"""MetaGPT-style Software Company Multi-Agent Benchmark.

Simulates MetaGPT's multi-agent software development pipeline:
- Product Manager → Architect → Engineer → QA
- Each agent builds on the previous agent's output
- Tests both sequential (dependent) and parallel (independent) agent patterns

Compares: nano-vllm (chat API) vs vLLM (prefix caching)
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

AGENTS = [
    {"role": "Product Manager", "system": "You are a Product Manager. Analyze requirements and write user stories. Be concise."},
    {"role": "Architect", "system": "You are a Software Architect. Design system architecture based on requirements. Be concise."},
    {"role": "Engineer", "system": "You are a Senior Engineer. Write code based on the architecture design. Be concise."},
    {"role": "QA Engineer", "system": "You are a QA Engineer. Review code and write test cases. Be concise."},
]

TASKS = [
    "Build a URL shortener service with click analytics",
    "Create a task management API with user authentication",
    "Design a real-time chat system with message persistence",
]


def pipeline_nanovllm(model_path, tasks, max_tokens=150):
    """Run MetaGPT-style pipeline with nano-vllm."""
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=4))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    total_tokens = 0
    t_start = time.time()

    for task in tasks:
        prev_output = task
        for agent in AGENTS:
            prompt = f"{agent['system']}\n\nTask: {task}\nInput from previous stage: {prev_output}\n\nYour output:"
            prompt_ids = tokenizer.encode(prompt)
            result = llm.generate([prompt_ids], sp, use_tqdm=False)
            total_tokens += len(result[0]["token_ids"])
            prev_output = result[0]["text"]

    elapsed = time.time() - t_start
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return total_tokens, elapsed


def pipeline_vllm(model_path, tasks, max_tokens=150):
    """Run MetaGPT-style pipeline with vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096,
              gpu_memory_utilization=0.9, enable_prefix_caching=True,
              disable_log_stats=True)
    llm.generate([{"prompt_token_ids": [0]*10}], SamplingParams(max_tokens=4))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    total_tokens = 0
    t_start = time.time()

    for task in tasks:
        prev_output = task
        for agent in AGENTS:
            prompt = f"{agent['system']}\n\nTask: {task}\nInput from previous stage: {prev_output}\n\nYour output:"
            prompt_ids = tokenizer.encode(prompt)
            results = llm.generate([{"prompt_token_ids": prompt_ids}], sp, use_tqdm=False)
            total_tokens += len(results[0].outputs[0].token_ids)
            prev_output = results[0].outputs[0].text

    elapsed = time.time() - t_start
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return total_tokens, elapsed


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "0.6B"
    model_path = os.path.expanduser(f"~/huggingface/Qwen3-{model}/")
    engine = sys.argv[2] if len(sys.argv) > 2 else "nano"

    print(f"MetaGPT Pipeline Benchmark: Qwen3-{model}, engine={engine}")
    print(f"Tasks: {len(TASKS)}, Agents: {len(AGENTS)}")
    print()

    if engine == "nano":
        tokens, elapsed = pipeline_nanovllm(model_path, TASKS)
        print(f"Nano-vLLM: {tokens} tok / {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s")
    elif engine == "vllm":
        tokens, elapsed = pipeline_vllm(model_path, TASKS)
        print(f"vLLM:      {tokens} tok / {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
