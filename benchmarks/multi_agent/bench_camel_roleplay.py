"""CAMEL-style Role-Playing Multi-Agent Benchmark.

Simulates CAMEL's role-playing protocol:
- Two agents (e.g., AI User + AI Assistant) take turns
- Each agent sees the full conversation history
- Measures total inference time for the multi-turn conversation

Compares: nano-vllm (chat API with KV reuse) vs vLLM (prefix caching)
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TASKS = [
    {
        "user_role": "Python Programmer",
        "assistant_role": "Chief Technology Officer",
        "task": "Design a microservice architecture for an e-commerce platform that handles 10K concurrent users.",
        "num_turns": 5,
    },
    {
        "user_role": "Data Scientist",
        "assistant_role": "Machine Learning Engineer",
        "task": "Build a recommendation system for a movie streaming platform using collaborative filtering.",
        "num_turns": 5,
    },
    {
        "user_role": "Product Manager",
        "assistant_role": "Senior Software Engineer",
        "task": "Plan and implement a real-time notification system for a social media app.",
        "num_turns": 5,
    },
]

SYSTEM_USER = "You are a {role}. You discuss with your partner to solve the task: {task}. Ask questions, propose ideas, and iterate on solutions. Keep responses concise (2-3 sentences)."
SYSTEM_ASST = "You are a {role}. You discuss with your partner to solve the task: {task}. Provide technical insights, answer questions, and refine solutions. Keep responses concise (2-3 sentences)."


def role_play_nanovllm(model_path, tasks, max_tokens=100):
    """Run role-playing with nano-vllm using multi-round KV reuse."""
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=4))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    total_tokens = 0
    t_start = time.time()

    for task_info in tasks:
        sys_user = SYSTEM_USER.format(role=task_info["user_role"], task=task_info["task"])
        sys_asst = SYSTEM_ASST.format(role=task_info["assistant_role"], task=task_info["task"])

        # Initialize both agents as separate sessions
        user_prompt = tokenizer.encode(sys_user + "\n\nStart the discussion.\nYou:")
        asst_prompt = tokenizer.encode(sys_asst + "\n\n")

        # User agent starts
        result_user = llm.chat(user_prompt, sp)
        user_seq_id = result_user["seq_id"]
        total_tokens += len(result_user["token_ids"])
        user_msg = result_user["text"]

        # Assistant agent gets user's first message
        asst_full = asst_prompt + tokenizer.encode(f"Partner: {user_msg}\nYou:", add_special_tokens=False)
        result_asst = llm.chat(asst_full, sp)
        asst_seq_id = result_asst["seq_id"]
        total_tokens += len(result_asst["token_ids"])
        asst_msg = result_asst["text"]

        # Multi-turn conversation
        for turn in range(1, task_info["num_turns"]):
            # User receives assistant's response
            user_cont = tokenizer.encode(f"\nPartner: {asst_msg}\nYou:", add_special_tokens=False)
            result_user = llm.chat(user_cont, sp, seq_id=user_seq_id)
            total_tokens += len(result_user["token_ids"])
            user_msg = result_user["text"]

            # Assistant receives user's response
            asst_cont = tokenizer.encode(f"\nPartner: {user_msg}\nYou:", add_special_tokens=False)
            result_asst = llm.chat(asst_cont, sp, seq_id=asst_seq_id)
            total_tokens += len(result_asst["token_ids"])
            asst_msg = result_asst["text"]

        llm.release_session(user_seq_id)
        llm.release_session(asst_seq_id)

    elapsed = time.time() - t_start
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return total_tokens, elapsed


def role_play_vllm(model_path, tasks, max_tokens=100):
    """Run role-playing with vLLM (prefix caching, full context each turn)."""
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

    for task_info in tasks:
        sys_user = SYSTEM_USER.format(role=task_info["user_role"], task=task_info["task"])
        sys_asst = SYSTEM_ASST.format(role=task_info["assistant_role"], task=task_info["task"])

        # User agent starts
        user_ids = tokenizer.encode(sys_user + "\n\nStart the discussion.\nYou:")
        results = llm.generate([{"prompt_token_ids": user_ids}], sp, use_tqdm=False)
        user_out = list(results[0].outputs[0].token_ids)
        total_tokens += len(user_out)
        user_msg = results[0].outputs[0].text
        user_all = user_ids + user_out

        # Assistant agent gets user's first message
        asst_ids = tokenizer.encode(sys_asst + "\n\n" + f"Partner: {user_msg}\nYou:")
        results = llm.generate([{"prompt_token_ids": asst_ids}], sp, use_tqdm=False)
        asst_out = list(results[0].outputs[0].token_ids)
        total_tokens += len(asst_out)
        asst_msg = results[0].outputs[0].text
        asst_all = asst_ids + asst_out

        for turn in range(1, task_info["num_turns"]):
            # User receives assistant's response
            user_cont = tokenizer.encode(f"\nPartner: {asst_msg}\nYou:", add_special_tokens=False)
            user_all = user_all + user_cont
            results = llm.generate([{"prompt_token_ids": user_all}], sp, use_tqdm=False)
            user_out = list(results[0].outputs[0].token_ids)
            total_tokens += len(user_out)
            user_msg = results[0].outputs[0].text
            user_all = user_all + user_out

            # Assistant receives user's response
            asst_cont = tokenizer.encode(f"\nPartner: {user_msg}\nYou:", add_special_tokens=False)
            asst_all = asst_all + asst_cont
            results = llm.generate([{"prompt_token_ids": asst_all}], sp, use_tqdm=False)
            asst_out = list(results[0].outputs[0].token_ids)
            total_tokens += len(asst_out)
            asst_msg = results[0].outputs[0].text
            asst_all = asst_all + asst_out

    elapsed = time.time() - t_start
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return total_tokens, elapsed


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "0.6B"
    model_path = os.path.expanduser(f"~/huggingface/Qwen3-{model}/")
    engine = sys.argv[2] if len(sys.argv) > 2 else "nano"

    print(f"CAMEL Role-Playing Benchmark: Qwen3-{model}, engine={engine}")
    print(f"Tasks: {len(TASKS)}, Turns/task: {TASKS[0]['num_turns']}")
    print()

    if engine == "nano":
        tokens, elapsed = role_play_nanovllm(model_path, TASKS)
        print(f"Nano-vLLM: {tokens} tok / {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s")
    elif engine == "vllm":
        tokens, elapsed = role_play_vllm(model_path, TASKS)
        print(f"vLLM:      {tokens} tok / {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s")
    else:
        print(f"Unknown engine: {engine}")


if __name__ == "__main__":
    main()
