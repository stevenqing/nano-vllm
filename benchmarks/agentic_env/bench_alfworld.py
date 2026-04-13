"""ALFWorld Benchmark: Nano-vLLM (with multi-round KV reuse) vs vLLM.

Uses ReAct-style prompting to solve ALFWorld tasks.
Measures wall-clock time for the full agent-environment interaction loop.
"""
import os
import sys
import json
import time
import yaml
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")

# -------- ALFWorld Environment Setup --------

def setup_alfworld_env(config_path="base_config.yaml", split="eval_out_of_distribution", max_games=20):
    sys.argv = ["bench", config_path]
    import alfworld.agents.modules.generic as generic
    import alfworld.agents.environment as envlib
    config = generic.load_config()
    env = envlib.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env


def get_task_type(obs):
    """Detect task type from observation for prompt selection."""
    if "heat" in obs: return "heat"
    if "cool" in obs: return "cool"
    if "clean" in obs: return "clean"
    if "examine" in obs or "light" in obs.lower(): return "examine"
    if "two" in obs or "both" in obs: return "puttwo"
    return "put"


def load_prompts(path="alfworld_3prompts.json"):
    with open(path) as f:
        return json.load(f)


def extract_action(response):
    """Extract action from LLM response (after '> ' or 'Action:')."""
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("> "):
            return line[2:].strip()
        if line.lower().startswith("action:"):
            return line[7:].strip()
    # fallback: return last non-empty line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else "look"


# -------- Nano-vLLM Backend --------

def run_alfworld_nanovllm(env, prompts, max_games=20, max_steps=30, max_tokens=100):
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, max_model_len=40960, max_num_batched_tokens=40960)
    llm.generate(["warmup"], SamplingParams(max_tokens=4))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    total_tokens = 0
    successes = 0
    t_start = time.time()

    for game_idx in range(max_games):
        obs, info = env.reset()
        obs = obs[0]
        task_type = get_task_type(obs)
        
        prompt_key = f"react_{task_type}_0"
        if prompt_key not in prompts:
            prompt_key = "react_put_0"
        few_shot = prompts[prompt_key]
        
        # Tokenize full initial prompt
        context_str = few_shot + "\n\n" + obs + "\n>"
        context_ids = tokenizer.encode(context_str)
        
        result = llm.chat(context_ids, sp)
        seq_id = result["seq_id"]
        total_tokens += len(result["token_ids"])
        
        # Track token-level context: prompt_ids + output_ids
        all_ids = context_ids + list(result["token_ids"])
        
        done = False
        for step in range(max_steps):
            action = extract_action(result["text"])
            obs_list, rewards, dones, infos = env.step([action])
            obs_new = obs_list[0]
            done = dones[0]
            
            if done:
                if rewards[0] > 0:
                    successes += 1
                break
            
            # Tokenize the new suffix and compute incremental token IDs
            suffix_str = "\n" + obs_new + "\n>"
            suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
            new_ids = suffix_ids
            all_ids = all_ids + suffix_ids
            
            result = llm.chat(new_ids, sp, seq_id=seq_id)
            total_tokens += len(result["token_ids"])
        
        llm.release_session(seq_id)
    
    elapsed = time.time() - t_start
    del llm
    return total_tokens, elapsed, successes, max_games


# -------- vLLM Backend --------

def run_alfworld_vllm(env, prompts, max_games=20, max_steps=30, max_tokens=100):
    from vllm import LLM, SamplingParams
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096,
              gpu_memory_utilization=0.9, enable_prefix_caching=True,
              disable_log_stats=True)
    llm.generate([{"prompt_token_ids": [0]*10}], SamplingParams(max_tokens=4))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    total_tokens = 0
    successes = 0
    t_start = time.time()

    for game_idx in range(max_games):
        obs, info = env.reset()
        obs = obs[0]
        task_type = get_task_type(obs)
        
        prompt_key = f"react_{task_type}_0"
        if prompt_key not in prompts:
            prompt_key = "react_put_0"
        few_shot = prompts[prompt_key]
        
        # vLLM: no session, rebuild full context each step
        context = few_shot + "\n\n" + obs + "\n>"
        results = llm.generate([context], sp, use_tqdm=False)
        response = results[0].outputs[0].text
        total_tokens += len(results[0].outputs[0].token_ids)
        
        done = False
        for step in range(max_steps):
            action = extract_action(response)
            obs_list, rewards, dones, infos = env.step([action])
            obs_new = obs_list[0]
            done = dones[0]
            
            if done:
                if rewards[0] > 0:
                    successes += 1
                break
            
            # Append to full context and re-send
            context += response + "\n" + obs_new + "\n>"
            results = llm.generate([context], sp, use_tqdm=False)
            response = results[0].outputs[0].text
            total_tokens += len(results[0].outputs[0].token_ids)
    
    elapsed = time.time() - t_start
    del llm
    return total_tokens, elapsed, successes, max_games


# -------- Main --------

def main():
    max_games = 20
    max_steps = 30
    max_tokens = 100
    
    prompts = load_prompts("alfworld_3prompts.json")
    
    print(f"ALFWorld Benchmark: {max_games} games, max {max_steps} steps/game")
    print()
    
    # Nano-vLLM
    print("=" * 60)
    print("Running Nano-vLLM (multi-round KV reuse via chat())...")
    print("=" * 60)
    env = setup_alfworld_env(max_games=max_games)
    tokens_nano, time_nano, succ_nano, total_nano = run_alfworld_nanovllm(
        env, prompts, max_games=max_games, max_steps=max_steps, max_tokens=max_tokens
    )
    print(f"  Tokens: {tokens_nano}, Time: {time_nano:.2f}s, Throughput: {tokens_nano/time_nano:.1f} tok/s")
    print(f"  Success: {succ_nano}/{total_nano} = {succ_nano/total_nano*100:.1f}%")
    print()
    
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    
    # vLLM
    print("=" * 60)
    print("Running vLLM (prefix caching, full context each step)...")
    print("=" * 60)
    env2 = setup_alfworld_env(max_games=max_games)
    tokens_vllm, time_vllm, succ_vllm, total_vllm = run_alfworld_vllm(
        env2, prompts, max_games=max_games, max_steps=max_steps, max_tokens=max_tokens
    )
    print(f"  Tokens: {tokens_vllm}, Time: {time_vllm:.2f}s, Throughput: {tokens_vllm/time_vllm:.1f} tok/s")
    print(f"  Success: {succ_vllm}/{total_vllm} = {succ_vllm/total_vllm*100:.1f}%")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Nano-vLLM: {tokens_nano} tok / {time_nano:.2f}s = {tokens_nano/time_nano:.1f} tok/s, success={succ_nano}/{total_nano}")
    print(f"  vLLM:      {tokens_vllm} tok / {time_vllm:.2f}s = {tokens_vllm/time_vllm:.1f} tok/s, success={succ_vllm}/{total_vllm}")
    speedup = time_vllm / time_nano if time_nano > 0 else 0
    print(f"  Nano-vLLM speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
