"""ALFWorld Benchmark: vLLM only (run separately to avoid GPU memory conflict)."""
import os
import sys
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")


def setup_alfworld_env(config_path="base_config.yaml", split="eval_out_of_distribution"):
    sys.argv = ["bench", config_path]
    import alfworld.agents.modules.generic as generic
    import alfworld.agents.environment as envlib
    config = generic.load_config()
    env = envlib.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env


def get_task_type(obs):
    if "heat" in obs: return "heat"
    if "cool" in obs: return "cool"
    if "clean" in obs: return "clean"
    if "examine" in obs or "light" in obs.lower(): return "examine"
    if "two" in obs or "both" in obs: return "puttwo"
    return "put"


def extract_action(response):
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("> "):
            return line[2:].strip()
        if line.lower().startswith("action:"):
            return line[7:].strip()
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else "look"


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    max_games = 20
    max_steps = 30
    max_tokens = 100

    with open("alfworld_3prompts.json") as f:
        prompts = json.load(f)

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=False, max_model_len=40960,
              gpu_memory_utilization=0.9, enable_prefix_caching=True,
              disable_log_stats=True)
    llm.generate([{"prompt_token_ids": [0]*10}], SamplingParams(max_tokens=4))
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)

    env = setup_alfworld_env()

    print(f"ALFWorld vLLM Benchmark: {max_games} games, max {max_steps} steps/game")

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

        # Build context as token IDs (same method as nano-vllm)
        context_str = few_shot + "\n\n" + obs + "\n>"
        context_ids = tokenizer.encode(context_str)
        
        results = llm.generate([{"prompt_token_ids": context_ids}], sp, use_tqdm=False)
        output_ids = list(results[0].outputs[0].token_ids)
        response = results[0].outputs[0].text
        total_tokens += len(output_ids)
        
        # Track full token sequence
        all_ids = context_ids + output_ids

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

            # Append suffix tokens (same method as nano-vllm)
            suffix_str = "\n" + obs_new + "\n>"
            suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
            all_ids = all_ids + suffix_ids
            
            results = llm.generate([{"prompt_token_ids": all_ids}], sp, use_tqdm=False)
            output_ids = list(results[0].outputs[0].token_ids)
            response = results[0].outputs[0].text
            total_tokens += len(output_ids)
            all_ids = all_ids + output_ids

        if (game_idx + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  [{game_idx+1}/{max_games}] {total_tokens} tok, {elapsed:.1f}s, {total_tokens/elapsed:.1f} tok/s")

    elapsed = time.time() - t_start
    print(f"\n  Tokens: {total_tokens}, Time: {elapsed:.2f}s, Throughput: {total_tokens/elapsed:.1f} tok/s")
    print(f"  Success: {successes}/{max_games} = {successes/max_games*100:.1f}%")


if __name__ == "__main__":
    main()
