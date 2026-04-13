"""Experiment 3: Speculative KV Reuse — Accept Rate Measurement.

Protocol:
1. Agent A prefills [System_A + Context] → KV_A
2. Agent B uses transplanted KV (B's system + A's context) to generate K draft tokens
3. Agent B does ground truth prefill → KV_B (correct)
4. Verify draft tokens via rejection sampling (greedy: compare argmax)
5. Measure: accept rate, speedup potential
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

SYSTEM_PROMPTS = {
    "planner": "You are a planning agent. Your job is to break down complex tasks into step-by-step plans. Think carefully about dependencies between steps.",
    "coder": "You are a coding agent. Your job is to write clean, efficient code based on the given plan. Follow best practices and add error handling.",
    "reviewer": "You are a code review agent. Your job is to review code for bugs, performance issues, and style problems. Be thorough and constructive.",
}

SHARED_CONTEXTS = [
    "We are building a REST API for a todo list application. The API should support creating, reading, updating, and deleting todo items. Each todo item has a title, description, status, and creation date. The API should use JSON format. Please provide your response.",
    "The current codebase has a memory leak in the data processing pipeline. When processing large CSV files, the application gradually consumes all available RAM. We need to identify and fix the memory leak. Please provide your analysis.",
    "Design a real-time chat application with support for multiple chat rooms, user presence indicators, message history with pagination, and file attachments. Please provide your design.",
]

K_VALUES = [1, 2, 4, 8, 16]  # Number of draft tokens to speculate


def build_transplanted_kv(kv_a, kv_b, sys_len_a, sys_len_b):
    """Build transplanted KV: Agent B's system + Agent A's context."""
    transplanted = []
    for layer in range(len(kv_a)):
        k_b_sys = kv_b[layer][0][:, :, :sys_len_b, :]
        v_b_sys = kv_b[layer][1][:, :, :sys_len_b, :]
        k_a_ctx = kv_a[layer][0][:, :, sys_len_a:, :]
        v_a_ctx = kv_a[layer][1][:, :, sys_len_a:, :]
        transplanted.append((
            torch.cat([k_b_sys, k_a_ctx], dim=2),
            torch.cat([v_b_sys, v_a_ctx], dim=2),
        ))
    return tuple(transplanted)


def to_dynamic_cache(kv_tuple):
    cache = DynamicCache()
    for k, v in kv_tuple:
        cache.update(k, v, layer_idx=len(cache))
    return cache


def speculative_kv_reuse(model, tokenizer, system_a, system_b, context, K):
    """Run speculative KV reuse and return accept rate."""
    prompt_a = system_a + "\n\n" + context + "\n\nResponse:"
    prompt_b = system_b + "\n\n" + context + "\n\nResponse:"
    
    ids_a = tokenizer.encode(prompt_a, return_tensors="pt").cuda()
    ids_b = tokenizer.encode(prompt_b, return_tensors="pt").cuda()
    sys_len_a = len(tokenizer.encode(system_a))
    sys_len_b = len(tokenizer.encode(system_b))
    
    with torch.no_grad():
        # Get KV caches
        out_a = model(ids_a, use_cache=True)
        kv_a = out_a.past_key_values
        out_b = model(ids_b, use_cache=True)
        kv_b = out_b.past_key_values
        
        # Build transplanted KV
        tp_kv_tuple = build_transplanted_kv(kv_a, kv_b, sys_len_a, sys_len_b)
        
        # Step 1: Generate K draft tokens using transplanted KV
        tp_cache = to_dynamic_cache(tp_kv_tuple)
        draft_tokens = []
        next_id = ids_b[:, -1:]
        for _ in range(K):
            out = model(next_id, past_key_values=tp_cache, use_cache=True)
            tp_cache = out.past_key_values
            token = out.logits[0, -1].argmax().item()
            draft_tokens.append(token)
            next_id = torch.tensor([[token]], device="cuda")
        
        # Step 2: Verify with ground truth KV
        # Feed all K draft tokens to the model with correct KV in one forward
        gt_cache = to_dynamic_cache(tuple(
            (kv_b[l][0].clone(), kv_b[l][1].clone()) for l in range(len(kv_b))
        ))
        
        # Feed from last prompt token through all draft tokens
        verify_input = torch.tensor(
            [[ids_b[0, -1].item()] + draft_tokens], device="cuda"
        )
        verify_out = model(verify_input, past_key_values=gt_cache, use_cache=False)
        # verify_out.logits: [1, K+1, vocab_size]
        # Position i's logits predict token at position i+1
        # So logits[0, 0] predicts draft_tokens[0], logits[0, 1] predicts draft_tokens[1], etc.
        
        # Step 3: Greedy rejection — accept while argmax matches
        accepted = 0
        for i in range(K):
            gt_token = verify_out.logits[0, i].argmax().item()
            if gt_token == draft_tokens[i]:
                accepted += 1
            else:
                break  # Stop at first mismatch (greedy)
        
        # Also get the bonus token (what ground truth would generate after accepted tokens)
        bonus_token = verify_out.logits[0, accepted].argmax().item()
    
    return accepted, K, bonus_token, draft_tokens[:accepted+1]


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n")
    
    agent_names = list(SYSTEM_PROMPTS.keys())
    
    print(f"{'Pair':<25s} | {'K':>3s} | {'Accepted':>8s} | {'Rate':>6s} | {'Effective':>9s}")
    print("-" * 65)
    
    total_results = {k: [] for k in K_VALUES}
    
    for ctx_idx, context in enumerate(SHARED_CONTEXTS):
        for i in range(len(agent_names)):
            for j in range(len(agent_names)):
                if i == j:
                    continue
                a, b = agent_names[i], agent_names[j]
                
                for K in K_VALUES:
                    accepted, total, bonus, _ = speculative_kv_reuse(
                        model, tokenizer,
                        SYSTEM_PROMPTS[a], SYSTEM_PROMPTS[b],
                        context, K
                    )
                    rate = accepted / K
                    # Effective tokens = accepted + 1 (bonus) per verification step
                    effective = accepted + 1
                    total_results[K].append(accepted)
                    
                    print(f"  ctx{ctx_idx} {a[:4]}→{b[:4]} | {K:>3d} | {accepted:>8d} | {rate:>5.0%} | {effective:>6d}/{K+1}")
        
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY: Average Accept Rate by K")
    print("=" * 65)
    for K in K_VALUES:
        vals = total_results[K]
        avg_accepted = sum(vals) / len(vals)
        avg_rate = avg_accepted / K
        avg_effective = avg_accepted + 1
        # Speedup = effective_tokens / 1 (vs normal decode)
        # But verification costs ~1 forward for K tokens
        # Net speedup = effective_tokens / (1 + draft_cost/verify_cost)
        # With transplanted KV, draft cost ≈ 0 (KV already cached)
        # So speedup ≈ effective_tokens
        print(f"  K={K:>2d}: avg_accepted={avg_accepted:.1f}/{K}, "
              f"rate={avg_rate:.1%}, effective={avg_effective:.1f} tok/verify, "
              f"~speedup={avg_effective:.1f}x")


if __name__ == "__main__":
    main()
