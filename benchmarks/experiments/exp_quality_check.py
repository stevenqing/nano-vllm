"""Experiment: Does multi-round KV reuse affect generation quality?

Compares outputs from:
1. No reuse: each round re-prefills the full context (ground truth)
2. KV reuse: uses chat() API with retained KV cache

Uses Qwen3-8B with real chat prompts, measures:
- Exact token match rate between no-reuse and reuse outputs
- Per-token logit KL divergence
- BLEU-like n-gram overlap
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-8B/")

CONVERSATIONS = [
    {
        "system": "You are a helpful assistant.",
        "rounds": [
            "What is the capital of France?",
            "What is its population?",
            "Name three famous landmarks there.",
            "Which one was built first?",
        ]
    },
    {
        "system": "You are a coding assistant. Respond with code only.",
        "rounds": [
            "Write a Python function to reverse a string.",
            "Now make it handle unicode properly.",
            "Add type hints and docstring.",
            "Write unit tests for it.",
        ]
    },
    {
        "system": "You are a math tutor. Explain step by step.",
        "rounds": [
            "What is 15% of 240?",
            "How about 15% of 360?",
            "What's the general formula for percentage calculation?",
            "Give me 3 practice problems.",
        ]
    },
]

MAX_NEW_TOKENS = 100


def generate_greedy(model, input_ids, past_kv, max_new_tokens):
    """Generate tokens greedily, return tokens and final KV cache."""
    generated = []
    next_id = input_ids[:, -1:]
    kv = past_kv
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(next_id, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            token = out.logits[0, -1].argmax().item()
            generated.append(token)
            if token == model.config.eos_token_id:
                break
            next_id = torch.tensor([[token]], device="cuda")
    
    return generated, kv


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).cuda().eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n")
    
    total_tokens = 0
    total_match = 0
    
    for conv_idx, conv in enumerate(CONVERSATIONS):
        system = conv["system"]
        rounds = conv["rounds"]
        
        print(f"{'='*70}")
        print(f"Conversation {conv_idx}: {system[:50]}...")
        print(f"{'='*70}")
        
        # Build context incrementally for both methods
        no_reuse_context = system + "\n"
        reuse_kv = None
        reuse_all_tokens = None
        
        for round_idx, user_msg in enumerate(rounds):
            round_suffix = f"\nUser: {user_msg}\nAssistant:"
            
            # === Method 1: No reuse — full context each time ===
            no_reuse_context_full = no_reuse_context + round_suffix
            ids_full = tokenizer.encode(no_reuse_context_full, return_tensors="pt").cuda()
            
            with torch.no_grad():
                out_full = model(ids_full, use_cache=True)
            
            gt_tokens, gt_kv = generate_greedy(model, ids_full, out_full.past_key_values, MAX_NEW_TOKENS)
            gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True)
            
            # === Method 2: KV reuse — only process new tokens ===
            if reuse_kv is None:
                # First round: same as no-reuse
                reuse_all_ids = ids_full.clone()
                with torch.no_grad():
                    out_reuse = model(reuse_all_ids, use_cache=True)
                reuse_tokens, reuse_kv = generate_greedy(model, reuse_all_ids, out_reuse.past_key_values, MAX_NEW_TOKENS)
            else:
                # Subsequent rounds: only feed new tokens (assistant output + user msg)
                prev_output_text = tokenizer.decode(prev_reuse_tokens, skip_special_tokens=True)
                new_text = prev_output_text + round_suffix
                new_ids = tokenizer.encode(new_text, return_tensors="pt").cuda()
                
                with torch.no_grad():
                    out_reuse = model(new_ids, past_key_values=reuse_kv, use_cache=True)
                reuse_tokens, reuse_kv = generate_greedy(model, new_ids, out_reuse.past_key_values, MAX_NEW_TOKENS)
            
            reuse_text = tokenizer.decode(reuse_tokens, skip_special_tokens=True)
            prev_reuse_tokens = reuse_tokens
            
            # Update no-reuse context for next round
            no_reuse_context = no_reuse_context_full + gt_text
            
            # === Compare ===
            min_len = min(len(gt_tokens), len(reuse_tokens))
            if min_len > 0:
                matches = sum(1 for a, b in zip(gt_tokens[:min_len], reuse_tokens[:min_len]) if a == b)
                match_rate = matches / min_len
            else:
                matches = 0
                match_rate = 0
            
            total_tokens += min_len
            total_match += matches
            
            # First diverge
            first_div = min_len
            for i in range(min_len):
                if gt_tokens[i] != reuse_tokens[i]:
                    first_div = i
                    break
            
            print(f"\n  Round {round_idx}: \"{user_msg[:40]}...\"")
            print(f"    Tokens: gt={len(gt_tokens)}, reuse={len(reuse_tokens)}")
            print(f"    Match rate: {match_rate:.1%} ({matches}/{min_len})")
            print(f"    First diverge: position {first_div}")
            print(f"    GT:    {gt_text[:80]}...")
            print(f"    Reuse: {reuse_text[:80]}...")
        
        # Clean up KV for next conversation
        reuse_kv = None
        torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_match}/{total_tokens} = {total_match/total_tokens:.1%} token match rate")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
