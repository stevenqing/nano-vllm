"""Experiment 2: KV Transplant Quality Verification.

Uses Agent A's KV cache for Agent B's context region, generates output,
and compares with ground truth (Agent B's own KV cache).

Measures: token-level agreement, logits KL divergence, and output text similarity.
"""
import os
import torch
import numpy as np
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
]

MAX_NEW_TOKENS = 50


def generate_with_kv(model, tokenizer, system_a, system_b, context, max_new_tokens=50):
    """Compare generation with:
    1. Ground truth: Agent B with its own KV
    2. Transplant: Agent B with Agent A's KV for the context region
    """
    # Full prompts
    prompt_a = system_a + "\n\n" + context + "\n\nResponse:"
    prompt_b = system_b + "\n\n" + context + "\n\nResponse:"
    
    ids_a = tokenizer.encode(prompt_a, return_tensors="pt").cuda()
    ids_b = tokenizer.encode(prompt_b, return_tensors="pt").cuda()
    
    # 1. Ground truth: Agent B normal generation
    with torch.no_grad():
        gt_output = model.generate(ids_b, max_new_tokens=max_new_tokens, 
                                    do_sample=False, temperature=1.0)
    gt_tokens = gt_output[0, ids_b.shape[1]:].tolist()
    gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True)
    
    # 2. Get Agent A's past_key_values
    with torch.no_grad():
        out_a = model(ids_a, use_cache=True)
        kv_a = out_a.past_key_values
    
    # 3. Get Agent B's past_key_values (ground truth KV)
    with torch.no_grad():
        out_b = model(ids_b, use_cache=True)
        kv_b = out_b.past_key_values
    
    # 4. Build transplanted KV: Agent B's system region + Agent A's context region
    sys_len_a = len(tokenizer.encode(system_a))
    sys_len_b = len(tokenizer.encode(system_b))
    ctx_len = len(tokenizer.encode("\n\n" + context + "\n\nResponse:"))
    
    transplanted_kv = []
    for layer in range(len(kv_a)):
        k_b_sys = kv_b[layer][0][:, :, :sys_len_b, :]   # Agent B's system KV
        v_b_sys = kv_b[layer][1][:, :, :sys_len_b, :]
        k_a_ctx = kv_a[layer][0][:, :, sys_len_a:, :]    # Agent A's context KV
        v_a_ctx = kv_a[layer][1][:, :, sys_len_a:, :]
        
        # Transplant: [B's system KV] + [A's context KV]
        k_transplant = torch.cat([k_b_sys, k_a_ctx], dim=2)
        v_transplant = torch.cat([v_b_sys, v_a_ctx], dim=2)
        transplanted_kv.append((k_transplant, v_transplant))
    
    transplanted_kv = tuple(transplanted_kv)
    
    # Convert to DynamicCache for HF compatibility
    tp_cache = DynamicCache()
    for k, v in transplanted_kv:
        tp_cache.update(k, v, layer_idx=len(tp_cache))
    
    # 5. Generate with transplanted KV
    # Need to create proper input for generation continuation
    # The transplanted KV has length = sys_len_b + (total_a - sys_len_a)
    # We need to feed exactly the right number of position tokens
    transplant_len = transplanted_kv[0][0].shape[2]
    
    # Generate token by token with transplanted KV
    transplant_tokens = []
    current_kv = transplanted_kv
    # Start from the last token logits
    with torch.no_grad():
        # Get logits from last position of transplanted KV
        # We need to feed one token to continue generation
        last_token_id = ids_b[0, -1:].unsqueeze(0) if transplant_len < ids_b.shape[1] else ids_b[:, -1:]
        
        # Actually, regenerate from transplanted KV by feeding remaining tokens
        # The transplanted KV covers positions 0..transplant_len-1
        # If transplant_len == len(ids_b), we just need to generate
        if transplant_len >= ids_b.shape[1]:
            # KV already covers full prompt, generate from last logits
            for _ in range(max_new_tokens):
                logits = out_a.logits  # placeholder
                break  # simplified
        
        # Simpler approach: compare logits at the last prompt position
        # using ground truth KV vs transplanted KV
        
        # Ground truth logits (Agent B, own KV)
        gt_logits = out_b.logits[0, -1, :]  # [vocab_size]
        
        # Transplanted logits: feed last token with transplanted KV
        # We need the position to match
        dummy_input = ids_b[:, -1:]  # last token
        transplant_out = model(dummy_input, past_key_values=tp_cache, use_cache=False)
        tp_logits = transplant_out.logits[0, -1, :]  # [vocab_size]
    
    # Compare logits
    gt_probs = torch.softmax(gt_logits.float(), dim=-1)
    tp_probs = torch.softmax(tp_logits.float(), dim=-1)
    
    # KL divergence
    kl_div = torch.nn.functional.kl_div(
        tp_probs.log(), gt_probs, reduction='sum'
    ).item()
    
    # Top-1 agreement
    gt_top1 = gt_logits.argmax().item()
    tp_top1 = tp_logits.argmax().item()
    top1_match = (gt_top1 == tp_top1)
    
    # Top-5 agreement
    gt_top5 = set(gt_logits.topk(5).indices.tolist())
    tp_top5 = set(tp_logits.topk(5).indices.tolist())
    top5_overlap = len(gt_top5 & tp_top5) / 5
    
    # Cosine similarity of logits
    logit_cos = torch.nn.functional.cosine_similarity(
        gt_logits.float().unsqueeze(0), tp_logits.float().unsqueeze(0)
    ).item()
    
    # Now do multi-token generation comparison
    # Generate with ground truth KV
    gt_gen_tokens = []
    gt_kv = kv_b
    next_id = ids_b[:, -1:]
    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(next_id, past_key_values=gt_kv, use_cache=True)
            gt_kv = out.past_key_values
            next_token = out.logits[0, -1].argmax().item()
            gt_gen_tokens.append(next_token)
            next_id = torch.tensor([[next_token]], device="cuda")
            if next_token == tokenizer.eos_token_id:
                break
    
    # Generate with transplanted KV
    tp_gen_tokens = []
    # Rebuild fresh DynamicCache for generation
    tp_kv = DynamicCache()
    for k, v in transplanted_kv:
        tp_kv.update(k, v, layer_idx=len(tp_kv))
    next_id = ids_b[:, -1:]
    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(next_id, past_key_values=tp_kv, use_cache=True)
            tp_kv = out.past_key_values
            next_token = out.logits[0, -1].argmax().item()
            tp_gen_tokens.append(next_token)
            next_id = torch.tensor([[next_token]], device="cuda")
            if next_token == tokenizer.eos_token_id:
                break
    
    # Token match rate
    min_len = min(len(gt_gen_tokens), len(tp_gen_tokens))
    token_matches = sum(1 for a, b in zip(gt_gen_tokens[:min_len], tp_gen_tokens[:min_len]) if a == b)
    token_match_rate = token_matches / min_len if min_len > 0 else 0
    
    # First diverge position
    first_diverge = min_len
    for i in range(min_len):
        if gt_gen_tokens[i] != tp_gen_tokens[i]:
            first_diverge = i
            break
    
    gt_gen_text = tokenizer.decode(gt_gen_tokens, skip_special_tokens=True)
    tp_gen_text = tokenizer.decode(tp_gen_tokens, skip_special_tokens=True)
    
    return {
        "kl_div": kl_div,
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "logit_cos": logit_cos,
        "token_match_rate": token_match_rate,
        "first_diverge": first_diverge,
        "gt_text": gt_gen_text[:100],
        "tp_text": tp_gen_text[:100],
    }


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params\n")
    
    agent_names = list(SYSTEM_PROMPTS.keys())
    
    for ctx_idx, context in enumerate(SHARED_CONTEXTS):
        print(f"{'='*70}")
        print(f"Context {ctx_idx}: {context[:60]}...")
        print(f"{'='*70}")
        
        for i in range(len(agent_names)):
            for j in range(len(agent_names)):
                if i == j:
                    continue
                a, b = agent_names[i], agent_names[j]
                
                result = generate_with_kv(
                    model, tokenizer,
                    SYSTEM_PROMPTS[a], SYSTEM_PROMPTS[b],
                    context, MAX_NEW_TOKENS
                )
                
                print(f"\n  Transplant: {a}'s KV → {b}'s generation")
                print(f"    Logit cosine sim:  {result['logit_cos']:.4f}")
                print(f"    KL divergence:     {result['kl_div']:.4f}")
                print(f"    Top-1 match:       {result['top1_match']}")
                print(f"    Top-5 overlap:     {result['top5_overlap']:.1%}")
                print(f"    Token match rate:  {result['token_match_rate']:.1%} ({MAX_NEW_TOKENS} tokens)")
                print(f"    First diverge at:  position {result['first_diverge']}")
                print(f"    GT:  {result['gt_text']}")
                print(f"    TP:  {result['tp_text']}")
        
        print()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
