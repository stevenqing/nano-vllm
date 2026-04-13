"""KV Similarity Profiling: Measure KV cache similarity across different system prompts.

Goal: For the same shared context, how similar are the KV caches when preceded by
different system prompts? This determines whether KV Transplant / Speculative KV Reuse
is viable.

Output: Per-layer, per-position cosine similarity heatmap.
"""
import os
import torch
import numpy as np
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-8B/")

# Three typical agent system prompts
SYSTEM_PROMPTS = {
    "planner": "You are a planning agent. Your job is to break down complex tasks into step-by-step plans. Think carefully about dependencies between steps.",
    "coder": "You are a coding agent. Your job is to write clean, efficient code based on the given plan. Follow best practices and add error handling.",
    "reviewer": "You are a code review agent. Your job is to review code for bugs, performance issues, and style problems. Be thorough and constructive.",
}

SHARED_CONTEXTS = [
    "We are building a REST API for a todo list application. The API should support creating, reading, updating, and deleting todo items. Each todo item has a title, description, status, and creation date. The API should use JSON format for requests and responses. Authentication is required using JWT tokens. The backend should use Python with FastAPI framework and SQLite database. Please implement this step by step.",
    "The current codebase has a memory leak in the data processing pipeline. When processing large CSV files (>1GB), the application gradually consumes all available RAM and eventually crashes with an OOM error. The pipeline reads data in chunks, applies transformations, and writes results to a PostgreSQL database. We need to identify and fix the memory leak while maintaining processing throughput.",
    "Design a real-time chat application with the following requirements: support for multiple chat rooms, user presence indicators, message history with pagination, file attachments up to 10MB, and end-to-end encryption for private messages. The system should handle at least 10000 concurrent users with less than 100ms message delivery latency.",
]


def get_kv_caches(model, tokenizer, full_prompt):
    """Run model forward and extract KV caches from all layers."""
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").cuda()
    
    # Hook to capture KV caches
    kv_caches = {}
    hooks = []
    
    layer_id = 0
    for module in model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            # This is an Attention module in nano-vllm's Qwen3
            # We need to capture the key/value BEFORE they go into the cache
            pass
    
    # Actually, let's use HuggingFace model directly for cleaner KV extraction
    from transformers import AutoModelForCausalLM
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, use_cache=True)
    
    # outputs.past_key_values: tuple of (key, value) per layer
    # key shape: [batch, num_kv_heads, seq_len, head_dim]
    return outputs.past_key_values, input_ids.shape[1]


def cosine_sim_per_position(kv_a, kv_b, context_start_a, context_start_b, context_len):
    """Compute per-layer, per-position cosine similarity of KV caches 
    for the shared context region."""
    num_layers = len(kv_a)
    sims_k = np.zeros((num_layers, context_len))
    sims_v = np.zeros((num_layers, context_len))
    
    for layer in range(num_layers):
        k_a = kv_a[layer][0][0, :, context_start_a:context_start_a+context_len, :]  # [heads, ctx_len, dim]
        k_b = kv_b[layer][0][0, :, context_start_b:context_start_b+context_len, :]
        v_a = kv_a[layer][1][0, :, context_start_a:context_start_a+context_len, :]
        v_b = kv_b[layer][1][0, :, context_start_b:context_start_b+context_len, :]
        
        # Flatten heads for cosine sim
        for pos in range(context_len):
            ka = k_a[:, pos, :].flatten().float()
            kb = k_b[:, pos, :].flatten().float()
            va = v_a[:, pos, :].flatten().float()
            vb = v_b[:, pos, :].flatten().float()
            
            sims_k[layer, pos] = torch.nn.functional.cosine_similarity(ka.unsqueeze(0), kb.unsqueeze(0)).item()
            sims_v[layer, pos] = torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
    
    return sims_k, sims_v


def main():
    from transformers import AutoModelForCausalLM
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    print(f"Layers: {model.config.num_hidden_layers}")
    print()
    
    agent_names = list(SYSTEM_PROMPTS.keys())
    
    for ctx_idx, context in enumerate(SHARED_CONTEXTS):
        print(f"=" * 70)
        print(f"Context {ctx_idx}: {context[:80]}...")
        print(f"=" * 70)
        
        # Tokenize each agent's full prompt
        kv_data = {}
        ctx_starts = {}
        ctx_lens = {}
        
        for name in agent_names:
            system = SYSTEM_PROMPTS[name]
            full = system + "\n\n" + context
            sys_tokens = tokenizer.encode(system)
            ctx_tokens = tokenizer.encode("\n\n" + context)
            full_tokens = tokenizer.encode(full)
            
            ctx_start = len(sys_tokens)  # where context starts in the sequence
            ctx_len = len(ctx_tokens)
            
            with torch.no_grad():
                input_ids = torch.tensor([full_tokens], dtype=torch.long).cuda()
                outputs = model(input_ids, use_cache=True)
                kv_data[name] = outputs.past_key_values
                ctx_starts[name] = ctx_start
                ctx_lens[name] = ctx_len
            
            print(f"  {name}: sys={len(sys_tokens)} tok, ctx_start={ctx_start}, ctx_len={ctx_len}, total={len(full_tokens)}")
        
        # Pairwise comparison
        print()
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                a, b = agent_names[i], agent_names[j]
                min_ctx = min(ctx_lens[a], ctx_lens[b])
                # Ensure we don't exceed actual sequence length
                max_pos_a = kv_data[a][0][0].shape[2] - ctx_starts[a]
                max_pos_b = kv_data[b][0][0].shape[2] - ctx_starts[b]
                min_ctx = min(min_ctx, max_pos_a, max_pos_b)
                
                sims_k, sims_v = cosine_sim_per_position(
                    kv_data[a], kv_data[b],
                    ctx_starts[a], ctx_starts[b], min_ctx
                )
                
                # Summary stats
                print(f"  {a} vs {b} (context region, {min_ctx} positions):")
                print(f"    Key cosine sim:   mean={sims_k.mean():.4f}, min={sims_k.min():.4f}, "
                      f"last10_mean={sims_k[:,-10:].mean():.4f}")
                print(f"    Value cosine sim: mean={sims_v.mean():.4f}, min={sims_v.min():.4f}, "
                      f"last10_mean={sims_v[:,-10:].mean():.4f}")
                
                # Per-layer summary (first, middle, last layers)
                L = sims_k.shape[0]
                for layer_idx in [0, L//4, L//2, 3*L//4, L-1]:
                    print(f"    Layer {layer_idx:2d}: K_sim={sims_k[layer_idx].mean():.4f}, "
                          f"V_sim={sims_v[layer_idx].mean():.4f}")
                print()
        
        # Clean up
        del kv_data
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
