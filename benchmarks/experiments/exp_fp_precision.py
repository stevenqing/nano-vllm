"""FP Precision Test: Does nano-vllm's KV reuse produce bit-identical output?

Compares within the SAME engine:
1. generate() — full prefill each round (ground truth)
2. chat() — KV reuse across rounds

If they differ, the issue is in our KV reuse implementation.
If identical, the nano-vllm vs vLLM difference is from kernel-level FP differences.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-8B/")

def main():
    llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=4096)
    sp = SamplingParams(temperature=0, max_tokens=30)
    
    # Warmup
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=4))
    
    conversations = [
        ["Hello, what is 2+2?", "Now multiply that by 3.", "What is the square root?"],
        ["Write a Python hello world.", "Add a loop that prints 1 to 5.", "Add error handling."],
        ["What is the capital of France?", "What about Germany?", "And Spain?"],
    ]
    
    all_match = True
    
    for conv_idx, messages in enumerate(conversations):
        print(f"\n{'='*60}")
        print(f"Conversation {conv_idx}")
        print(f"{'='*60}")
        
        # Method 1: generate() — full prefill each round, using TOKEN IDS
        gen_outputs = []
        gen_all_ids = []
        first_str = f"User: {messages[0]}\nAssistant:"
        gen_all_ids = list(llm.tokenizer.encode(first_str))
        result = llm.generate([gen_all_ids], sp, use_tqdm=False)
        gen_outputs.append(list(result[0]["token_ids"]))
        gen_all_ids = gen_all_ids + list(result[0]["token_ids"])
        
        for msg in messages[1:]:
            suffix_str = f"\nUser: {msg}\nAssistant:"
            suffix_ids = list(llm.tokenizer.encode(suffix_str, add_special_tokens=False))
            gen_all_ids = gen_all_ids + suffix_ids
            result = llm.generate([gen_all_ids], sp, use_tqdm=False)
            gen_outputs.append(list(result[0]["token_ids"]))
            gen_all_ids = gen_all_ids + list(result[0]["token_ids"])
        
        # Method 2: chat() — KV reuse, using TOKEN IDS (not strings)
        chat_outputs = []
        first_str = f"User: {messages[0]}\nAssistant:"
        first_ids = llm.tokenizer.encode(first_str)
        result = llm.chat(first_ids, sp)
        seq_id = result["seq_id"]
        chat_outputs.append(list(result["token_ids"]))
        
        # Track full context as token IDs (same as generate)
        all_ids = first_ids + list(result["token_ids"])
        
        for msg in messages[1:]:
            # Build suffix and tokenize WITHOUT special tokens
            suffix_str = f"\nUser: {msg}\nAssistant:"
            suffix_ids = llm.tokenizer.encode(suffix_str, add_special_tokens=False)
            all_ids = all_ids + suffix_ids
            
            # Pass only the new suffix token IDs
            result = llm.chat(suffix_ids, sp, seq_id=seq_id)
            # chat() returns completion_token_ids which includes suffix + output
            # We need to strip the suffix to compare with generate()
            raw_completion = list(result["token_ids"])
            chat_output = raw_completion[len(suffix_ids):]  # strip suffix
            chat_outputs.append(chat_output)
            all_ids = all_ids + list(result["token_ids"])[len(suffix_ids):]  # only output tokens
        
        llm.release_session(seq_id)
        
        # Compare
        for round_idx in range(len(messages)):
            gen_toks = gen_outputs[round_idx]
            chat_toks = chat_outputs[round_idx]
            
            match = (gen_toks == chat_toks)
            if not match:
                all_match = False
                min_len = min(len(gen_toks), len(chat_toks))
                first_diff = min_len
                for i in range(min_len):
                    if gen_toks[i] != chat_toks[i]:
                        first_diff = i
                        break
                print(f"  Round {round_idx}: MISMATCH at pos {first_diff}")
                print(f"    gen  [{len(gen_toks)} tok]: first5={gen_toks[:5]} ...diff={gen_toks[max(0,first_diff-2):first_diff+3]}...")
                print(f"    chat [{len(chat_toks)} tok]: first5={chat_toks[:5]} ...diff={chat_toks[max(0,first_diff-2):first_diff+3]}...")
                # Decode both to see text
                print(f"    gen_text:  {llm.tokenizer.decode(gen_toks[:first_diff+5])!r}")
                print(f"    chat_text: {llm.tokenizer.decode(chat_toks[:first_diff+5])!r}")
            else:
                print(f"  Round {round_idx}: MATCH ({len(gen_toks)} tokens)")
    
    print(f"\n{'='*60}")
    if all_match:
        print("RESULT: ALL ROUNDS BIT-IDENTICAL ✅")
        print("KV reuse produces exact same output as full prefill.")
        print("Any nano-vllm vs vLLM difference is from kernel-level FP differences.")
    else:
        print("RESULT: MISMATCHES FOUND ❌")
        print("KV reuse introduces numerical differences.")
        print("Need to investigate partial block re-computation.")

if __name__ == "__main__":
    main()
