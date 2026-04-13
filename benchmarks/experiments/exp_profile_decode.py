"""Profile nano-vllm decode step breakdown: where does time go?"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-8B/")

def main():
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096)
    sp = SamplingParams(temperature=0, max_tokens=1)
    llm.generate(["warmup warmup warmup"], SamplingParams(temperature=0, max_tokens=8))
    
    # Create a session with some context
    prompt = "You are an assistant. " * 20 + "\nUser: Tell me about machine learning.\nAssistant:"
    prompt_ids = llm.tokenizer.encode(prompt)
    print(f"Prompt: {len(prompt_ids)} tokens")
    
    result = llm.chat(prompt_ids, SamplingParams(temperature=0, max_tokens=50))
    seq_id = result["seq_id"]
    print(f"Round 0: {len(result['token_ids'])} tokens generated")
    
    # Now profile individual decode steps in chat mode
    suffix_ids = llm.tokenizer.encode("\nUser: Continue.\nAssistant:", add_special_tokens=False)
    
    # Resume and manually step through
    llm.resume_request(seq_id, suffix_ids, SamplingParams(temperature=0, max_tokens=100))
    
    # Profile each step
    times = {"schedule": [], "copy": [], "run": [], "postprocess": [], "total": []}
    
    for step in range(50):
        t0 = time.perf_counter()
        
        # Schedule
        t_sched_start = time.perf_counter()
        seqs, num_prefill_seqs = llm.scheduler.schedule()
        t_sched = time.perf_counter() - t_sched_start
        
        # Copies
        t_copy_start = time.perf_counter()
        for src, dst in llm.scheduler.block_manager.get_pending_copies():
            llm.model_runner.call("copy_block_kv", src, dst)
        t_copy = time.perf_counter() - t_copy_start
        
        # Model run
        t_run_start = time.perf_counter()
        token_ids = llm.model_runner.call("run", seqs, num_prefill_seqs)
        t_run = time.perf_counter() - t_run_start
        
        # Postprocess
        t_post_start = time.perf_counter()
        llm.scheduler.postprocess(seqs, token_ids, num_prefill_seqs)
        t_post = time.perf_counter() - t_post_start
        
        t_total = time.perf_counter() - t0
        
        if num_prefill_seqs > 0:
            print(f"  step {step}: PREFILL {t_total*1000:.2f}ms")
            continue
        
        times["schedule"].append(t_sched)
        times["run"].append(t_run)
        times["copy"].append(t_copy)
        times["postprocess"].append(t_post)
        times["total"].append(t_total)
        
        if seqs[0].is_finished or seqs[0].status.name == "WAITING_FOR_NEXT_ROUND":
            break
    
    llm.release_session(seq_id)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Decode step breakdown (avg over {len(times['total'])} steps):")
    print(f"{'='*60}")
    for key in ["schedule", "copy", "run", "postprocess", "total"]:
        vals = times[key]
        if vals:
            avg = sum(vals)/len(vals)*1000
            pct = avg / (sum(times["total"])/len(times["total"])*1000) * 100
            print(f"  {key:15s}: {avg:8.3f} ms ({pct:5.1f}%)")
    
    total_avg = sum(times["total"])/len(times["total"])*1000
    overhead = total_avg - sum(times["run"])/len(times["run"])*1000
    print(f"\n  Python overhead: {overhead:.3f} ms ({overhead/total_avg*100:.1f}%)")
    print(f"  Effective tok/s: {1000/total_avg:.1f}")

if __name__ == "__main__":
    main()
