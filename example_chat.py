"""Multi-round chat example demonstrating KV cache reuse across conversation turns."""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)

    # Build initial prompt
    messages = [{"role": "user", "content": "Hello! What is 2+3?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt)

    # Round 1
    result = llm.chat(prompt_ids, sampling_params)
    print(f"Round 1 output: {result['text']!r}")
    seq_id = result["seq_id"]

    # Round 2: continue the conversation
    messages_round2 = [{"role": "user", "content": "Now multiply the result by 10."}]
    new_prompt = tokenizer.apply_chat_template(messages_round2, tokenize=False, add_generation_prompt=True)
    # In multi-round, we send the assistant's output + new user turn
    assistant_output = result["text"]
    continuation = assistant_output + new_prompt
    continuation_ids = tokenizer.encode(continuation)

    result2 = llm.chat(continuation_ids, sampling_params, seq_id=seq_id)
    print(f"Round 2 output: {result2['text']!r}")

    # Round 3: one more turn
    messages_round3 = [{"role": "user", "content": "What is the square root of that?"}]
    new_prompt3 = tokenizer.apply_chat_template(messages_round3, tokenize=False, add_generation_prompt=True)
    continuation3 = result2["text"] + new_prompt3
    continuation_ids3 = tokenizer.encode(continuation3)

    result3 = llm.chat(continuation_ids3, sampling_params, seq_id=seq_id)
    print(f"Round 3 output: {result3['text']!r}")

    # Release session when done
    llm.release_session(seq_id)


if __name__ == "__main__":
    main()
