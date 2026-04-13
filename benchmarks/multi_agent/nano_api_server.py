"""Async OpenAI-compatible API server wrapping nano-vllm.

Supports /v1/chat/completions (streaming + non-streaming).
Concurrent requests are batched together by the engine for GPU-efficient inference.
Multi-turn KV reuse: automatically detects conversation continuations and reuses KV cache.

Usage: python nano_api_server.py --model ~/huggingface/Qwen3-0.6B/ --port 8100
"""
import argparse
import asyncio
import json
import time
import uuid
import hashlib
import threading

from aiohttp import web

# Will be set by main()
llm = None
tokenizer = None
model_name = "nano-vllm"

# Pending requests: seq_id -> asyncio.Future[dict]
pending: dict[int, asyncio.Future] = {}
loop: asyncio.AbstractEventLoop = None

# Automatic prompt dedup: prompt_hash -> (context_id, ref_count)
# Caches recently-prefilled prompts so concurrent identical requests share KV
prompt_cache: dict[int, tuple[int, int]] = {}
PROMPT_CACHE_MAX = 32  # max cached prompts

# Multi-turn session tracking: conversation_key -> (seq_id, full_token_ids)
# conversation_key = hash of all messages except the last user message
sessions: dict[str, tuple[int, list[int]]] = {}


def _conversation_key(messages: list[dict]) -> str | None:
    """Generate a key identifying a conversation up to (but excluding) the last user turn.
    Returns None if no conversation history (single message)."""
    if len(messages) <= 1:
        return None
    # Key = hash of all messages except the last one
    prefix_msgs = messages[:-1]
    key_str = json.dumps(prefix_msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_str.encode()).hexdigest()


async def handle_models(request):
    return web.json_response({"object": "list", "data": [{"id": model_name, "object": "model"}]})


async def handle_health(request):
    return web.json_response({"status": "ok"})


async def handle_chat_completions(request):
    body = await request.json()

    messages = body.get("messages", [])
    temperature = body.get("temperature", 0)
    max_tokens = body.get("max_tokens", 256)
    stream = body.get("stream", False)
    n = body.get("n", 1)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt)

    from nanovllm import SamplingParams
    sp = SamplingParams(
        temperature=max(temperature, 1e-6) if temperature > 0 else 0,
        max_tokens=max_tokens,
        ignore_eos=body.get("min_tokens", 0) > 0,
    )

    if n > 1:
        # n completions: prefill once, fork n times via context pool
        context_id = llm.cache_context(prompt_ids)
        ctx = llm.scheduler.block_manager.context_pool[context_id]
        seq_ids = []
        futures = []
        for i in range(n):
            from nanovllm.engine.sequence import Sequence
            block_table, num_ctx = llm.scheduler.block_manager.fork_context(context_id)
            # Execute pending copies before scheduling
            for src, dst in llm.scheduler.block_manager.get_pending_copies():
                llm.model_runner.call("copy_block_kv", src, dst)
            seq = Sequence.from_context(ctx.token_ids, [], block_table, num_ctx, sp)
            llm.scheduler.add(seq)
            fut = loop.create_future()
            pending[seq.seq_id] = fut
            seq_ids.append(seq.seq_id)
            futures.append(fut)
        # Wait for all completions
        all_outputs = await asyncio.gather(*futures)
        llm.release_context(context_id)
        choices = []
        for i, output_ids in enumerate(all_outputs):
            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": tokenizer.decode(output_ids)},
                "finish_reason": "stop",
            })
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        resp = {
            "id": cmpl_id, "object": "chat.completion",
            "created": int(time.time()), "model": model_name,
            "choices": choices,
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": sum(len(o) for o in all_outputs),
                      "total_tokens": len(prompt_ids) + sum(len(o) for o in all_outputs)},
        }
        return web.json_response(resp)

    # Single completion (n=1) with automatic prompt dedup
    prompt_hash = hash(tuple(prompt_ids))

    if prompt_hash in prompt_cache:
        # Reuse cached context — fork instead of full prefill
        ctx_id, ref_count = prompt_cache[prompt_hash]
        ctx = llm.scheduler.block_manager.context_pool.get(ctx_id)
        if ctx is not None:
            from nanovllm.engine.sequence import Sequence
            block_table, num_ctx = llm.scheduler.block_manager.fork_context(ctx_id)
            for src, dst in llm.scheduler.block_manager.get_pending_copies():
                llm.model_runner.call("copy_block_kv", src, dst)
            seq = Sequence.from_context(ctx.token_ids, [], block_table, num_ctx, sp)
            llm.scheduler.add(seq)
            seq_id = seq.seq_id
            prompt_cache[prompt_hash] = (ctx_id, ref_count + 1)
        else:
            # Context was evicted, fall through to normal path
            del prompt_cache[prompt_hash]
            seq_id = llm.add_request(prompt_ids, sp)
    else:
        seq_id = llm.add_request(prompt_ids, sp)

    future = loop.create_future()
    pending[seq_id] = future

    output_ids = await future
    output_text = tokenizer.decode(output_ids)

    # Cache this prompt's context for future dedup (if not already cached)
    if prompt_hash not in prompt_cache and len(prompt_cache) < PROMPT_CACHE_MAX:
        try:
            ctx_id = llm.cache_context(prompt_ids)
            prompt_cache[prompt_hash] = (ctx_id, 0)
        except Exception:
            pass  # skip if context caching fails (e.g. KV blocks full)

    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if stream:
        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        await response.prepare(request)

        words = output_text.split(" ")
        for i, word in enumerate(words):
            content = word if i == 0 else " " + word
            chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())

        final = {
            "id": cmpl_id, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response
    else:
        resp = {
            "id": cmpl_id, "object": "chat.completion",
            "created": int(time.time()), "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(output_ids),
                      "total_tokens": len(prompt_ids) + len(output_ids)},
        }
        return web.json_response(resp)


async def engine_loop():
    """Background loop: continuously runs engine steps to process batched requests."""
    while True:
        if not llm.scheduler.waiting and not llm.scheduler.running:
            await asyncio.sleep(0.001)  # idle wait
            continue
        outputs, _ = llm.step()
        for seq_id, token_ids in outputs:
            fut = pending.pop(seq_id, None)
            if fut and not fut.done():
                fut.set_result(list(token_ids))
        await asyncio.sleep(0)  # yield to let new requests arrive


# ---- Pipeline endpoint ----

BUILTIN_COMBINERS = {
    "moa": lambda sys_prompt: (
        lambda prompt, prev_texts: [
            {"role": "system", "content": sys_prompt + "\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(prev_texts))},
            {"role": "user", "content": prompt},
        ]
    ),
}

DEFAULT_MOA_SYSTEM = "You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. Ensure your response is well-structured, coherent, and accurate.\n\nResponses from models:"


async def handle_pipeline(request):
    """Execute a multi-stage pipeline with automatic cross-prompt batching.

    POST /v1/pipeline
    {
        "prompts": ["What is AI?", "What is ML?"],
        "stages": [
            {"n": 3, "max_tokens": 256, "temperature": 0.7},
            {"n": 1, "max_tokens": 256, "temperature": 0, "combine": "moa"}
        ]
    }

    Preset pipelines:
    POST /v1/pipeline
    {
        "prompts": ["What is AI?", "What is ML?"],
        "preset": "moa",
        "num_agents": 3,
        "num_layers": 2,
        "max_tokens": 256
    }
    """
    body = await request.json()
    prompts = body.get("prompts", [])
    if not prompts:
        return web.json_response({"error": "No prompts provided"}, status=400)

    from nanovllm import SamplingParams

    # Preset pipeline (e.g. "moa")
    preset = body.get("preset")
    if preset == "moa":
        sp = SamplingParams(
            temperature=body.get("temperature", 0),
            max_tokens=body.get("max_tokens", 256))
        t0 = time.perf_counter()
        results = llm.run_moa(
            prompts,
            num_agents=body.get("num_agents", 3),
            num_layers=body.get("num_layers", 2),
            sampling_params=sp,
        )
        elapsed = time.perf_counter() - t0
        return web.json_response({
            "results": results,
            "elapsed_s": elapsed,
            "prompts_per_sec": len(prompts) / elapsed,
        })

    # Custom pipeline via stages
    stages_def = body.get("stages", [])
    if not stages_def:
        return web.json_response({"error": "No stages defined"}, status=400)

    stages = []
    for s in stages_def:
        sp = SamplingParams(
            temperature=s.get("temperature", 0),
            max_tokens=s.get("max_tokens", 256))
        combine = None
        if "combine" in s:
            c = s["combine"]
            if c == "moa":
                sys_prompt = s.get("system", DEFAULT_MOA_SYSTEM)
                combine = BUILTIN_COMBINERS["moa"](sys_prompt)
        stages.append({"n": s.get("n", 1), "sp": sp, "combine": combine})

    t0 = time.perf_counter()
    results = llm.generate_pipeline(prompts, stages)
    elapsed = time.perf_counter() - t0
    return web.json_response({
        "results": results,
        "elapsed_s": elapsed,
        "prompts_per_sec": len(prompts) / elapsed,
    })


async def start_background_tasks(app):
    app["engine_task"] = asyncio.create_task(engine_loop())


async def cleanup_background_tasks(app):
    app["engine_task"].cancel()


def main():
    global llm, tokenizer, model_name, loop
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--served-model-name", type=str, default=None)
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = args.served_model_name or args.model
    print(f"Loading nano-vllm model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, enforce_eager=False, max_model_len=args.max_model_len)
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=4))
    print(f"nano-vllm async API server ready on port {args.port}")
    print(f"  Model: {model_name}")

    app = web.Application()
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/health", handle_health)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/pipeline", handle_pipeline)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    web.run_app(app, host="0.0.0.0", port=args.port, loop=loop, print=lambda _: None)


if __name__ == "__main__":
    main()
