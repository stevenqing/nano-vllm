import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, resumable: bool = False):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, resumable=resumable)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, num_prefill_seqs = self.scheduler.schedule()
        # Execute pending KV block copies (for context fork CoW)
        for src_id, dst_id in self.scheduler.block_manager.get_pending_copies():
            self.model_runner.call("copy_block_kv", src_id, dst_id)
        token_ids = self.model_runner.call("run", seqs, num_prefill_seqs)
        self.scheduler.postprocess(seqs, token_ids, num_prefill_seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished or seq.status == SequenceStatus.WAITING_FOR_NEXT_ROUND]
        num_tokens = sum(seq.num_computed_tokens - getattr(seq, '_chunk_start', 0) for seq in seqs[:num_prefill_seqs]) if num_prefill_seqs > 0 else -len(seqs)
        return outputs, num_tokens

    def resume_request(self, seq_id: int, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.resume_request(seq_id, prompt, sampling_params)

    def release_session(self, seq_id: int):
        self.scheduler.release_session(seq_id)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def chat(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        seq_id: int | None = None,
    ) -> dict:
        """Multi-round chat interface. Returns dict with 'text', 'token_ids', 'seq_id'.
        Pass seq_id from previous round to continue the conversation."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if seq_id is not None:
            self.resume_request(seq_id, prompt, sampling_params)
        else:
            seq_id = self.add_request(prompt, sampling_params, resumable=True)
        while not self.is_finished():
            output, _ = self.step()
            for sid, token_ids in output:
                if sid == seq_id:
                    return {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "seq_id": seq_id}
        raise RuntimeError("chat request not found")

    # ---- Context Pool (KV Fork) API ----

    def cache_context(self, context: str | list[int]) -> int:
        """Prefill a shared context and cache its KV blocks. Returns context_id."""
        if isinstance(context, str):
            context = self.tokenizer.encode(context)
        # Use resumable=True to prevent deallocation after first token
        seq = Sequence(context, SamplingParams(max_tokens=1, ignore_eos=True), resumable=True)
        self.scheduler.add(seq)
        # Run until seq completes one token (enters WAITING_FOR_NEXT_ROUND)
        while seq.status != SequenceStatus.WAITING_FOR_NEXT_ROUND:
            self.step()
        # Remove from sessions
        self.scheduler.sessions.pop(seq.seq_id, None)
        # Cache the block_table (transfers ownership)
        context_id = self.scheduler.block_manager.cache_context(seq)
        # Clear seq's reference so blocks aren't double-freed
        seq.block_table = []
        return context_id

    def generate_with_context(
        self,
        suffixes: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        context_id: int,
        use_tqdm: bool = False,
    ) -> list[dict]:
        """Generate with a pre-cached context prefix for multiple agents in parallel."""
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(suffixes)
        ctx = self.scheduler.block_manager.context_pool[context_id]

        seq_ids = []
        for suffix, sp in zip(suffixes, sampling_params):
            if isinstance(suffix, str):
                suffix = self.tokenizer.encode(suffix)
            block_table, num_ctx_tokens = self.scheduler.block_manager.fork_context(context_id)
            seq = Sequence.from_context(ctx.token_ids, suffix, block_table, num_ctx_tokens, sp)
            self.scheduler.add(seq)
            seq_ids.append(seq.seq_id)

        outputs = {}
        while not self.is_finished():
            output, _ = self.step()
            for sid, token_ids in output:
                outputs[sid] = token_ids

        results = []
        for sid in seq_ids:
            tids = outputs[sid]
            results.append({"text": self.tokenizer.decode(tids), "token_ids": tids, "seq_id": sid})
        return results

    def release_context(self, context_id: int):
        """Release a cached context."""
        self.scheduler.block_manager.release_context(context_id)

    # ---- Pipeline Scheduler ----

    def _run_stage(self, prompts_per_item: list[list[list[int]]], sp: SamplingParams) -> list[list[list[int]]]:
        """Run one pipeline stage: submit all seqs, wait for completion, return outputs grouped by item.
        prompts_per_item[i][j] = token_ids for item i, agent j."""
        seq_map = []  # (item_idx, seq_id)
        for i, agent_prompts in enumerate(prompts_per_item):
            for pids in agent_prompts:
                sid = self.add_request(pids, sp)
                seq_map.append((i, sid))

        completed = {}
        while len(completed) < len(seq_map):
            output, _ = self.step()
            for sid, tids in output:
                completed[sid] = tids

        results = [[] for _ in range(len(prompts_per_item))]
        for i, sid in seq_map:
            results[i].append(list(completed[sid]))
        return results

    def generate_pipeline(
        self,
        prompts: list[str],
        stages: list[dict],
    ) -> list[dict]:
        """General multi-stage generation pipeline with automatic cross-prompt batching.

        Each stage is a dict with:
          - "n": int — number of parallel agents per prompt (default 1)
          - "sp": SamplingParams
          - "combine": callable(prompt_str, prev_outputs: list[str]) -> list[dict]
                       builds messages for this stage from previous outputs.
                       If None, uses the raw prompt.

        All prompts' agents within a stage decode simultaneously (high BS).

        Example — MoA 2-layer:
            results = llm.generate_pipeline(prompts, [
                {"n": 3, "sp": sp},  # L1: 3 agents per prompt
                {"n": 1, "sp": sp, "combine": moa_aggregator},  # L2: aggregate
            ])
        """
        n_items = len(prompts)
        prev_outputs = None  # list[list[str]], per item

        for stage_idx, stage in enumerate(stages):
            n_agents = stage.get("n", 1)
            sp = stage.get("sp", SamplingParams(temperature=0, max_tokens=256))
            combine_fn = stage.get("combine", None)

            prompts_per_item = []
            for i in range(n_items):
                if combine_fn and prev_outputs:
                    # Build messages from previous outputs
                    msgs = combine_fn(prompts[i], prev_outputs[i])
                    text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    ids = self.tokenizer.encode(text)
                else:
                    # First stage: use raw prompt
                    text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True)
                    ids = self.tokenizer.encode(text)
                prompts_per_item.append([ids] * n_agents)

            # Submit ALL agents for ALL items → max batch size
            output_ids = self._run_stage(prompts_per_item, sp)

            # Decode outputs to text for next stage
            prev_outputs = []
            for item_outputs in output_ids:
                prev_outputs.append([self.tokenizer.decode(o) for o in item_outputs])

        return [{"text": prev_outputs[i][0], "prompt_index": i} for i in range(n_items)]

    def run_moa(
        self,
        prompts: list[str],
        num_agents: int = 3,
        num_layers: int = 2,
        sampling_params: SamplingParams | None = None,
        aggregator_system: str = "You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. Ensure your response is well-structured, coherent, and accurate.\n\nResponses from models:",
    ) -> list[dict]:
        """Mixture-of-Agents via generate_pipeline."""
        sp = sampling_params or SamplingParams(temperature=0, max_tokens=256)

        def moa_combine(prompt, prev_texts):
            agg = aggregator_system + "\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(prev_texts))
            return [{"role": "system", "content": agg}, {"role": "user", "content": prompt}]

        stages = [{"n": num_agents, "sp": sp}]
        for layer in range(1, num_layers - 1):
            stages.append({"n": num_agents, "sp": sp, "combine": moa_combine})
        stages.append({"n": 1, "sp": SamplingParams(temperature=0, max_tokens=sp.max_tokens), "combine": moa_combine})

        return self.generate_pipeline(prompts, stages)

    def run_debate(
        self,
        tasks: list[tuple[str, str, str]],
        num_turns: int = 5,
        sampling_params: SamplingParams | None = None,
    ) -> list[list[str]]:
        """Multi-task debate: pairs of agents alternate, all tasks batched per turn.

        tasks = [(role_a, role_b, task_description), ...]
        Returns list of conversation transcripts per task.

        All tasks' current turn agents decode simultaneously (BS = len(tasks)).
        """
        sp = sampling_params or SamplingParams(temperature=0, max_tokens=150)
        n = len(tasks)

        # Build system prompts
        sys_a = [f"You are a {r}. Discuss: {t}. Keep responses to 2-3 sentences." for r, _, t in tasks]
        sys_b = [f"You are a {r}. Discuss: {t}. Keep responses to 2-3 sentences." for _, r, t in tasks]

        transcripts = [[] for _ in range(n)]
        last_texts = [f"Let's discuss: {t}. What's your initial approach?" for _, _, t in tasks]

        for turn in range(num_turns * 2 - 1):
            is_a = (turn % 2 == 0)
            systems = sys_a if is_a else sys_b

            # Build prompts for all tasks at this turn
            prompts_per_item = []
            for i in range(n):
                msgs = [{"role": "system", "content": systems[i]},
                        {"role": "user", "content": last_texts[i]}]
                text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = self.tokenizer.encode(text)
                prompts_per_item.append([ids])

            # All tasks decode simultaneously (BS = n)
            outputs = self._run_stage(prompts_per_item, sp)

            for i in range(n):
                out_text = self.tokenizer.decode(outputs[i][0])
                transcripts[i].append(out_text)
                last_texts[i] = out_text

        return transcripts
