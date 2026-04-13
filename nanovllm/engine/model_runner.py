import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        if not self.enforce_eager:
            self.compile_model()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # Use eager mode for memory profiling warmup to avoid compile-time allocation spikes
        enforce_eager_backup = self.enforce_eager
        self.enforce_eager = True
        self.run(seqs, num_seqs)  # all prefill
        self.enforce_eager = enforce_eager_backup
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            chunk_start = getattr(seq, '_chunk_start', seq.num_cached_tokens)
            chunk_end = seq.num_computed_tokens if seq.num_computed_tokens > 0 else seqlen
            input_ids.extend(seq[chunk_start:chunk_end])
            positions.extend(list(range(chunk_start, chunk_end)))
            seqlen_q = chunk_end - chunk_start
            seqlen_k = chunk_end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for pos in range(chunk_start, chunk_end):
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + offset)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache or chunked
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_chunked(self, seqs: list[Sequence], num_prefill_seqs: int):
        """Prepare inputs for a mixed batch of prefill chunks + decode tokens.
        Uses flash_attn_varlen_func with block_table for all sequences."""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        # Prefill sequences (chunked)
        for seq in seqs[:num_prefill_seqs]:
            chunk_start = seq._chunk_start
            chunk_end = seq.num_computed_tokens
            chunk_size = chunk_end - chunk_start

            input_ids.extend(seq[chunk_start:chunk_end])
            positions.extend(range(chunk_start, chunk_end))

            cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
            cu_seqlens_k.append(cu_seqlens_k[-1] + chunk_end)
            max_seqlen_q = max(max_seqlen_q, chunk_size)
            max_seqlen_k = max(max_seqlen_k, chunk_end)

            for pos in range(chunk_start, chunk_end):
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + offset)

        # Decode sequences (1 token each)
        for seq in seqs[num_prefill_seqs:]:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)

            cu_seqlens_q.append(cu_seqlens_q[-1] + 1)
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))
            max_seqlen_q = max(max_seqlen_q, 1)
            max_seqlen_k = max(max_seqlen_k, len(seq))

            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode_direct(self, seqs: list[Sequence]):
        """Write decode inputs directly into CUDA graph buffers — zero tensor allocation."""
        bs = len(seqs)
        gv = self.graph_vars
        # Use numpy views for fast CPU writes (avoids InferenceMode issues)
        np_ids = self._np_staging["input_ids"]
        np_pos = self._np_staging["positions"]
        np_ctx = self._np_staging["context_lens"]
        np_slot = self._np_staging["slot_mapping"]
        np_bt = self._np_staging["block_tables"]
        max_bt_len = 0
        for i, seq in enumerate(seqs):
            np_ids[i] = seq.last_token
            np_pos[i] = len(seq) - 1
            np_ctx[i] = len(seq)
            np_slot[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            bt = seq.block_table
            bt_len = len(bt)
            if bt_len > max_bt_len:
                max_bt_len = bt_len
            np_bt[i, :bt_len] = bt
            np_bt[i, bt_len:] = 0
        # Bulk copy: pinned CPU → GPU graph buffers
        gv["input_ids"][:bs].copy_(self._cpu_staging["input_ids"][:bs], non_blocking=True)
        gv["positions"][:bs].copy_(self._cpu_staging["positions"][:bs], non_blocking=True)
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs].copy_(self._cpu_staging["slot_mapping"][:bs], non_blocking=True)
        gv["context_lens"].zero_()
        gv["context_lens"][:bs].copy_(self._cpu_staging["context_lens"][:bs], non_blocking=True)
        gv["block_tables"][:bs, :max_bt_len].copy_(self._cpu_staging["block_tables"][:bs, :max_bt_len], non_blocking=True)
        if max_bt_len < gv["block_tables"].size(1):
            gv["block_tables"][:bs, max_bt_len:].zero_()

    def prepare_decode_legacy(self, seqs: list[Sequence]):
        """Legacy prepare_decode: creates new tensors. Used for eager mode and warmup."""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def prepare_sample_direct(self, seqs: list[Sequence]):
        """Write temperatures into pre-allocated staging buffer."""
        bs = len(seqs)
        np_temps = self._np_staging["temperatures"]
        for i, seq in enumerate(seqs):
            np_temps[i] = seq.temperature
        self._gpu_temps[:bs].copy_(self._cpu_staging["temperatures"][:bs], non_blocking=True)
        return self._gpu_temps[:bs]

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Legacy CUDA graph path (used when prepare_decode_legacy is called)
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            gv = self.graph_vars
            gv["input_ids"][:bs] = input_ids
            gv["positions"][:bs] = positions
            gv["slot_mapping"].fill_(-1)
            gv["slot_mapping"][:bs] = context.slot_mapping
            gv["context_lens"].zero_()
            gv["context_lens"][:bs] = context.context_lens
            gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return gv["logits"][:bs] if self.rank == 0 else None

    @torch.inference_mode()
    def run_model_decode_direct(self, bs: int):
        """Run decode with data already written into graph_vars. No tensor copy."""
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph.replay()
        return self.graph_vars["logits"][:bs] if self.rank == 0 else None

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], num_prefill_seqs: int) -> list[int]:
        is_prefill = num_prefill_seqs > 0
        if is_prefill:
            num_decode_seqs = len(seqs) - num_prefill_seqs
            if num_decode_seqs > 0:
                input_ids, positions = self.prepare_chunked(seqs, num_prefill_seqs)
            else:
                input_ids, positions = self.prepare_prefill(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, True)
        elif not self.enforce_eager and len(seqs) <= 512:
            # Fast path: write directly into graph buffers, skip tensor allocation
            self.prepare_decode_direct(seqs)
            temperatures = self.prepare_sample_direct(seqs) if self.rank == 0 else None
            logits = self.run_model_decode_direct(len(seqs))
        else:
            input_ids, positions = self.prepare_decode_legacy(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, False)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def copy_block_kv(self, src_id: int, dst_id: int):
        """Copy KV cache data from src block to dst block (all layers)."""
        self.kv_cache[:, :, dst_id].copy_(self.kv_cache[:, :, src_id])

    @torch.inference_mode()
    def compile_model(self):
        """Compile the transformer with torch.compile.
        Attention.forward is excluded (@torch.compiler.disable), so inductor
        can fuse norms, projections, RoPE, and MLP across layer boundaries."""
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = 1
        import torch._inductor.runtime.triton_heuristics as th
        th.TRITON_MAX_BLOCK["X"] = 16384
        self.model.model = torch.compile(self.model.model, fullgraph=False)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        vocab_size = hf_config.vocab_size
        logits = torch.zeros(max_bs, vocab_size) if self.rank == 0 else None
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.rank == 0:
                logits[:bs] = self.model.compute_logits(outputs[:bs])
            # capture: model forward + compute_logits in one graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                if self.rank == 0:
                    logits[:bs] = self.model.compute_logits(outputs[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            logits=logits,
        )
        # Pre-allocate pinned CPU staging buffers for prepare_decode_direct
        self._cpu_staging = dict(
            input_ids=torch.zeros(max_bs, dtype=torch.int64, device="cpu").pin_memory(),
            positions=torch.zeros(max_bs, dtype=torch.int64, device="cpu").pin_memory(),
            slot_mapping=torch.zeros(max_bs, dtype=torch.int32, device="cpu").pin_memory(),
            context_lens=torch.zeros(max_bs, dtype=torch.int32, device="cpu").pin_memory(),
            block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cpu").pin_memory(),
            temperatures=torch.zeros(max_bs, dtype=torch.float32, device="cpu").pin_memory(),
        )
        self._gpu_temps = torch.zeros(max_bs, dtype=torch.float32, device="cuda")
        # Numpy views into pinned CPU staging for fast Python writes
        self._np_staging = {k: v.numpy() for k, v in self._cpu_staging.items()}
