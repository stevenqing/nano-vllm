from collections import deque

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.sessions: dict[int, Sequence] = {}  # seq_id -> seq for resumable seqs

    def is_finished(self):
        return not self.waiting and not self.running

    def has_active_sessions(self):
        return bool(self.sessions)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], int]:
        """Schedule sequences for execution.
        Returns (seqs, num_prefill_seqs):
          - seqs[:num_prefill_seqs] are prefill (possibly chunked)
          - seqs[num_prefill_seqs:] are decode (1 token each)
        When num_prefill_seqs > 0, use unified varlen attention path.
        When num_prefill_seqs == 0, use CUDA graph decode path.
        """
        scheduled_prefill = []
        scheduled_decode = []
        num_batched_tokens = 0

        # Phase 1: Schedule decode tokens from running queue (1 token each)
        remaining_running = deque()
        while self.running:
            seq = self.running.popleft()
            if len(scheduled_decode) >= self.max_num_seqs:
                remaining_running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                elif remaining_running:
                    self.preempt(remaining_running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                scheduled_decode.append(seq)
                num_batched_tokens += 1
        self.running = remaining_running

        # Phase 2: Fill remaining budget with prefill chunks from waiting queue
        num_total_seqs = len(scheduled_decode)
        while self.waiting and num_total_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # Allocate blocks (only on first chunk for this sequence)
            if seq.num_computed_tokens == 0:
                if seq.block_table:
                    # Resumed or context-forked sequence — incremental allocation
                    if not self.block_manager.can_allocate_incremental(seq):
                        break
                    self.block_manager.allocate_incremental(seq)
                else:
                    # Fresh sequence — full allocation
                    if not self.block_manager.can_allocate(seq):
                        break
                    self.block_manager.allocate(seq)
                seq.num_computed_tokens = seq.num_cached_tokens

            remaining = len(seq) - seq.num_computed_tokens
            chunk_size = min(remaining, self.max_num_batched_tokens - num_batched_tokens)
            if chunk_size <= 0:
                break

            self.waiting.popleft()
            seq._chunk_start = seq.num_computed_tokens
            seq.num_computed_tokens += chunk_size
            seq.status = SequenceStatus.RUNNING
            scheduled_prefill.append(seq)
            num_batched_tokens += chunk_size
            num_total_seqs += 1

            if seq.num_computed_tokens < len(seq):
                # Partially prefilled → back to waiting for next chunk
                self.waiting.appendleft(seq)
            else:
                # Fully prefilled → move to running (will decode next step)
                self.running.append(seq)

        all_seqs = scheduled_prefill + scheduled_decode
        assert all_seqs, "No sequences to schedule"

        if scheduled_decode and not scheduled_prefill:
            # Pure decode: sort by block_table prefix for L2 cache locality
            all_seqs.sort(key=lambda s: tuple(s.block_table[:4]))
        self.running.extendleft(reversed(scheduled_decode))
        return all_seqs, len(scheduled_prefill)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], num_prefill_seqs: int = 0) -> list[bool]:
        for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
            # Skip partially-prefilled sequences (intermediate chunks)
            if i < num_prefill_seqs and seq.num_computed_tokens < len(seq):
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens or len(seq) >= self.max_model_len:
                if seq.resumable:
                    seq.status = SequenceStatus.WAITING_FOR_NEXT_ROUND
                    self.sessions[seq.seq_id] = seq
                else:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def resume_request(self, seq_id: int, new_token_ids: list[int], sampling_params: SamplingParams):
        """Resume a finished resumable sequence with new tokens for the next round."""
        seq = self.sessions.get(seq_id)
        assert seq is not None and seq.status == SequenceStatus.WAITING_FOR_NEXT_ROUND
        seq.resume(new_token_ids, sampling_params)
        self.waiting.append(seq)

    def release_session(self, seq_id: int):
        """Release a session's KV cache blocks."""
        seq = self.sessions.pop(seq_id, None)
        if seq is not None:
            self.block_manager.deallocate(seq)
