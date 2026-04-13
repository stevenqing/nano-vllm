from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    WAITING_FOR_NEXT_ROUND = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), resumable: bool = False):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_computed_tokens = 0  # tokens whose KV is already computed (for chunked prefill)
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.resumable = resumable

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def resume(self, new_token_ids: list[int], sampling_params: SamplingParams):
        """Resume this sequence for the next round of multi-round conversation.
        Merge previous output into prompt, append new tokens, keep block_table."""
        assert self.status == SequenceStatus.WAITING_FOR_NEXT_ROUND
        # Previous output becomes part of prompt
        self.num_prompt_tokens = self.num_tokens
        # Append new user tokens
        self.token_ids.extend(new_token_ids)
        self.num_tokens += len(new_token_ids)
        self.last_token = self.token_ids[-1]
        # Block-align cached tokens: only full blocks count as cached
        # The partial last block will be re-computed (slot_mapping handles it)
        self.num_cached_tokens = (self.num_prompt_tokens // self.block_size) * self.block_size
        # Update sampling params for this round
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.status = SequenceStatus.WAITING

    @classmethod
    def from_context(cls, context_token_ids: list[int], suffix_token_ids: list[int],
                     block_table: list[int], num_context_tokens: int,
                     sampling_params: SamplingParams = SamplingParams(),
                     resumable: bool = False):
        """Create a sequence forked from a cached context."""
        token_ids = context_token_ids + suffix_token_ids
        seq = cls(token_ids, sampling_params, resumable=resumable)
        seq.block_table = block_table
        seq.num_cached_tokens = (num_context_tokens // cls.block_size) * cls.block_size
        return seq

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.num_computed_tokens, getattr(self, '_chunk_start', 0),
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        if len(state) == 7:
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table, self.num_computed_tokens, self._chunk_start = state[:-1]
        else:
            # backward compat with old 5-tuple format
            self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:4]
            self.num_computed_tokens = self.num_cached_tokens
            self._chunk_start = self.num_cached_tokens
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
