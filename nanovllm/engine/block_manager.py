from collections import deque
from dataclasses import dataclass, field
from itertools import count
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


@dataclass
class ContextEntry:
    block_table: list[int]
    num_tokens: int
    token_ids: list[int]


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.context_pool: dict[int, ContextEntry] = {}
        self._context_counter = count()
        self._pending_copies: list[tuple[int, int]] = []

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_allocate_incremental(self, seq: Sequence) -> bool:
        """Check if we can allocate blocks for new tokens in a resumed sequence."""
        existing_blocks = len(seq.block_table)
        needed_blocks = seq.num_blocks - existing_blocks
        return len(self.free_block_ids) >= needed_blocks

    def allocate_incremental(self, seq: Sequence):
        """Allocate only the new blocks needed for a resumed sequence.
        Existing blocks in block_table are kept as-is."""
        existing_blocks = len(seq.block_table)

        # Handle last existing block: if it was partial and now full, update its hash
        if existing_blocks > 0:
            last_block_id = seq.block_table[-1]
            last_block = self.blocks[last_block_id]
            if last_block.hash == -1:
                old_last_tokens = seq.block(existing_blocks - 1)
                if len(old_last_tokens) == self.block_size:
                    prefix = self.blocks[seq.block_table[-2]].hash if existing_blocks > 1 else -1
                    h = self.compute_hash(old_last_tokens, prefix)
                    last_block.update(h, old_last_tokens)
                    self.hash_to_block_id[h] = last_block_id

        # Start hash chain from last existing block
        h = self.blocks[seq.block_table[-1]].hash if existing_blocks > 0 else -1

        for i in range(existing_blocks, seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # Try prefix cache hit
            block_id = self.hash_to_block_id.get(h, -1) if h != -1 else -1
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    self.blocks[block_id].ref_count += 1
                else:
                    self._allocate_block(block_id)
            else:
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
            block = self.blocks[block_id]
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    # ---- Context Pool (KV Fork) ----

    def cache_context(self, seq: Sequence) -> int:
        """Store a prefilled context's block_table for future forking.
        Takes ownership of seq's blocks (don't deallocate seq after this)."""
        context_id = next(self._context_counter)
        self.context_pool[context_id] = ContextEntry(
            block_table=list(seq.block_table),
            num_tokens=seq.num_tokens,
            token_ids=list(seq.token_ids),
        )
        # Blocks already have ref_count=1 from allocate(). No need to increment.
        return context_id

    def fork_context(self, context_id: int) -> tuple[list[int], int]:
        """Fork a context: return (block_table copy, num_context_tokens).
        CoW for partial last block. Appends pending GPU copies to _pending_copies."""
        ctx = self.context_pool[context_id]
        block_table = list(ctx.block_table)

        for block_id in block_table:
            self.blocks[block_id].ref_count += 1

        # CoW for last partial block (can't share writable block)
        last_block = self.blocks[block_table[-1]]
        if last_block.hash == -1 and len(block_table) > 0:
            new_id = self.free_block_ids[0]
            self._allocate_block(new_id)
            self._pending_copies.append((last_block.block_id, new_id))
            last_block.ref_count -= 1
            block_table[-1] = new_id

        return block_table, ctx.num_tokens

    def release_context(self, context_id: int):
        """Release a cached context's blocks."""
        ctx = self.context_pool.pop(context_id, None)
        if ctx:
            for block_id in ctx.block_table:
                self.blocks[block_id].ref_count -= 1
                if self.blocks[block_id].ref_count == 0:
                    self._deallocate_block(block_id)

    def get_pending_copies(self) -> list[tuple[int, int]]:
        """Return and clear pending GPU block copies."""
        copies = self._pending_copies
        self._pending_copies = []
        return copies
