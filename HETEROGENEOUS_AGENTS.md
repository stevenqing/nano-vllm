# Heterogeneous Multi-Agent KV Cache Sharing: 研究方向

## 问题定义

在 heterogeneous multi-agent 场景中，多个 Agent 使用**同一个 LLM**但有**不同的 system prompt**，处理**共享的上下文**：

```
Agent A (Planner):  [System_A (S_a tokens)] + [shared_context (C tokens)] + [task (T tokens)]
Agent B (Coder):    [System_B (S_b tokens)] + [shared_context (C tokens)] + [plan_from_A]
Agent C (Reviewer): [System_C (S_c tokens)] + [shared_context (C tokens)] + [code_from_B]
```

**现状问题**：当前的 prefix caching（无论 vLLM 还是 nano-vllm）基于 token-exact hash 匹配。因为 `System_X` 不同，**hash 从第一个 block 就 miss**，`shared_context` 的 KV cache 完全无法跨 Agent 复用。

**直觉**：虽然 system prompt 不同，但到了 `shared_context` 的后半段，不同 Agent 的 KV cache（hidden states）可能已经高度相似——因为 Transformer 的自注意力会让远距离的前缀影响逐层衰减。

---

## 方案 0: Prompt Reordering（零代码改动）

### 核心洞察

**问题的根源不在引擎，在 prompt 的排列顺序**。

当前所有 multi-agent 框架默认的 prompt 格式：

```
[System_A] + [Context] + [Task]
```

System prompt 在最前面，导致不同 Agent 的 Context 部分 hash 全部 miss。但如果我们**把 Context 提到前面**：

```
[Context] + [System_A] + [Task]
```

那么 `[Context]` 对所有 Agent 来说是**完全相同的前缀**！现有的 prefix caching **不需要任何代码改动**就能自动复用 Context 的 KV blocks。

### 可行性分析

在 causal attention 中，排列顺序决定了 "谁能看到谁"：
- `[System] + [Context]`：System 看不到 Context（但 Context 能看到 System）
- `[Context] + [System]`：Context 看不到 System（但 System 能看到 Context）

**关键问题**：Context 在编码时看不到 System prompt，是否影响质量？

**答案**：影响很小，原因：
1. **RAG 系统的标准做法**就是 `[Document] + [Instruction] + [Query]` — 文档在前、指令在后
2. Context 的编码主要是"理解内容本身"，不需要 System 的角色定义来指导
3. System prompt 的影响主要体现在**生成**阶段（decode），而非 Context 的**编码**阶段（prefill）
4. 大量实践（Anthropic Claude, GPT-4 的上下文注入）证明这个顺序有效

### 局限

- **不适用于所有场景**：某些任务中 System prompt 会影响 Context 的"理解方式"（如不同语言、不同视角）
- **需要上层框架配合**：修改 chat template 或 prompt 构建逻辑
- **非通用方案**：只是 workaround，不解决根本的引擎层问题

---

## 方案 1: Context Pool（KV Fork）— 推荐方案

### 核心思路

类比 Unix `fork()` — 多个 Agent 从同一个 Context KV "分叉"出去，每个加自己的后缀。

```
               ┌─→ [System_A] + [Task_A] → Output_A
[Context KV] ──┼─→ [System_B] + [Task_B] → Output_B
               └─→ [System_C] + [Task_C] → Output_C
```

Context 的 KV blocks **物理上只存一份**，通过 `ref_count` 被多个 Sequence 共享。

### API 设计

```python
from nanovllm import LLM, SamplingParams

llm = LLM(model_path)

# Step 1: 预计算共享 Context 的 KV，得到 context_id
context_id = llm.cache_context(shared_context_tokens)

# Step 2: 各 Agent 引用 context_id，只需 prefill 自己的 System+Task
sp = SamplingParams(temperature=0.6, max_tokens=256)
result_a = llm.generate_with_context(system_a + task_a, sp, context_id=context_id)
result_b = llm.generate_with_context(system_b + task_b, sp, context_id=context_id)
result_c = llm.generate_with_context(system_c + task_c, sp, context_id=context_id)

# Step 3: 用完释放
llm.release_context(context_id)
```

### 内部数据流

```
cache_context(tokens):
  1. 创建一个临时 Sequence，token_ids = tokens
  2. Prefill: 计算 Context 的完整 KV
  3. 将 block_table 存入 context_pool[context_id]
  4. 返回 context_id（不生成任何 output）

generate_with_context(suffix_tokens, sp, context_id):
  1. 从 context_pool[context_id] 获取 context_block_table
  2. 创建新 Sequence:
     - token_ids = context_tokens + suffix_tokens
     - block_table = copy(context_block_table)  # 共享 blocks
     - 对共享的 blocks 增加 ref_count
     - num_cached_tokens = len(context_tokens)  # 全部已缓存
  3. 只需 prefill suffix_tokens（System+Task 部分）
  4. Decode 正常生成
```

### 技术细节

#### 问题 1: Attention 的正确性

Agent A 的完整序列是 `[Context] + [System_A] + [Task_A]`，在 causal attention 中：
- `System_A` 和 `Task_A` 的 Query 需要 attend to `Context` 的 Key/Value
- 但 `Context` 的 KV 是在**没有 System_A** 的情况下计算的

**这和方案 0（Prompt Reordering）是等价的**！Context 在前，System 在后。

但有一个关键区别：在 Prompt Reordering 中，positions 是连续的（Context: 0..C-1, System: C..C+S-1）。在 Context Pool 中，我们也需要保证 position encoding 的正确性：

```
Context KV: positions 0, 1, ..., C-1 (已计算好的)
Agent A 的 suffix: positions C, C+1, ..., C+S_a+T_a-1 (需要新计算)
```

**Position 天然正确**！因为 suffix 的 position 就应该从 C 开始。

#### 问题 2: Block Table 的共享

```
context_block_table = [B0, B1, B2, B3]  # 4 blocks for Context
Agent A's block_table = [B0, B1, B2, B3, B4, B5]  # + 2 blocks for System_A+Task_A
Agent B's block_table = [B0, B1, B2, B3, B6, B7]  # + 2 blocks for System_B+Task_B
                        ^^^^^^^^^^^^^^^^^^^^
                        共享，ref_count=3 (context + A + B)
```

当 Agent A 完成后 deallocate，B0-B3 的 ref_count 从 3 降到 2（Context + B），不会被释放。

#### 问题 3: Context 的最后一个 Block 可能是 Partial

如果 Context 长度不是 block_size 的整数倍，最后一个 block 是 partial。Agent A 的 suffix 需要填入这个 partial block 的剩余空间。

```
Context: 300 tokens → Block 0 (256 full) + Block 1 (44 tokens, partial)
Agent A suffix: 100 tokens → Block 1 gets 44+100=144 tokens? 

不行！Block 1 的 KV 已经被 Context 和其他 Agent 共享，不能写入 Agent A 的数据。
```

**解决方案：Copy-on-Write (CoW)**

当 Agent A 需要写入 Context 的最后一个 partial block 时：
1. 复制该 block 到新 block（只有最后一个 partial block 需要 CoW，$O(block\_size)$）
2. Agent A 的 block_table 中最后一个 context block 指向新 block
3. Context 的 block_table 不变
4. 新 block 中写入 Agent A 的 suffix tokens

```python
def fork_context(self, context_id, suffix_tokens):
    ctx = self.context_pool[context_id]
    block_table = list(ctx.block_table)  # shallow copy block IDs
    
    # 增加所有共享 block 的 ref_count
    for block_id in block_table:
        self.blocks[block_id].ref_count += 1
    
    # CoW: 最后一个 partial block
    last_block = self.blocks[block_table[-1]]
    if last_block.hash == -1:  # partial block
        # 复制到新 block
        new_block_id = self.free_block_ids[0]
        new_block = self._allocate_block(new_block_id)
        # 复制 KV cache 数据 (GPU memcpy)
        self._copy_block_kv(src=last_block.block_id, dst=new_block_id)
        # 减少原 block 的 ref_count
        last_block.ref_count -= 1
        # 替换 block_table 中最后一个
        block_table[-1] = new_block_id
    
    return block_table
```

### 实现计划

需要修改的文件：

#### 1. `nanovllm/engine/block_manager.py` — 新增 Context Pool

```python
class BlockManager:
    def __init__(self, ...):
        ...
        self.context_pool: dict[int, ContextEntry] = {}
        self._context_counter = itertools.count()
    
    def cache_context(self, seq: Sequence) -> int:
        """Store a context's block_table for future forking."""
        context_id = next(self._context_counter)
        self.context_pool[context_id] = ContextEntry(
            block_table=list(seq.block_table),
            num_tokens=seq.num_tokens,
            token_ids=list(seq.token_ids),
        )
        # 增加 ref_count 防止被回收
        for block_id in seq.block_table:
            self.blocks[block_id].ref_count += 1
        return context_id
    
    def fork_context(self, context_id: int) -> tuple[list[int], int]:
        """Fork a context: return (block_table copy, num_context_tokens).
        Handles CoW for partial last block."""
        ctx = self.context_pool[context_id]
        block_table = list(ctx.block_table)
        
        for block_id in block_table:
            self.blocks[block_id].ref_count += 1
        
        # CoW for last partial block
        last_block = self.blocks[block_table[-1]]
        if last_block.hash == -1:  # partial
            new_id = self.free_block_ids[0]
            self._allocate_block(new_id)
            self._copy_block_kv(last_block.block_id, new_id)
            last_block.ref_count -= 1
            block_table[-1] = new_id
        
        return block_table, ctx.num_tokens
    
    def release_context(self, context_id: int):
        """Release a cached context."""
        ctx = self.context_pool.pop(context_id, None)
        if ctx:
            for block_id in ctx.block_table:
                self.blocks[block_id].ref_count -= 1
                if self.blocks[block_id].ref_count == 0:
                    self._deallocate_block(block_id)
    
    def _copy_block_kv(self, src_id: int, dst_id: int):
        """Copy KV cache data between blocks (GPU memcpy)."""
        # 需要在 model_runner 中实现实际的 GPU 拷贝
        self._pending_copies.append((src_id, dst_id))
```

#### 2. `nanovllm/engine/sequence.py` — 支持从 Context 分叉

```python
class Sequence:
    @classmethod
    def from_context(cls, context_token_ids, suffix_token_ids, 
                     block_table, num_context_tokens, sampling_params):
        """Create a sequence forked from a cached context."""
        token_ids = context_token_ids + suffix_token_ids
        seq = cls(token_ids, sampling_params)
        seq.block_table = block_table
        seq.num_cached_tokens = (num_context_tokens // cls.block_size) * cls.block_size
        return seq
```

#### 3. `nanovllm/engine/scheduler.py` — 支持 context forked sequences

调度器的 `schedule()` 已有 `if seq.block_table:` 分支走 `allocate_incremental()`。
Context-forked 的 Sequence 天然有 `block_table`，无需改调度逻辑。

#### 4. `nanovllm/engine/llm_engine.py` — 新增 API

```python
class LLMEngine:
    def cache_context(self, context: str | list[int]) -> int:
        """Prefill and cache a shared context. Returns context_id."""
        if isinstance(context, str):
            context = self.tokenizer.encode(context)
        # 创建临时 sequence，prefill 但不 decode
        seq = Sequence(context, SamplingParams(max_tokens=0))
        self.scheduler.add(seq)
        # 执行 prefill
        while not seq.status == SequenceStatus.RUNNING:
            self.step()
        # 缓存 block_table
        context_id = self.scheduler.block_manager.cache_context(seq)
        # 从 running 中移除（不需要 decode）
        self.scheduler.running.remove(seq)
        return context_id
    
    def generate_with_context(self, suffix, sampling_params, context_id):
        """Generate with a pre-cached context prefix."""
        if isinstance(suffix, str):
            suffix = self.tokenizer.encode(suffix)
        ctx = self.scheduler.block_manager.context_pool[context_id]
        block_table, num_ctx_tokens = self.scheduler.block_manager.fork_context(context_id)
        seq = Sequence.from_context(
            ctx.token_ids, suffix, block_table, num_ctx_tokens, sampling_params
        )
        self.scheduler.add(seq)
        # 跑到完成
        while not self.is_finished():
            output, _ = self.step()
            for sid, token_ids in output:
                if sid == seq.seq_id:
                    return {"text": self.tokenizer.decode(token_ids), 
                            "token_ids": token_ids, "seq_id": seq.seq_id}
    
    def release_context(self, context_id: int):
        self.scheduler.block_manager.release_context(context_id)
```

#### 5. `nanovllm/engine/model_runner.py` — GPU block copy

```python
class ModelRunner:
    def copy_block_kv(self, src_id: int, dst_id: int):
        """Copy KV cache data from src block to dst block."""
        # kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache[:, :, dst_id] = self.kv_cache[:, :, src_id]
```

### 性能分析

| 场景 | 无优化 (baseline) | Context Pool | 加速比 |
|------|-------------------|-------------|--------|
| 3 Agent, Context=1000tok, System=200tok | 3 × prefill(1200) = 3600 | 1 × prefill(1000) + 3 × prefill(200) = 1600 | **2.25x** |
| 5 Agent, Context=2000tok, System=100tok | 5 × prefill(2100) = 10500 | 1 × prefill(2000) + 5 × prefill(100) = 2500 | **4.2x** |
| 10 Agent, Context=4000tok, System=50tok | 10 × prefill(4050) = 40500 | 1 × prefill(4000) + 10 × prefill(50) = 4500 | **9.0x** |

**Agent 越多、Context 越长、System 越短，加速比越大。**

### 与 Prompt Reordering 的对比

| 方面 | Prompt Reordering | Context Pool |
|------|-------------------|-------------|
| 代码改动 | 零 (prompt 层面) | 引擎层面 (~100 行) |
| 精确性 | 等价（Context 在前） | 等价（Context 在前） |
| 通用性 | 需要上层改 prompt template | 引擎透明支持 |
| 显存 | Context KV 仍被 prefix cache 共享 | Context KV 显式共享（ref_count） |
| 多轮 | 需要和 multi-round 分开处理 | 可与 multi-round `chat()` 组合 |

---

## 方案 2: Agent-Aware Inference Scheduler（核心创新）

### 动机：Multi-Agent 不是一堆独立请求

当前引擎（包括 vLLM）把每个请求当作**独立的、无关联的**。但 multi-agent 工作流是一个**有向无环图（DAG）**：

```
                    ┌─→ Agent B (Coder) ──→ Agent C (Reviewer)
Context(共享) ──→ Agent A (Planner) ─┤                         │
                    └─→ Agent D (Tester)        ↓ (reject)
                                          Agent B' (Fix) ──→ Agent C' (Re-review)
```

如果调度器**理解这个 DAG**，可以做到三件引擎级别不可能做到的事：

### 优化 1: Speculative Prefill（提前计算已知部分）

Agent B 的输入 = `[Context] + [System_B] + [Output_A]`

其中 `[Context]` 和 `[System_B]` 在 **A 还没开始生成**时就已知。只有 `[Output_A]` 需要等 A 完成。

```
传统 (串行等待):
  时间线: ───[A prefill]──[A decode ×64]──────[B prefill (全部)]──[B decode ×200]──
                                        等待A
                                        完成
  总时间: T_A_prefill + T_A_decode + T_B_prefill + T_B_decode

DAG-aware (Speculative Prefill):
  时间线: ───[A prefill]──[A decode ⊕ B_partial_prefill]──[B append Output_A]──[B decode ×200]──
              │           │← B 的 Context+System_B 趁 A decode 空闲时计算 →│
              │           │  (chunked prefill 与 A decode 交错执行)        │
              │                                                           │
              │← 计算量: Context + System_B 的 prefill 被完全掩盖 →│       │
                                                                         │
  B 需要额外 prefill 的只有 Output_A 的 64 tokens ──────────────────────→│
  
  节省: T_B_prefill - T_append_output_A ≈ 减少 80-95% 的 B prefill 时间
```

**为什么可行**：
- A 的 decode 阶段是 **memory-bandwidth bound**（GPU 计算空闲 ~95%）
- B 的部分 prefill 是 **compute-bound**（利用空闲的计算单元）
- Chunked prefill 把 B 的长 prefill 拆成小块，插入 A 的 decode 步之间
- 不影响 A 的 decode 速度（它们用的是 GPU 的不同资源）

### 优化 2: 依赖图驱动的 KV 生命周期管理

当前的 KV cache 释放策略：请求完成 → 立即释放（或 prefix cache 被动保留）。

DAG-aware 策略：

```python
# 调度器知道 Agent A 的 Output 将被 Agent B 和 Agent D 使用
# A 完成时：不释放 KV，标记为 "downstream pending"
# B 和 D 都用完 A 的 KV 后：才释放

agent_a.downstream = [agent_b, agent_d]  # DAG 信息
# 当 A 完成：
#   KV ref_count += len(downstream)
# 当 B 使用完：
#   KV ref_count -= 1
# 当 D 使用完：
#   KV ref_count -= 1  → ref_count == 0 → 释放
```

这避免了"A 完成后释放 KV → B 重新 prefill A 的 output"的浪费。本质上是 Context Pool 的自动化版本——不需要用户手动 `cache_context()` / `release_context()`，由 DAG 自动管理。

### 优化 3: Agent 异构特征感知调度

不同 Agent 有不同的**计算特征**：

| Agent 类型 | Prefill 特征 | Decode 特征 | 调度策略 |
|-----------|-------------|------------|---------|
| Planner | 长 context → 重 prefill | 短输出 (~50 tok) → 轻 decode | 优先调度，快速释放 KV |
| Coder | 中等 prefill | 长输出 (~500 tok) → 重 decode | 长时间占 KV，尽早开始 |
| Reviewer | 长 context → 重 prefill | 短输出 (~20 tok) → 极轻 decode | 可以和 Coder decode 重叠 |
| Tool Caller | 极短 | 极短，但有外部等待 | 发出 tool call 后挂起，释放 GPU |

**调度含义**：

```
传统 FCFS:
  [Planner prefill][Planner decode×50][Coder prefill][Coder decode×500][Reviewer prefill][Reviewer decode×20]
  
异构感知:
  [Planner pf][Plan decode×50 ⊕ Coder partial pf][Coder pf_rest][Coder dec×500 ⊕ Reviewer partial pf][Rev dec×20]
               ↑ 利用 Planner 短 decode 的空隙           ↑ 利用 Coder 长 decode 的空隙
               │ 提前准备 Coder                           │ 提前准备 Reviewer
```

### 完整系统设计

#### API

```python
from nanovllm import LLM, SamplingParams, AgentDAG

llm = LLM(model_path)

# 定义 Agent 工作流 DAG
dag = AgentDAG()
planner = dag.add_agent("planner", system_prompt=system_a)
coder = dag.add_agent("coder", system_prompt=system_b, depends_on=[planner])
reviewer = dag.add_agent("reviewer", system_prompt=system_c, depends_on=[coder])

# 提交共享 context + 启动 DAG
results = llm.run_agents(
    dag,
    context=shared_context,
    task="Build a REST API for user management",
    sampling_params=SamplingParams(temperature=0.6, max_tokens=512),
)
# results = {"planner": {...}, "coder": {...}, "reviewer": {...}}
```

#### 内部架构

```
┌───────────────────────────────────────────────┐
│                 AgentDAG                       │
│   planner ──→ coder ──→ reviewer              │
└────────────────────┬──────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────┐
│          DAG-Aware Scheduler                   │
│                                               │
│  1. Context Pool: 预计算共享 context KV         │
│  2. Ready Queue: 依赖已满足的 agent             │
│  3. Speculative Prefill: 提前算已知部分          │
│  4. KV 生命周期: ref_count by downstream        │
│  5. 异构感知: 根据 agent 特征调度 prefill/decode │
│                                               │
│  调度循环:                                     │
│    while dag not complete:                    │
│      // 检查哪些 agent 的依赖已满足             │
│      ready_agents = dag.get_ready()           │
│      // 提交 ready agents（自动 batch）         │
│      for agent in ready_agents:               │
│        fork context KV + append upstream output│
│        add to scheduler waiting queue          │
│      // 执行 prefill + decode 调度             │
│      step()  // 已有的 prefill/decode 循环      │
│      // 检查完成的 agent                       │
│      for finished in get_finished():          │
│        dag.mark_done(finished)                │
│        // 触发下游 agent                       │
│        for downstream in finished.dependents: │
│          prepare_speculative_prefill(downstream)│
└───────────────────────────────────────────────┘
```

#### 核心数据结构

```python
@dataclass
class AgentNode:
    name: str
    system_prompt: list[int]
    depends_on: list[str]           # 上游 agent 名
    dependents: list[str]           # 下游 agent 名
    status: AgentStatus             # PENDING / PREFILLING / DECODING / DONE
    seq_id: int | None = None       # 关联的 Sequence
    output_token_ids: list[int] | None = None

class AgentDAG:
    def __init__(self):
        self.agents: dict[str, AgentNode] = {}
        self.context_id: int | None = None  # 关联的 Context Pool 条目
    
    def add_agent(self, name, system_prompt, depends_on=None):
        node = AgentNode(name=name, system_prompt=system_prompt,
                        depends_on=depends_on or [], dependents=[])
        for dep in node.depends_on:
            self.agents[dep].dependents.append(name)
        self.agents[name] = node
        return node
    
    def get_ready(self) -> list[AgentNode]:
        """返回所有依赖已满足的 agent"""
        return [a for a in self.agents.values()
                if a.status == AgentStatus.PENDING
                and all(self.agents[d].status == AgentStatus.DONE 
                       for d in a.depends_on)]
    
    def is_complete(self) -> bool:
        return all(a.status == AgentStatus.DONE for a in self.agents.values())
```

#### 与已有系统的集成

| 组件 | 改动 | 说明 |
|------|------|------|
| `Sequence` | 无 | 已有 `from_context()` + `resume()` 足够 |
| `BlockManager` | 小改 | Context Pool 已设计好；增加 DAG-aware ref_count |
| `Scheduler` | 主要改动 | 新增 `schedule_dag()` 循环，嵌套调用现有 `schedule()` |
| `ModelRunner` | 无 | `prepare_prefill` / `prepare_decode` 不需要改 |
| `LLMEngine` | 新增 | `run_agents()` API 封装 DAG 调度循环 |

**关键点**：DAG Scheduler 不替代现有调度器，而是**嵌套在上层**——它管理 agent 间的依赖和触发，agent 内的 prefill/decode 仍走原有调度器。

### 预期收益计算

以 3-Agent (Planner→Coder→Reviewer) 为例：

```
参数:
  Context = 1000 tokens
  System = 200 tokens (每个 agent)
  Planner output = 64 tokens
  Coder output = 256 tokens
  Reviewer output = 32 tokens

Baseline (串行):
  Prefill: 3 × (1000 + 200) = 3600 tokens
  Decode:  64 + 256 + 32 = 352 steps (BS=1 each)
  
Context Pool only:
  Prefill: 1×1000 + 3×200 = 1600 tokens           (节省 56% prefill)
  Decode:  64 + 256 + 32 = 352 steps (BS=1 each)
  
Context Pool + Speculative Prefill:
  Prefill: 1×1000 + 200(planner) = 1200 visible    (coder/reviewer 被 decode 掩盖)
         + 64(append output_A) + 32(append output_B) = 1296 visible tokens
  Decode:  64 + 256 + 32 = 352 steps
  但 coder decode 期间 reviewer 可以 speculative prefill → 进一步重叠
  
Context Pool + Spec Prefill + Batched Decode:
  当多个 agent decode 阶段重叠时，batch size > 1 → GPU 利用率更高
  如果 coder 和 reviewer 同时 decode: BS=2, throughput ~1.8x 单请求
  
理论最大加速: ~3-4x vs baseline (取决于具体时间分布)
```

---

## 方向 2: Logits-Guided KV Sharing（研究向，参考）

### 核心思路

用 logits（或 hidden states）的相似度来判断两个 Agent 在某个 position 的 KV cache 是否可以互用。

### 流程

```
Step 1: Agent A 完整推理 shared_context，缓存每个 block 的 KV + logits checkpoint
Step 2: Agent B 开始推理时，对 shared_context 的每个 block：
  a) 用 Agent B 的完整前缀（System_B + 前面的 blocks）计算该 block 的 logits
  b) 与 Agent A 缓存的 logits 做 similarity（cosine / KL-div）
  c) 如果 sim > threshold → 后续 blocks 直接复用 Agent A 的 KV
  d) 如果 sim < threshold → 该 block 及后续需要重新计算
```

### 开销分析

设：
- $L$ = 模型层数
- $H$ = hidden size
- $V$ = vocabulary size
- $B$ = block size (256 tokens)
- $N_{shared}$ = shared_context 中的 block 数 = $\lceil C / B \rceil$
- $S$ = system prompt 长度

#### Case 1: 无共享（Baseline）

Agent B 需要完整 prefill：$S_b + C + T$ tokens

**计算量**: 每个 token 经过 $L$ 层 Transformer，每层 ≈ $12 H^2$ FLOPs (QKV proj + attn + FFN)

$$\text{Cost}_{\text{baseline}} = (S_b + C + T) \times L \times 12H^2$$

#### Case 2: Logits-Guided Sharing

**验证开销**：找到 diverge point 需要逐 block 计算 logits

最好情况（Block 0 就相似）：
- 只需计算 Block 0 的 logits = $B$ 个 token 的前向 = $B \times L \times 12H^2$
- 然后直接复用 Agent A 剩余的 $N_{shared} - 1$ 个 blocks
- **节省**: $(C - B) \times L \times 12H^2$

最坏情况（全部不相似）：
- 逐 block 检查 $N_{shared}$ 个 blocks，每个计算 $B$ token
- 总计: $C \times L \times 12H^2$（等于完整 prefill）
- 额外浪费了比较 logits 的开销

**关键问题**：验证一个 block 的 logits 需要 **完整前向计算该 block 的 hidden states**，这和直接 prefill 该 block 的开销一样！

#### 结论

$$\text{Cost}_{\text{verify}} \approx \text{Cost}_{\text{compute\_KV}}$$

**验证的 FLOPs 等于直接计算 KV 的 FLOPs**。Logits-Guided Sharing 在计算量上没有优势——因为要判断 KV 是否可复用，必须先把 KV 算出来（或等价地算 logits）。

### 但是——有优化空间

#### 优化 1: 采样验证（Amortized Verification）

不是每个 token 都验证，而是每隔 $k$ 个 block 验证一次：

$$\text{Cost}_{\text{verify}} = \frac{C}{k} \times L \times 12H^2$$

如果验证间隔 $k$ 足够大，验证开销可忽略。代价是精度下降。

#### 优化 2: 浅层验证（Layer-wise Shortcut）

不需要完整 $L$ 层前向来比较 logits。可以只比较**前几层的 hidden states**：

- 如果前 $l$ 层（$l \ll L$）的 hidden states 就已经相似，那后续层大概率也相似
- **验证开销**: $B \times l \times 12H^2$（仅前 $l$ 层）
- 但 KV cache 需要**全部 $L$ 层都匹配**才能复用

#### 优化 3: Offline Profiling（最实用）

对特定的 agent 组合做**离线分析**：

```python
# 离线：给定 Agent A, B, C 的 system prompts
# 用一批典型 shared_context 做 rollout
# 统计每个 position 的 KV similarity 分布
# 确定 "safe reuse point"：从哪个 position 开始可以安全复用

for position in range(max_len):
    kv_a = model.forward(System_A + context[:position])
    kv_b = model.forward(System_B + context[:position])
    similarity[position] = cosine_sim(kv_a, kv_b)

safe_point = first position where similarity > threshold for all subsequent positions
```

然后在线推理时：
- Position < safe_point → 各自独立计算 KV
- Position >= safe_point → 直接复用 Agent A 的 KV

**这本质上是 Speculative KV Reuse（方向 2）的离线版本。**

---

## 方向 3: Speculative KV Reuse（研究向）

### 核心思路

类比 Speculative Decoding：先"投机"地复用 Agent A 的 KV 给 Agent B，然后通过验证决定 accept/reject。

### 流程

```
Step 1: Agent A 推理完成，KV cache 保留
Step 2: Agent B 开始推理：
  a) 直接用 Agent A 的 KV blocks 作为 Agent B 的 cached KV
  b) 只计算 System_B 的 KV（独立前缀部分）
  c) Agent B 的 decode 阶段使用混合 KV：
     - [System_B 自己的 KV] + [Agent A 的 shared_context KV]
  d) 比较 Agent B 在关键 position 的 logits 与 ground truth
  e) 如果差异 < ε → Accept，无需重算
  f) 如果差异 > ε → Reject，从 diverge point 重算
```

### 开销分析

**Accept 路径**: 
$$\text{Cost} = S_b \times L \times 12H^2 + \text{verification}$$

只需 prefill System_B（$S_b$ tokens），shared_context 完全免费！

**Reject 路径**:
$$\text{Cost} = S_b \times L \times 12H^2 + \text{verification} + C \times L \times 12H^2$$

等于完整 prefill + 额外验证开销。

**关键指标**: Accept Rate。如果 accept rate > 50%，平均下来就有收益。

### Verification 方法

Option A: **终端验证** — 只在 shared_context 最后一个 token 比较 logits  
- 开销: 1 个 token 的前向  
- 风险: miss 中间 diverge

Option B: **checkpoint 验证** — 在 shared_context 中均匀选几个 checkpoint position  
- 开销: $k$ 个 token 的前向  
- 更可靠

Option C: **Output quality 验证** — 不验证 KV 本身，而是看最终输出是否合理  
- 零额外推理开销  
- 需要 output quality metric

---

## 方向 4: Similarity-Aware Block Grouping（研究向）

### 核心思路

离线建立 agent 之间的 KV 相似度图谱，在线推理时直接用预计算的 sharing policy。

### 具体方案

```python
# 离线 Profiling
agents = [Agent_A, Agent_B, Agent_C]
sample_contexts = load_representative_contexts(n=100)

sharing_policy = {}
for a_i, a_j in combinations(agents, 2):
    diverge_points = []
    for context in sample_contexts:
        kv_i = compute_kv(a_i.system_prompt + context)
        kv_j = compute_kv(a_j.system_prompt + context)
        # 逐层逐 position 比较 KV 相似度
        for pos in range(len(context)):
            sim = layerwise_kv_similarity(kv_i, kv_j, pos)
            if sim < threshold:
                diverge_points.append(pos)
                break
    # 取保守值
    safe_reuse_offset = max(diverge_points) + margin
    sharing_policy[(a_i, a_j)] = safe_reuse_offset

# 在线推理
def schedule_agent_b(agent_b, shared_context, agent_a_kv_cache):
    offset = sharing_policy[(agent_a, agent_b)]
    # Position < offset: Agent B 独立计算
    # Position >= offset: 复用 Agent A 的 KV blocks
```

### 优势

- **在线零验证开销** — sharing policy 预计算好
- **确定性** — 不是投机，而是基于统计保证的安全复用
- 适合 **固定 agent 组合**（实际部署中 agent 配置通常不频繁变化）

### 局限

- 需要离线 profiling，不适合动态创建的 agent
- Safety margin 可能导致实际复用比例较低
- Context-dependent — 不同 context 的 diverge point 可能差异大

---

## 初步判断

| 方案 | 实现难度 | 潜在收益 | 质量损失 | 适用场景 |
|------|---------|---------|---------|---------|
| **0. Prompt Reorder** | **零** | 中 | 零 | 允许改 prompt 的场景 |
| **1. Context Pool (KV Fork)** | **低 (~100行)** | **高 (2-9x prefill)** | **零** | **通用 multi-agent** |
| **2. DAG-Aware Scheduler** | 中 (~300行) | 高 (prefill overlap) | 零 | Agent 工作流 |
| **3. Shared-Context Attention** | 高 (Triton kernel) | **极高 (decode)** | **零** | **大模型+长context** |
| 4. Prefill FFN Dedup | 中 | 低 (6% total) | 零 | 理论价值 |
| 5. Logits-Guided KV | 中 | 低-中 | 需验证 | 理论研究 |
| 6. Speculative KV Reuse | 中 | 高 | accept rate 决定 | 实际部署 |

---

## 方案 3: Shared-Context Attention（Decode 阶段核心优化）

### 问题分析：Decode 的真正瓶颈

Multi-agent 场景下，decode 占总推理时间的 ~90%（小模型短 context）到 ~99%（大模型长 context）。

Decode 每步的开销分解：

```
Weight loading:  模型参数从 HBM → SM   (bandwidth-bound, 占 90%+ 时间)
KV cache reads:  seq_len × KV_bytes     (bandwidth-bound)
Compute:         Q @ K^T + softmax @ V  (compute, 几乎免费)
```

瓶颈在 **HBM 带宽**。加速的唯一有效路径：减少每步 HBM 读取量。

### 核心洞察：共享 KV Block 被重复加载

当 N 个 agent batched decode 且共享 context KV blocks 时，Flash Attention 当前的处理：

```python
# 现有 Flash Attention: 每个 agent 独立遍历所有 KV blocks
for agent_i in batch:                     
    for block in agent_i.block_table:     # 包含共享 blocks
        K = load_from_HBM(k_cache[block]) # ← 共享 block 被加载 N 次！
        V = load_from_HBM(v_cache[block]) # ← 完全重复的带宽浪费
        partial_attn(Q[i], K, V)
```

共享 context 的 KV blocks 被**重复加载 N 次**。

### 方案：加载一次，计算 N 次

```python
# Shared-Context Attention: 共享 blocks 只加载 1 次
# Phase 1: 共享 blocks
for block in shared_blocks:
    K = load_from_HBM(k_cache[block])     # 1 次 HBM 读取
    V = load_from_HBM(v_cache[block])     # 1 次 HBM 读取
    for agent_i in agents:                # N 次计算（K, V 在寄存器/L2）
        partial_attn(Q[i], K, V, accum[i])

# Phase 2: 各 agent 独有 blocks（正常处理）
for agent_i in agents:
    for block in unique_blocks[i]:
        K, V = load(k_cache[block]), load(v_cache[block])
        partial_attn(Q[i], K, V, accum[i])
```

**1 次 HBM 加载 + N 次寄存器计算** = 完美的带宽摊销。

### 与 GQA 的类比

这个优化和 GQA (Grouped-Query Attention) 本质相同：

```
GQA:   多个 Q heads → 共享 1 个 KV head   (head 维度共享, 省 HBM)
SCA:   多个 Q agents → 共享 KV blocks      (request 维度共享, 省 HBM)
```

可以用 **同一套 kernel 逻辑** — 把 "N agents × H heads" 展平为 "N×H query groups"，共享 blocks 的 KV 只加载一次。

### 带宽节省计算

$$\text{Bandwidth\_saved} = (N-1) \times L \times B_{shared} \times \text{KV\_bytes\_per\_block}$$

其中 $L$ = 层数, $B_{shared}$ = 共享 block 数, $N$ = agent 数。

#### 小模型（Qwen3-0.6B）

```
28 layers, 2 KV heads, head_dim=64, block_size=256:
  KV/block/layer = 2 × 256 × 2 × 64 × 2 bytes = 128 KB
  Context=1000 (4 blocks), 3 agents:
  
  不优化: 3 × 28 × 4 × 128KB = 42.9 MB/step
  优化后: 1 × 28 × 4 × 128KB = 14.3 MB/step + unique
  节省:   28.6 MB/step
  vs 权重: ~1,200 MB/step → 2.4% → 效果微弱
```

#### 大模型 INT4（Llama-3-70B）

```
80 layers, 8 KV heads, head_dim=128, block_size=256:
  KV/block/layer = 2 × 256 × 8 × 128 × 2 bytes = 1 MB
  
情况 A: Context=4K (16 blocks), 10 agents:
  不优化: 10 × 80 × 16 × 1MB = 12,800 MB/step
  优化后: 1  × 80 × 16 × 1MB =  1,280 MB/step + unique
  节省:   11,520 MB/step
  vs 权重 (INT4): ~35,000 MB/step
  总读取: 47,800 → 36,280 MB/step → **节省 24%**
  
情况 B: Context=32K (128 blocks), 10 agents:
  不优化: 10 × 80 × 128 × 1MB = 102,400 MB/step
  优化后: 1  × 80 × 128 × 1MB =  10,240 MB/step
  节省:   92,160 MB/step
  vs 权重: 35,000 MB/step
  总读取: 137,400 → 45,240 MB/step → **节省 67%**

  KV 读取已超过权重！这时 SCA 比减少权重（量化）还有效。
```

#### 关键转折点

$$\text{SCA 收益显著} \iff N \times C \times \text{KV\_per\_token} > \text{Weight\_size}$$

即 **agent数 × context长度 × 每token KV大小 > 模型权重大小**。

| 模型 / 条件 | 转折点 (N×C) | 示例 |
|-------------|-------------|------|
| Qwen3-0.6B (bf16) | N×C > 84K | 10 agents × 8K context |
| Llama-3-8B (INT4) | N×C > 34K | 5 agents × 7K context |
| Llama-3-70B (INT4) | N×C > 15K | 3 agents × 5K context |

**模型越大（权重越小/量化后），SCA 越早产生收益。**

### 实现路径

#### Level 1: Python 层优化（简单但有效）

不改 attention kernel，而是在 `prepare_decode` 时**重排 batch**，让共享 blocks 的 agent 在 batch 中相邻 → 利用 L2 cache 的 temporal locality。

```python
# model_runner.py — prepare_decode() 中
# 按 shared context group 排序 batch
seqs.sort(key=lambda s: tuple(s.block_table[:shared_prefix_len]))
# → 共享 blocks 的 agent 连续处理 → L2 cache 自然命中
```

**零 kernel 改动，可能获得 L2 级别的部分收益。**

#### Level 2: Triton Kernel（完整优化）

```python
@triton.jit
def shared_context_decode_attention(
    Q,                      # [N_agents, num_heads, head_dim]  
    K_cache, V_cache,       # [num_blocks, block_size, num_kv_heads, head_dim]
    shared_block_ids,       # [num_shared_blocks]
    unique_block_tables,    # [N_agents, max_unique_blocks]
    output,                 # [N_agents, num_heads, head_dim]
    ...
):
    agent_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    q = tl.load(Q + agent_id * stride_q + head_id * head_dim + offsets)
    
    # Online softmax accumulators
    m_prev = -float('inf')
    l_prev = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    kv_head_id = head_id // (num_heads // num_kv_heads)  # GQA mapping
    
    # Phase 1: Shared blocks — K/V loaded once, computed for all agents
    # (Triton L2 cache handles this: blocks stay in L2 across agent_ids)
    for i in range(num_shared_blocks):
        block_id = tl.load(shared_block_ids + i)
        for j in range(block_size):
            k = tl.load(K_cache + block_id * block_stride + j * kv_stride 
                       + kv_head_id * head_dim + offsets)
            v = tl.load(V_cache + block_id * block_stride + j * kv_stride 
                       + kv_head_id * head_dim + offsets)
            score = tl.sum(q * k) * scale
            # Online softmax update
            m_new = tl.maximum(m_prev, score)
            p = tl.exp(score - m_new)
            l_new = l_prev * tl.exp(m_prev - m_new) + p
            acc = acc * (l_prev * tl.exp(m_prev - m_new) / l_new) + p / l_new * v
            m_prev, l_prev = m_new, l_new
    
    # Phase 2: Unique blocks per agent
    for i in range(max_unique_blocks):
        block_id = tl.load(unique_block_tables + agent_id * max_unique_blocks + i)
        if block_id == -1:
            break
        for j in range(block_size):
            k = tl.load(K_cache + block_id * block_stride + j * kv_stride 
                       + kv_head_id * head_dim + offsets)
            v = tl.load(V_cache + block_id * block_stride + j * kv_stride 
                       + kv_head_id * head_dim + offsets)
            score = tl.sum(q * k) * scale
            m_new = tl.maximum(m_prev, score)
            p = tl.exp(score - m_new)
            l_new = l_prev * tl.exp(m_prev - m_new) + p
            acc = acc * (l_prev * tl.exp(m_prev - m_new) / l_new) + p / l_new * v
            m_prev, l_prev = m_new, l_new
    
    tl.store(output + agent_id * stride_o + head_id * head_dim + offsets, acc)
```

#### Level 3: 集成到 Flash Attention / FlashInfer

向 Flash Attention 上游提交 **shared block table** 支持：让 `flash_attn_with_kvcache` 接收一个额外参数 `shared_block_table`，在其实现中对共享 blocks 做 broadcast load。

### 与其他优化的组合

```
方案 1 (Context Pool) + 方案 3 (SCA):
  Context Pool 提供了"哪些 blocks 被共享"的信息 (ref_count > 1)
  SCA 利用这个信息在 attention kernel 中做 broadcast load
  → 端到端：prefill 省 Context × (N-1)，decode 省 KV 带宽 × (N-1)

方案 2 (DAG Scheduler) + 方案 3:
  DAG 自动管理 agent 间依赖和 KV 生命周期
  SCA 自动检测 batch 中的共享 blocks
  → 用户只需提交 agent DAG，一切优化自动发生
```

### 总结

| 维度 | 现有方案 | Shared-Context Attention |
|------|---------|------------------------|
| Decode KV 带宽 | N 份重复读取 | **1 份加载 + N 次计算** |
| 数学正确性 | - | **精确等价（零近似）** |
| 大模型+长context | KV bandwidth 是主要瓶颈 | **节省 24%-67% 带宽** |
| 小模型 | KV 占比太小 | ~2% 效果微弱 |
| 实现 | 不需要 | Triton kernel / Flash Attention patch |
| 与 GQA 关系 | head 维度共享 | **request 维度共享（同一思想）** |

**SCA 是 multi-agent inference 在 decode 阶段的本质优化。当 $N \times C$ 足够大时，它比量化带来的带宽节省还大。**

---

## 下一步：实验验证

写一个实验脚本：
1. 取 2-3 个典型 agent system prompt
2. 用同一个 shared_context
3. 分别计算各 agent 在每个 position 的 KV / logits
4. 度量 pairwise similarity（逐层、逐 position）
5. 确定是否存在 "convergence point"——从哪个 position 开始 KV 足够相似

---

## Speculative Decoding 深度分析 & 迁移思考

### Speculative Decoding 的本质

Speculative Decoding (SD) 的核心是一个**信息不对等的 trade-off**：

```
Draft Model (小/快) → 生成 K 个候选 token → Target Model (大/慢) 一次性验证 K+1 个 token
```

**为什么有效**：
- Decode 阶段是 **memory-bandwidth bound**（每步只生成 1 个 token，GPU 计算单元大量空闲）
- Target model 验证 K 个 token 的 FLOPs ≈ 生成 1 个 token（batch parallelism）
- 所以用"1 步 target"的代价可以推进 K+1 步 → 理论 K+1x 加速

**核心数学保证**（Rejection Sampling）：

$$P(\text{accept token } d_i) = \min\left(1, \frac{p_{\text{target}}(d_i)}{p_{\text{draft}}(d_i)}\right)$$

- 如果 $p_{\text{target}} \geq p_{\text{draft}}$：总是 accept
- 如果 $p_{\text{target}} < p_{\text{draft}}$：按比值概率 accept
- 被 reject 时：从**修正分布** $p'(t) = \max(0, p_{\text{target}}(t) - p_{\text{draft}}(t))$ 中采样

**数学性质**：**最终输出分布与直接用 target model 采样完全一致**。这不是近似，是精确等价。

### SD 的优势与局限

| 优势 | 局限 |
|------|------|
| **零质量损失** — 输出分布数学等价 | draft 与 target 需要高 acceptance rate |
| **利用空闲计算** — bandwidth-bound → compute-bound | prefill 阶段无收益（已经是 compute-bound） |
| **可组合** — 任意 draft（小模型/n-gram/EAGLE） | KV cache 开销增加（lookahead 预分配） |
| **不需要训练** — 即插即用 | 高 temperature 下 acceptance rate 低 |

### vLLM 中 SD 的 KV Cache 处理

```python
# 调度器过度分配：为 draft tokens 预留 slot
new_blocks = kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    num_lookahead_tokens=num_speculative_tokens  # 多分配 5-8 个 slot
)

# 验证后：rejected tokens 的 KV slot 被标记为无效
# 通过 slot_mapping 更新，rejected 位置的 KV 被后续 token 覆盖
```

vLLM 的关键设计：**KV cache 是预分配的**，rejected tokens 的 KV 并不需要"回滚"——只需修正 slot_mapping，让下一步的 token 写到正确位置。

---

### 从 SD 到 Heterogeneous Multi-Agent: 思维迁移

让我们重新审视 heterogeneous multi-agent 的问题：

```
Agent A: [System_A] + [Context] → Output_A
Agent B: [System_B] + [Context] → Output_B   (Context 与 A 相同)
```

#### 错误类比：把 Agent A 当 "Draft", Agent B 当 "Target"

这不成立，因为：
1. SD 中 draft 和 target 处理**相同的 token sequence**
2. 这里 Agent A 和 B 处理的是**不同的 prefix** (System_A ≠ System_B)
3. SD 验证的是 token-level 的概率分布，而我们要验证的是 **KV cache 的等价性**

#### 更优雅的类比：Speculative KV Prefill

**核心洞察**：在 SD 中，draft model 为 target model **预填充候选 token**，target 一次验证。类比地，Agent A 的 KV cache 为 Agent B **预填充候选 KV states**，Agent B 一次验证。

```
传统 SD:
  Draft Model → generates tokens d_1..d_K → Target Model verifies d_1..d_K in 1 step
  
Speculative KV Prefill:
  Agent A's KV → proposed KV states for positions p_1..p_N → Agent B verifies in 1 step
```

**但关键区别在于验证方式**：

- SD 验证：Target model 做一次前向得到 logits，比较 $p_{\text{target}}(d_i)$ vs $p_{\text{draft}}(d_i)$
- KV 验证：Agent B 需要什么信息来验证 Agent A 的 KV 是否可用？

#### 核心问题：KV Cache 的验证不像 token 验证那么便宜

SD 中验证 K 个 token 和正常 decode 1 个 token 的 FLOPs 几乎一样（batch parallelism）。

但验证 KV cache 等价性：
- 需要算 Agent B 的 "ground truth" KV → 这就是完整 prefill → 没省任何计算

**这是方向 1 的根本困境。**

---

### 更优雅的想法：不验证 KV，验证 Output

#### 思路: Speculative Prefill + Output Verification

```
Step 1: Agent A 推理 [System_A + Context] → KV_A, Output_A
Step 2: Agent B 推理时：
  a) 用 Agent B 的 System_B 单独计算 KV_B_prefix（仅 System_B 部分）
  b) Context 部分：直接使用 Agent A 的 KV_A（投机复用）
  c) Decode 阶段正常生成 Output_B
  d) 不做 KV 验证 — 让 Output_B "自己说话"

Step 3: 质量验证（可选/异步）
  a) 对 Output_B 的 perplexity 做 sanity check
  b) 或用另一次独立推理验单（offline）
```

**为什么这更优雅**：
- 完全跳过 KV 验证（最大开销来源）
- 如果 KV 足够相似 → Output_B 质量无损
- 如果 KV 不够相似 → Output_B 质量下降，但可以通过 output-level 检测

#### 但这有一个数学问题......

SD 的美在于它有**精确的数学保证**：输出分布等价。上面的"投机 KV 复用"没有这个保证——它引入了近似误差。

**能否给 KV 复用也加上 rejection sampling？**

#### 新思路: Per-Layer KV Rejection

```
对 Transformer 的每一层 l = 1..L:
  
  Agent B 在第 l 层的 "ground truth" hidden state: h_B^l = Attn(Q_B, K_B, V_B) + FFN(...)
  Agent B 用 Agent A 的 KV 在第 l 层的 hidden state: h_B'^l = Attn(Q_B, K_A, V_A) + FFN(...)
  
  如果 cos_sim(h_B^l, h_B'^l) > τ:
    Accept: 第 l 层继续用 Agent A 的 KV
  else:
    Reject: 第 l 层及后续层用 Agent B 自己重算的 KV
```

**问题**：要计算 $h_B^l$（ground truth），需要先有 Agent B 的 KV......循环论证。

#### 打破循环：利用 Transformer 的局部性

**关键物理直觉**：在 Transformer 中，early layers 主要学的是 local/syntactic pattern，late layers 学 semantic pattern。System prompt 的影响主要体现在 early layers，到了 late layers 对 shared context 的编码差异很小。

**分层投机策略**：

```
Agent B 推理 [System_B + Context]:

Layer 1-L_split:  用 Agent B 自己的参数完整计算 KV（精确）
Layer L_split+1-L:  复用 Agent A 的 KV（投机）

验证：只比较 Layer L_split 处 Agent A 和 Agent B 的 hidden states
如果相似 → 后续层全部复用（因为输入相似 → 输出也相似）
如果不相似 → 所有层都独立计算
```

**关键优势**：
- 验证只需在**一个断面**做，不需要逐层验证
- 前 $L_{split}$ 层的计算量 = $\frac{L_{split}}{L}$ × 完整 prefill
- 如果 $L_{split}$ 小（比如 1/4 的层数），节省 75% 的 prefill 计算

**但还有更优雅的思路......**

---

### 最优雅的方案：Attention Sink + KV Transplant

#### 核心观察

论文 [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) 发现：Transformer 的 attention 有一个 "sink" 现象——**大量 attention weight 集中在最初的几个 token**，而中间 token 的具体 KV 值对后续生成影响很小。

在 multi-agent 场景中：
- System prompt = attention sink 区域（前几个 token 接收大量 attention）
- Shared context = 中间区域（attention weight 分散）
- 当前 token = 正在生成的位置

Agent A 和 Agent B 的 KV 差异主要来自 System prompt（sink 区域）。对于 shared context 中的位置，**KV 值主要由 local context 决定，system prompt 的影响通过 attention weight 衰减**。

#### KV Transplant 方案

```
Agent A: [System_A (S_a tokens)] + [Context (C tokens)]
Agent B: [System_B (S_b tokens)] + [Context (C tokens)]

Step 1: Agent A 正常推理，缓存所有 KV

Step 2: Agent B 推理时：
  a) 独立计算 System_B 的 KV（positions 0..S_b-1）
  b) 对 Context 部分（positions S_b..S_b+C-1）：
     - Key/Value 从 Agent A 的 KV cache 中 **移植**
     - 但 position offset 需要调整（因为 S_a ≠ S_b → position 不同 → RoPE 不同）
  c) RoPE 修正：对移植的 K 做 position re-encoding
     - K_transplant = RoPE(K_A, pos_B) / RoPE(K_A, pos_A) × K_A
     - 即：撤销 Agent A 的 RoPE，施加 Agent B 的 RoPE

Step 3: 正常 decode
```

#### RoPE 修正的数学

RoPE 是乘性的：$K_{rope} = K \odot e^{i \cdot pos \cdot \theta}$

所以 position 修正可以精确计算：

$$K_B^{transplant}(pos_B) = K_A(pos_A) \odot \frac{e^{i \cdot pos_B \cdot \theta}}{e^{i \cdot pos_A \cdot \theta}} = K_A(pos_A) \odot e^{i \cdot (pos_B - pos_A) \cdot \theta}$$

**这是一个 O(1) 的逐元素操作！不需要重新计算 QKV projection 和 FFN。**

#### 问题

1. **KV 本身的差异**：即使修正了 RoPE，Key 和 Value 的值也因为 system prompt 不同而有差异（FFN 的输出不同）
2. **Attention pattern 差异**：Agent B 对 Context 部分的 attention 计算使用了 System_B 的 Query，但 Key 来自 Agent A（经过不同 FFN）

所以 KV Transplant 也是**近似**的，不是精确等价。其精度取决于 system prompt 差异对 hidden states 的影响有多大。

---

### 综合思考：什么是 "最优雅" 的方案？

回到 Speculative Decoding 的核心美学：**数学保证 + 零质量损失 + 自适应 fallback**。

能否对 KV 复用做到这三点？

#### Speculative KV Reuse with Output-Level Rejection

```
核心协议:

1. SPECULATE: 
   Agent B 使用 Agent A 的 KV（with RoPE correction）做 decode
   生成 K 个 draft tokens: d_1, d_2, ..., d_K

2. VERIFY (cheap):
   Agent B 对前 K 个 draft token 做一次独立的 forward（用正确的 KV）
   只需要在 K 个位置比较 logits

3. ACCEPT/REJECT:
   使用标准的 rejection sampling：
   对每个 d_i: accept if p_correct(d_i) / p_speculated(d_i) >= u_i
   
4. FALLBACK:
   如果 rejection rate > threshold:
     放弃 KV transplant，完整 prefill Agent B
   否则:
     继续用 transplanted KV 生成
```

**优势**：
- 如果 KV 足够相似 → accept rate 高 → 省了 prefill 的 C×L×12H² FLOPs
- 如果 KV 不够相似 → 前 K 个 token 的 verify 暴露了不一致 → fallback
- Verify 的开销 = K 个 token 的 forward = 远小于完整 prefill C 个 token
- **保持了 SD 的 "自适应" 特性**：不需要预知质量，运行时自动判断

**局限**：
- 不像 SD 那样有**精确等价**的数学保证
- Accept 后的后续 token 仍然使用 approximate KV
- 但可以定期 verify（每 K 步做一次 rejection check）

---

### 实验计划

#### 实验 1: KV Similarity Profiling

目标：度量不同 system prompt 下 shared context KV 的实际相似度

```python
# 用 Qwen3-0.6B
# 3 组 agent system prompts（planner/coder/reviewer）
# 10 种 shared context
# 度量：每层、每 position 的 KV cosine similarity
# 输出：similarity heatmap (layer × position)
```

#### 实验 2: KV Transplant 质量

目标：直接用 Agent A 的 KV 给 Agent B，看输出质量变化

```python
# Ground truth: Agent B 完整 prefill 的输出
# Approximate: Agent B 用 Agent A 的 KV (with RoPE correction)
# Metric: BLEU / perplexity / exact match
```

#### 实验 3: Speculative KV Reuse 的 Accept Rate

目标：测量 output-level rejection sampling 的 accept rate

```python
# 用 transplanted KV 生成 K=8 个 token
# 用 ground truth KV 验证
# 统计 accept rate vs system prompt 差异
```
