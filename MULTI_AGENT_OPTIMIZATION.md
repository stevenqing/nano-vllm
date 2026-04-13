# Multi-Agent Multi-Round 优化方案

## 1. 场景分析

### 1.1 什么是 Multi-Agent Multi-Round

```
Agent A (System Prompt A)          Agent B (System Prompt B)
   │                                  │
   ├── Round 1: User → Agent A        │
   │   └── Agent A 输出               │
   ├── Round 2: A输出 → Agent B ──────┤
   │                                  ├── Agent B 处理
   │   ┌── Agent B 输出 ──────────────┘
   ├── Round 3: B输出 → Agent A       │
   │   └── Agent A 继续推理           │
   └── Round N: ...                   └── ...
```

典型场景：
- **多 Agent 协作**：Planner → Coder → Reviewer → Executor，每个 Agent 有独立 system prompt + tool 定义
- **多轮对话**：同一 Agent 的上下文随着对话轮次不断增长
- **混合模式**：多个 Agent 各自进行多轮对话，且 Agent 间传递中间结果

### 1.2 Workload 特征

| 特征 | 描述 | 对引擎的影响 |
|------|------|-------------|
| **大量共享前缀** | 多个 Agent 共享相同 system prompt / tool schema | Prefix Caching 命中率关键 |
| **增量式上下文增长** | 每轮对话追加 user+assistant 历史 | 需要高效的 append-only KV 复用 |
| **跨请求 KV 复用** | Round N+1 的 prompt = Round N 的完整上下文 + 新 user message | 当前引擎完全不支持 |
| **长上下文** | 多轮积累后 prompt 可达数千~数万 tokens | Prefill 成为瓶颈 |
| **突发请求模式** | Agent A 输出完成后立即触发 Agent B 请求 | 需要低延迟响应 |
| **高并发** | 多个用户各自有多 Agent 工作流并行 | 调度优先级与公平性 |

---

## 2. 当前引擎的瓶颈

### 2.1 跨请求 KV Cache 完全丢失（核心问题）

当前 `generate()` 是 **批量一次性** 的接口：

```python
# llm_engine.py
def generate(self, prompts, sampling_params):
    for prompt in prompts:
        self.add_request(prompt, sp)       # 全部加入
    while not self.is_finished():
        self.step()                         # 全部跑完
    return outputs                          # 一次性返回
```

**问题**：
1. 每次 `generate()` 调用是独立的，上一次调用的 KV Cache 在序列 FINISHED 后被 `deallocate()` 释放
2. Multi-round 场景中，Round N+1 的 prompt 包含 Round N 的所有历史，需要从头 prefill
3. 即使 Prefix Caching 能命中 block 级别的 hash，**已释放的 block 会被新请求覆盖**

**代价估算**：
```
Round 1: Prefill 500 tokens    → 500 tokens 计算
Round 2: Prefill 1200 tokens   → 1200 tokens 计算 (重复 700 tokens)
Round 3: Prefill 2000 tokens   → 2000 tokens 计算 (重复 1500 tokens)
...
Round 10: Prefill 8000 tokens  → 8000 tokens 计算 (重复 7500 tokens)
                                  总计: 浪费了 ~90% 的 prefill 计算
```

### 2.2 Prefix Caching 粒度太粗

当前实现：
- Block size = 256 tokens
- 只有**完整填满的 block** 才会被 hash 和缓存
- 最后一个不满的 block 的 hash = -1，不参与 prefix cache

**问题**：
- Multi-round 对话中，每轮新增的 user message 通常只有几十个 token
- 新 message 会破坏最后一个 block 的边界，导致该 block 之后的所有 block 都 cache miss
- 例：Round 1 = 500 tokens (Block 0 cached, Block 1 未满)，Round 2 追加 50 tokens → Block 1 内容变了 → cache miss

### 2.3 无 Streaming / Online Serving 接口

当前只有 `generate()` 一个 batch 接口：
- 不支持中途添加新请求
- 不支持 streaming token 输出
- 不支持保持 engine 常驻，跨调用复用 KV Cache

### 2.4 无 Session / Conversation 抽象

- `Sequence` 只建模单次请求，没有 "会话" 概念
- 无法表达 "这个请求是上一个请求的延续"
- 无法在多轮之间传递 block_table

### 2.5 调度器缺乏优先级机制

- FIFO 调度，无法区分不同 Agent 的紧急程度
- 无法优先调度 "已有大量 cached KV" 的请求（减少 prefill 计算）

---

## 3. 优化方案

### 3.1 核心优化：Multi-Round KV Cache 复用（Incremental Prefill）

**目标**：Round N+1 只计算新增的 token，复用 Round N 的全部 KV Cache。

#### 3.1.1 引入 Session 抽象

```python
class Session:
    """跨轮次对话会话，持有 KV Cache 的 block_table"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.token_ids: list[int] = []      # 历史所有 token
        self.block_table: list[int] = []     # 物理 block 映射
        self.num_tokens: int = 0
        self.is_active: bool = True          # 是否保持 KV Cache
```

#### 3.1.2 修改请求流程

```
Round 1 (新 Session):
  add_request(prompt, session_id="sess_1")
  → 创建 Session, 正常 prefill 500 tokens
  → 生成完毕，保留 block_table 和 token_ids，不 deallocate
  
Round 2 (续 Session):
  add_request(prompt_round2, session_id="sess_1")
  → 发现 sess_1 已有 500 tokens 的 KV Cache
  → 新 prompt = 历史 500 + 新增 50 tokens = 550 tokens
  → 只需 prefill 新增的 50 tokens（从 position 500 开始）
  → 复用原有 block_table，只追加新 block
```

#### 3.1.3 实现要点

```python
# 修改 Sequence，支持从 Session 继承
class Sequence:
    def __init__(self, token_ids, sampling_params, session=None):
        if session:
            self.token_ids = session.token_ids + token_ids  # 历史 + 新增
            self.block_table = session.block_table.copy()   # 继承 block_table
            self.num_cached_tokens = session.num_tokens     # 已有 KV 全部视为 cached
        ...

# 修改 BlockManager，支持 Session block 保持
class BlockManager:
    def hold_blocks(self, session: Session):
        """保持 session 的 blocks，增加 ref_count 防止被回收"""
        for block_id in session.block_table:
            self.blocks[block_id].ref_count += 1
    
    def release_session(self, session: Session):
        """释放 session 的所有 blocks"""
        for block_id in session.block_table:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self._deallocate_block(block_id)
```

**收益**：Multi-round 场景下 prefill 计算量从 $O(n^2)$ 降至 $O(n)$（累计）。

---

### 3.2 Online Serving 模式

**目标**：引擎常驻运行，支持随时添加请求 + streaming 输出。

#### 3.2.1 异步请求接口

```python
class LLMEngine:
    def add_request(self, prompt, sampling_params, session_id=None) -> str:
        """返回 request_id，非阻塞"""
        ...
    
    def step(self) -> list[StepOutput]:
        """执行一步推理，返回所有有新 token 的请求的增量输出"""
        ...
    
    def abort_request(self, request_id: str):
        """中止请求（Agent 决定不需要了）"""
        ...
    
    def serve(self, callback):
        """持续运行主循环"""
        while True:
            outputs = self.step()
            for output in outputs:
                callback(output)
```

#### 3.2.2 Streaming Token Output

```python
@dataclass
class StepOutput:
    request_id: str
    session_id: str | None
    token_id: int           # 本步生成的 token
    text: str               # 增量 detokenize 结果
    is_finished: bool
    finish_reason: str      # "eos" | "max_tokens" | "abort"
```

**收益**：
- Agent A 输出完成后，Agent B 可以立即提交请求，无需等待整个 batch
- 支持 streaming 输出，降低首 token 延迟 (TTFT)
- 引擎常驻避免反复初始化

---

### 3.3 Prefix Caching 优化

#### 3.3.1 更细粒度的 Hash（可选）

考虑支持小于 block_size 的 hash 粒度，或者 **Token-Tree Hashing**：

```python
# 当前：只有满 block 才 hash
h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

# 优化：部分 block 也支持 hash（用于 multi-round 场景）
# 思路：使用 (prefix_hash, partial_token_ids) 作为 key
# 如果下一轮的前缀恰好延续了这个部分 block，可以直接复用
```

但实际上，如果实现了 3.1 的 Session 级 KV 复用，prefix caching 的粒度问题就不那么关键了——因为同一 session 的 KV 完全不释放，天然是 token 级精确复用。

#### 3.3.2 Agent-Aware Prefix Grouping

```
Agent A: [System Prompt A (300 tokens)] + [对话历史 ...]
Agent B: [System Prompt B (400 tokens)] + [对话历史 ...]
Agent C: [System Prompt A (300 tokens)] + [不同对话历史 ...]
```

- Agent A 和 C 共享 System Prompt A 的 KV Cache（Block 0 和部分 Block 1）
- 可以主动标记 "这些请求共享相同前缀"，而非被动靠 hash 碰撞

```python
def add_request(self, prompt, sampling_params, prefix_group=None):
    """prefix_group: 标识共享前缀的组，同组请求优先复用 KV"""
    ...
```

#### 3.3.3 Prefix Cache Eviction 策略

当前没有显式的 eviction 策略，block 被释放后 hash 映射仍保留但 block 内容可能被覆盖。

优化方向：
- **LRU eviction**：给每个 cached block 维护 last_access_time
- **优先保留高复用前缀**：system prompt 的 block 优先保留
- **Session-aware eviction**：活跃 session 的 block 永不驱逐

---

### 3.4 调度优化

#### 3.4.1 Prefix-Aware 调度

优先调度 prefix cache 命中率高的请求，减少总 prefill 计算量：

```python
def schedule(self):
    # 按 cache 命中比例降序排列 waiting 队列
    # 命中率高的请求优先调度 → 更少的 prefill 计算
    self.waiting = deque(sorted(self.waiting, 
        key=lambda seq: self._estimate_cache_hit(seq), reverse=True))
    ...
```

#### 3.4.2 Multi-Agent 优先级

```python
class SamplingParams:
    priority: int = 0  # 请求优先级

class Scheduler:
    def schedule(self):
        # 高优先级请求优先调度
        # 支持 Agent 级别的优先级设定
        ...
```

#### 3.4.3 Chunked Prefill

对于长上下文 multi-round 请求，支持分块 prefill，与 decode 请求交错执行：

```python
# 当前：prefill 和 decode 互斥
if scheduled_seqs:    # 有 prefill 就不做 decode
    return scheduled_seqs, True

# 优化：chunked prefill
# 将长 prefill 拆成多个 chunk（如 2048 tokens/chunk）
# 与 decode batch 交替执行，降低 decode 请求的排队延迟
```

**收益**：避免长 prompt 的 prefill 阻塞所有 decode 请求。

---

### 3.5 Disaggregated Prefill / Decode（进阶）

将 Prefill 和 Decode 分离到不同的 GPU/实例：

```
┌─────────────────┐       ┌─────────────────┐
│  Prefill Worker  │──────▶│  Decode Worker   │
│  (计算密集型)     │ KV    │  (访存密集型)     │
│  处理新请求      │ 传输   │  处理 token 生成  │
└─────────────────┘       └─────────────────┘
```

这对 multi-agent 场景特别有用——大量新请求的 prefill 不会阻塞已有 agent 的 decode。

---

## 4. 实现优先级与路线图

### P0 — 基础能力（必须先做）

| 优化项 | 改动范围 | 预期收益 |
|--------|---------|---------|
| **Session 抽象 + Multi-Round KV 复用** | `sequence.py`, `block_manager.py`, `scheduler.py`, `llm_engine.py` | 多轮 prefill 计算量降 80-90% |
| **Online Serving 接口** | `llm_engine.py`, `llm.py` | 支持实时添加请求，引擎常驻 |

### P1 — 重要优化

| 优化项 | 改动范围 | 预期收益 |
|--------|---------|---------|
| **Streaming Output** | `llm_engine.py` | 降低 TTFT，改善交互体验 |
| **Chunked Prefill** | `scheduler.py`, `model_runner.py` | 长 prompt 不阻塞 decode |
| **Session-Aware Eviction** | `block_manager.py` | 提高 KV Cache 利用率 |

### P2 — 进阶优化

| 优化项 | 改动范围 | 预期收益 |
|--------|---------|---------|
| **Prefix-Aware 调度** | `scheduler.py` | 减少冗余 prefill |
| **Agent 优先级调度** | `scheduler.py`, `sampling_params.py` | 更好的延迟 SLA |
| **Disaggregated Prefill/Decode** | 全局架构 | 极致吞吐 |

---

## 5. 改动影响分析

```
nanovllm/
├── engine/
│   ├── session.py          [新增] Session 会话管理
│   ├── sequence.py         [修改] 支持从 Session 继承 block_table
│   ├── block_manager.py    [修改] Session hold/release, eviction 策略
│   ├── scheduler.py        [修改] chunked prefill, 优先级, 在线调度
│   ├── llm_engine.py       [修改] online serving, streaming, session 管理
│   └── model_runner.py     [微调] 支持 incremental prefill 的 prepare_prefill
├── llm.py                  [修改] 新增 chat/session API
├── config.py               [修改] 新增 session 相关配置
└── sampling_params.py      [修改] 新增 priority 等字段
```

---

## 6. 总结

Multi-Agent Multi-Round 场景下，**最大的优化杠杆是跨轮次 KV Cache 复用**。当前引擎每轮都从头 prefill，是最大的浪费。通过引入 Session 抽象 + Online Serving 模式，可以让多轮对话只计算增量 token，预计 **端到端推理速度提升 3-10 倍**（取决于对话轮数和上下文长度）。

后续的 Chunked Prefill、Prefix-Aware 调度、Disaggregated Prefill/Decode 则是在此基础上的进一步优化，针对高并发、长上下文场景持续提升吞吐和降低延迟。
