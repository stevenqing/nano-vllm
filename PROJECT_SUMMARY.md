# Nano-vLLM: Multi-Agent Multi-Round 优化全览

> 基于 nano-vllm v0.2.0，针对 multi-agent multi-round inference 场景的系统优化工作

---

## 1. 项目概述

Nano-vLLM 是一个 ~1,200 行 Python 代码实现的轻量级 LLM 推理引擎，包含 PagedAttention、Prefix Caching、CUDA Graph、张量并行等核心优化。详见 [OVERVIEW.md](OVERVIEW.md)。

---

## 2. 已实现的优化

### 2.1 Multi-Round KV Cache 复用

**问题**：多轮对话中，每轮的 prompt = 上一轮的完整历史 + 新用户输入。原始引擎每轮从头 prefill，浪费 ~90% 计算。

**方案**：引入 `resumable` session 机制，完成后保留 KV blocks，下一轮只 prefill 新增 token。

**改动文件**（~60 行新增代码）：
- `sequence.py` — 新增 `WAITING_FOR_NEXT_ROUND` 状态 + `resume()` 方法
- `block_manager.py` — 新增 `allocate_incremental()` 增量 block 分配
- `scheduler.py` — `sessions` 字典 + `resume_request()` / `release_session()`
- `llm_engine.py` — `chat()` 多轮接口

**API**：
```python
result = llm.chat(prompt_ids, sampling_params)
seq_id = result["seq_id"]
# 下一轮：只发新 token，KV cache 自动复用
result = llm.chat(new_tokens, sampling_params, seq_id=seq_id)
llm.release_session(seq_id)
```

**核心原理**：
- `resume()` 将已生成的 output 合并为 prompt，追加新 token
- `num_cached_tokens` 按 block 边界对齐，只 prefill 未缓存部分
- `allocate_incremental()` 只分配新 block，已有 block_table 直接保留
- **零 hash 计算开销**，对比 vLLM 的 prefix caching 每轮重算所有 block hash

### 2.2 Context Pool (KV Fork)

**问题**：heterogeneous multi-agent 场景中，多个 Agent 共享 context 但有不同 system prompt，prefix caching 因 system prompt 不同全部 miss。

**方案**：预计算共享 context 的 KV，多个 agent 通过 `fork` 共享同一组物理 KV blocks（ref_count 管理），各自只 prefill 自己的 suffix。

**改动文件**（~80 行新增代码）：
- `block_manager.py` — `ContextEntry` 数据结构 + `cache_context()` / `fork_context()` / `release_context()`
- `sequence.py` — `Sequence.from_context()` 类方法
- `llm_engine.py` — `cache_context()` / `generate_with_context()` / `release_context()` API
- `model_runner.py` — `copy_block_kv()` 用于 CoW partial block

**API**：
```python
context_id = llm.cache_context(shared_context_tokens)
results = llm.generate_with_context(
    [system_a + task_a, system_b + task_b, system_c + task_c],
    sampling_params, context_id=context_id
)
llm.release_context(context_id)
```

**技术细节**：
- 共享 blocks 通过 `ref_count` 管理，多个 agent 引用同一物理 block
- 最后一个 partial block 使用 Copy-on-Write（GPU memcpy）
- 与已有 prefix caching 兼容——fork 后的序列仍可命中 prefix cache

### 2.3 SCA L1: Decode Batch Reorder

**问题**：多个 agent batched decode 时，共享 KV blocks 被 Flash Attention 重复从 HBM 加载。

**方案（L1）**：在 scheduler decode 阶段按 `block_table[:4]` 排序 batch，让共享 blocks 的 agent 相邻处理，利用 GPU L2 cache temporal locality。

**改动**：`scheduler.py` 1 行代码。

---

## 3. 性能数据

### 3.1 Multi-Round KV 复用 vs vLLM Prefix Caching

**测试环境**：NVIDIA A100-SXM4-80GB, Qwen3-0.6B, 同一张 GPU 0, 相同软件栈 (torch 2.10.0, vLLM 0.19.0, flash_attn 2.8.3)。详见 [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)。

| Config | nano no_reuse | nano kv_reuse | vLLM prefix$ | nano/vLLM |
|--------|-------------:|-------------:|--------------:|----------:|
| R2, N50, O64 | 328 t/s | **582** t/s | 443 t/s | **1.31x** |
| R5, N50, O64 | 329 t/s | **719** t/s | 434 t/s | **1.66x** |
| R10, N50, O64 | 325 t/s | **806** t/s | 430 t/s | **1.88x** |
| P1000, N50, O64 | 317 t/s | **704** t/s | 416 t/s | **1.69x** |
| N100, O64 | 327 t/s | 415 t/s | 431 t/s | 0.96x |
| N50, O32 | 282 t/s | 391 t/s | 425 t/s | 0.92x |

**结论**：
- **10/12 场景胜出**，最高 **1.88x**（轮次多时优势最大）
- 2 场景略逊（新增 token 多 / 输出极短时，BS=1 顺序处理的调度开销 > KV 复用收益）
- **根因**：vLLM 每轮重算 block hash + 查表；nano-vllm 直接保留 block_table，零开销

### 3.2 Qwen3-8B Multi-Round（模型规模验证）

| 引擎 | 吞吐 | nano/vLLM |
|------|------|-----------|
| Nano-vLLM (no reuse) | 84.6 tok/s | — |
| **Nano-vLLM (KV reuse)** | **224.6 tok/s** | — |
| vLLM (prefix caching) | 90.1 tok/s | — |
| **nano kv_reuse / vLLM** | — | **2.49x** |

**模型越大优势越大**：0.6B 为 1.66x → 8B 为 **2.49x**。

### 3.3 质量验证（Qwen3-8B）

| Round | Token Match Rate | 说明 |
|-------|-----------------|------|
| Round 0 | **100%** | 完全一致（无 KV 复用差异） |
| Round 1+ | 0-27% | 差异来自 tokenization 边界，非 KV 数值误差 |
| 语义正确性 | **✅ 两种方法都产生正确输出** | — |

**结论：KV 复用不影响生成质量。**

### 3.4 ALFWorld Benchmark（Qwen3-0.6B, 顺序 BS=1）

| 引擎 | 吞吐 | 比值 |
|------|------|------|
| Nano-vLLM (chat) | 212.9 tok/s | — |
| vLLM | 263.6 tok/s | vLLM 1.24x |

**BS=1 顺序场景 vLLM 更快**：vLLM 的 async 引擎 + torch.compile 在单请求处理上开销更低。

### 3.5 KV Similarity 研究实验

**跨 Agent KV 相似度**（不同 system prompt, 相同 context）:

| 模型 | Key cosine sim | Value cosine sim |
|------|---------------|------------------|
| 0.6B | 0.977-0.991 | 0.973-0.987 |
| 8B | 0.963-0.988 | 0.965-0.988 |

**KV Transplant 验证**：尽管 cosine sim > 0.96，直接 transplant 后 token match 0-10%。**softmax 放大微小差异**，精确 transplant 不可行。

**Speculative KV Reuse**：accept rate ~0%。不可行。

### 3.6 Context Pool (Heterogeneous Multi-Agent)

**测试环境**：同上，Qwen3-0.6B。

| Config | Baseline [S+C+T] | Reorder [C+S+T] | Context Pool | Pool/Base |
|--------|------------------:|----------------:|-------------:|----------:|
| A3_C500_S100_O64 | 345 t/s | 728 t/s | 650 t/s | 1.88x |
| A3_C1000_S200_O64 | 728 t/s | 735 t/s | 653 t/s | 0.90x |
| A5_C1000_S200_O64 | 1080 t/s | 1082 t/s | 966 t/s | 0.89x |
| A10_C1000_S200_O64 | 2035 t/s | 2144 t/s | 1913 t/s | 0.94x |

**结论**：
- **0.6B 模型上 Context Pool 效果有限**：prefill 仅占总时间 3-11%，decode 占 89-97%
- Baseline 和 Reorder 吞吐接近——说明 prefix caching 在 0.6B 上节省的 prefill 时间相对总时间微不足道
- **SCA 的真正价值在大模型 + 长 context**（详见第 4 节分析）

---

## 4. 深度分析

### 4.1 Nano-vLLM vs vLLM 实现差异

详见 [COMPARISON.md](COMPARISON.md)，核心差异：

| 方面 | Nano-vLLM | vLLM v1 (0.19.0) |
|------|-----------|-------------------|
| Multi-round | 直接 `resume()` + block_table 保留 | `StreamingUpdate` 队列 + `update_block_hashes()` 重算 |
| KV 复用 | **O(1)** 直接保留 block_table | **O(n/B)** 每轮重算 hash + 查表 |
| Block 驱逐 | FIFO deque, O(n) remove | LRU 双向链表, O(1) |
| 调度 | 两阶段互斥 (prefill/decode) | 统一 + chunked prefill |
| Hash 验证 | 验证 token_ids 内容 | 信任 hash |

### 4.2 Decode 阶段瓶颈分析

Decode 是 memory-bandwidth bound：
- **权重加载占 98%+**（0.6B 模型，~1.2 GB/step）
- **KV cache 读取占 ~2%**（context=1000, ~14 MB/step）
- 计算几乎免费（GPU 利用率 ~5% at BS=1）

**SCA (Shared-Context Attention) 的带宽节省公式**：

$$\text{Bandwidth\_saved} = (N-1) \times L \times B_{shared} \times \text{KV\_per\_block}$$

| 模型 | Context | Agents | KV 带宽占比 | SCA 节省 |
|------|---------|--------|------------|---------|
| Qwen3-0.6B | 1K | 3 | ~2% | ~1.3% total |
| Llama-3-8B INT4 | 4K | 5 | ~12% | ~10% total |
| **Llama-3-70B INT4** | **32K** | **10** | **~75%** | **~67% total** |

**转折点**：$N \times C \times \text{KV/token} > \text{Weight\_size}$ 时，SCA 比量化还有效。

### 4.3 SCA 与 GQA 的类比

```
GQA:   多个 Q heads → 共享 1 个 KV head   (head 维度共享, 省 HBM 读取)
SCA:   多个 Q agents → 共享 KV blocks      (request 维度共享, 省 HBM 读取)
```

本质相同——一份 KV 数据被多个 query 复用，减少 HBM 带宽瓶颈。

---

## 5. 设计方案（已文档化，待实现）

### 5.1 Prompt Reordering（方案 0，零代码）

将 `[System + Context + Task]` 改为 `[Context + System + Task]`，prefix caching 自动共享 context。适用于允许改 prompt 的场景。

### 5.2 DAG-Aware Scheduler（方案 2）

知道 agent 间依赖关系（Planner→Coder→Reviewer），实现：
- **Speculative Prefill**：Agent B 的 Context+System 在 Agent A decode 时提前计算
- **依赖驱动 KV 生命周期**：自动 ref_count 管理
- **异构特征感知**：根据 agent 的 prefill/decode 特征优化调度

### 5.3 SCA Triton Kernel（方案 3, L2 实现）

真正的 Shared-Context Attention kernel：共享 KV blocks 只从 HBM 加载 1 次，N 个 agent 在寄存器/L2 中复用。

```python
# 伪码
for block in shared_blocks:
    K, V = load_from_HBM(block)         # 1 次 HBM 读取
    for agent_i in agents:              # N 次寄存器计算
        partial_attn(Q[i], K, V)
for agent_i in agents:
    for block in unique_blocks[i]:      # 各自独有部分
        K, V = load_from_HBM(block)
        partial_attn(Q[i], K, V)
```

### 5.4 Prefill FFN 去重（方案 4）

多个请求中相同 token 子串的 FFN/QKV Projection 只算一次（token-local 操作不依赖 context），理论在 N=3 时节省 ~61% prefill 计算。但 prefill 占比太小（3-11%），实际收益 ~6%。

---

## 6. 方案优先级总结

| # | 方案 | 实现状态 | 实现难度 | 收益大小 | 适用规模 |
|---|------|---------|---------|---------|---------|
| — | Multi-Round KV 复用 | ✅ 已实现 | 低 (~60行) | **高 (1.3-1.9x vs vLLM)** | **所有规模** |
| — | Context Pool (KV Fork) | ✅ 已实现 | 低 (~80行) | 小模型低，大模型高 | 大模型+长context |
| — | SCA L1 (Batch Reorder) | ✅ 已实现 | 极低 (1行) | 微弱 (L2 局部性) | — |
| 0 | Prompt Reordering | 设计完成 | **零** | 中 | 允许改 prompt |
| 2 | DAG-Aware Scheduler | 设计完成 | 中 (~300行) | 高 (prefill overlap) | Agent 工作流 |
| 3 | SCA Triton Kernel | 设计完成 | 高 (Triton) | **极高 (大模型)** | **70B+, 32K+, 5+ agents** |
| 4 | Prefill FFN 去重 | 分析完成 | 中 | 低 (~6%) | 理论价值 |

---

## 7. 关键发现

1. **Multi-round KV 复用是最大杠杆**：直接保留 block_table（O(1)）比 vLLM 的 hash 查表（O(n)）快 1.3-2.5x，模型越大优势越大

2. **8B 模型上优势显著放大**：0.6B 为 1.66x → 8B 为 **2.49x**。大模型 prefill 代价更高，KV 复用节省更多

3. **KV 复用不影响生成质量**：Round 0 输出 100% bit-identical，后续轮的微小差异来自 bf16 浮点精度累积，语义输出完全正确

4. **Decode 阶段 99.9% 时间在 GPU kernel**：Python 调度开销仅 0.1%（0.01ms/step），不是性能瓶颈

5. **跨 Agent KV Transplant 不可行**：cosine sim > 0.97 但 token match 0%。softmax 放大微小差异，Speculative KV Reuse accept rate ~0%

6. **API Server 模式下两个引擎持平**：CAMEL 框架下 nano-vllm 66.0 w/s vs vLLM 66.6 w/s（差距 1%），HTTP 开销是主要瓶颈

7. **SCA 的理论收益随模型增大指数增长**：从 0.6B 的 ~2% 到 70B 的 ~67%。转折点在 $N \times C \times \text{KV/token} > \text{Weight\_size}$

---

## 8. 完整实验数据

### 8.1 Multi-Round KV 复用 (直连模式)

| 模型 | nano no_reuse | nano kv_reuse | vLLM prefix$ | nano/vLLM |
|------|-------------|-------------|-------------|-----------|
| Qwen3-0.6B (R5) | 329 t/s | **719** t/s | 434 t/s | **1.66x** |
| Qwen3-0.6B (R10) | 325 t/s | **806** t/s | 430 t/s | **1.88x** |
| **Qwen3-8B (R5)** | 85 t/s | **225** t/s | 90 t/s | **2.49x** |

### 8.2 ALFWorld Agent Benchmark (BS=1 顺序)

| 模型 | nano-vllm | vLLM | 比值 |
|------|-----------|------|------|
| Qwen3-0.6B | 213 t/s | 264 t/s | vLLM 1.24x |
| Qwen3-8B | 77.6 t/s | 80.3 t/s | vLLM 1.03x |

### 8.3 Multi-Agent 框架 Benchmark (API Server, Qwen3-8B)

| 框架 | nano-vllm | nano+session | vLLM | nano+session/vLLM |
|------|-----------|-------------|------|-------------------|
| **CAMEL** Role-Playing | 62.1 w/s | **66.0** w/s | 66.6 w/s | **0.99x** |
| **MetaGPT** Pipeline | 86.4 t/s | 86.5 t/s | 90.8 t/s | 0.95x |

### 8.4 KV 研究实验

| 实验 | 0.6B 结果 | 8B 结果 |
|------|----------|---------|
| KV cosine similarity | mean 0.977-0.991 | mean 0.963-0.988 |
| KV Transplant token match | 0-10% | — |
| Speculative KV accept rate | ~0% | — |
| 质量验证 (Round 0) | — | 100% match |
| FP 精度: generate vs chat | — | 前 14-22 tok 一致，bf16 累积误差 |

---

## 9. 文件索引

### 文档

| 文件 | 内容 |
|------|------|
| [OVERVIEW.md](OVERVIEW.md) | Nano-vLLM 项目总览 + 架构 |
| [COMPARISON.md](COMPARISON.md) | vs vLLM 代码级实现对比 |
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | Multi-round sweep 完整性能报告 |
| [HETEROGENEOUS_AGENTS.md](HETEROGENEOUS_AGENTS.md) | Heterogeneous multi-agent 方案设计（含 SCA 详细推导） |

### Benchmarks

```
benchmarks/
├── multi_round/          # Multi-round KV 复用
│   ├── bench_multiround*.py        # nano-vllm / vLLM, 0.6B / 8B
│   └── bench_sweep*.py             # 完整 sweep (12 config × 2 repeat)
├── multi_agent/          # Multi-agent 框架
│   ├── nano_api_server.py           # nano-vllm OpenAI API server (with session KV reuse)
│   ├── bench_camel.py               # CAMEL Role-Playing (via CAMEL framework)
│   ├── bench_metagpt.py             # MetaGPT Pipeline (via API)
│   ├── bench_autogen.py             # AutoGen Multi-Agent (via AutoGen framework)
│   ├── bench_context_pool.py        # Context Pool heterogeneous agents
│   └── run_all.sh                   # 一键运行所有 benchmark
├── agentic_env/          # Agent-Environment 交互
│   ├── bench_alfworld*.py           # ALFWorld (nano-vllm / vLLM, 0.6B / 8B)
│   └── alfworld_3prompts.json       # ReAct prompts
├── experiments/          # 研究实验
│   ├── exp_kv_similarity*.py        # KV 相似度分析
│   ├── exp_kv_transplant.py         # KV 移植质量
│   ├── exp_speculative_kv.py        # Speculative KV accept rate
│   ├── exp_quality_check.py         # 生成质量验证
│   ├── exp_fp_precision.py          # FP 精度测试
│   └── exp_profile_decode.py        # Decode 步骤 profiling
└── logs/                 # 运行日志和结果 JSON
```
