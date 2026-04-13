# Nano-vLLM 项目总览

## 1. 项目简介

**Nano-vLLM** 是一个从零实现的轻量级 vLLM 推理引擎，仅用 ~1,200 行 Python 代码，在 Qwen3-0.6B 模型上实现了比 vLLM **高 5.3% 的吞吐量**（1,434 vs 1,362 tokens/s）。

- **作者**: Xingkai Yu
- **版本**: 0.2.0
- **License**: MIT
- **Python**: ≥3.10, <3.13

---

## 2. 整体架构

```
用户调用 LLM.generate()
        │
        ▼
┌──────────────────────────────────────────┐
│          LLMEngine (主循环)               │
│   • 接收请求 → Tokenize                   │
│   • 调度循环: Schedule → Execute → Sample  │
│   • 输出结果 → Detokenize                  │
├────────────┬─────────────────────────────┤
│  Scheduler │     BlockManager            │
│  连续批处理  │     KV Cache 管理            │
│  Prefill ↔  │     Prefix Caching          │
│  Decode     │     Block 分配/回收          │
├─────────────┴────────────────────────────┤
│          ModelRunner (GPU Worker)         │
│   • 构造输入张量                            │
│   • 执行模型前向传播                         │
│   • CUDA Graph 捕获与重放                   │
│   • KV Cache 内存池管理                     │
├──────────────────────────────────────────┤
│          Qwen3Model (模型层)               │
│   Embedding → N × TransformerBlock → Head │
│   各层使用优化算子 (Flash Attn, Fused Ops)   │
└──────────────────────────────────────────┘
```

---

## 3. 文件结构与职责

### 3.1 入口与配置

| 文件 | 职责 |
|------|------|
| `nanovllm/__init__.py` | 包入口，暴露 `LLM` 和 `SamplingParams` |
| `nanovllm/llm.py` | 用户 API 层，封装 `LLMEngine`，提供 `generate()` 接口 |
| `nanovllm/config.py` | 全局配置 dataclass（批大小、显存比例、TP 数、KV block 大小等） |
| `nanovllm/sampling_params.py` | 每个请求的采样参数（temperature、max_tokens、ignore_eos 等） |

### 3.2 引擎核心 (`nanovllm/engine/`)

| 文件 | 职责 |
|------|------|
| `llm_engine.py` | **推理主循环**：请求管理、调度-执行循环、多进程 TP 协调 |
| `scheduler.py` | **连续批处理调度器**：Prefill/Decode 阶段调度、显存不足时抢占 |
| `sequence.py` | **请求生命周期**：Sequence 数据结构，跟踪 token、block table、状态 |
| `block_manager.py` | **KV Cache 块管理器**：块的分配/回收、基于 xxhash 的 Prefix Caching、引用计数 |
| `model_runner.py` | **GPU 执行器**：构造输入张量、KV Cache 显存分配、CUDA Graph 捕获与重放 |

### 3.3 模型实现 (`nanovllm/models/`)

| 文件 | 职责 |
|------|------|
| `qwen3.py` | Qwen3 Transformer 模型实现，支持张量并行 |

### 3.4 优化算子层 (`nanovllm/layers/`)

| 文件 | 职责 |
|------|------|
| `attention.py` | Flash Attention（Prefill/Decode）+ Triton KV Cache 写入 kernel |
| `linear.py` | 张量并行线性层（ColumnParallel / RowParallel） |
| `layernorm.py` | Fused RMSNorm + 残差连接（torch.compile 融合） |
| `activation.py` | Fused SiLU × Mul 门控激活 |
| `rotary_embedding.py` | RoPE 旋转位置编码（缓存 cos/sin） |
| `sampler.py` | Gumbel-max 温度采样 |
| `embed_head.py` | Embedding + LM Head（支持 TP 词表分片） |

### 3.5 工具 (`nanovllm/utils/`)

| 文件 | 职责 |
|------|------|
| `context.py` | 线程安全的执行上下文（当前 Attention 元数据、TP rank 信息） |
| `loader.py` | Safetensors 权重加载器，支持 TP 自动切分 |

### 3.6 顶层脚本

| 文件 | 职责 |
|------|------|
| `example.py` | 快速上手示例 |
| `bench.py` | 与 vLLM 的吞吐对比 Benchmark |

---

## 4. 关键技术实现

### 4.1 连续批处理 (Continuous Batching)

调度器维护三个队列：`waiting`（等待 prefill）、`running`（正在 decode）、`swapped`（被抢占）。

**调度流程**：
1. **Decode 阶段**：先处理 running 队列，为每个 seq 分配一个新 block（如果当前 block 已满）
2. **Prefill 阶段**：从 waiting 队列取出请求，分配所需 block，直到达到 `max_num_seqs` 或 `max_num_batched_tokens` 限制
3. **抢占**：显存不足时，将低优先级 seq 的 block 释放并移到 swapped 队列

### 4.2 PagedAttention 与 KV Cache 管理

- KV Cache 按 **固定大小的 Block**（默认 256 tokens/block）管理
- 每个 Sequence 维护一个 `block_table`（逻辑块 → 物理块映射）
- Triton kernel 将 KV 值写入对应 block 的正确偏移位置
- Decode 阶段使用 Flash Attention 从 block 化的 KV Cache 中读取

### 4.3 Prefix Caching

- 使用 **xxhash** 对每个 block 的 token 内容计算哈希
- 新请求的 prompt 若与已缓存 block 的 token 相同，直接复用已有 KV Cache
- 引用计数管理：block 被多个 seq 引用时不会被释放
- 显著加速共享 system prompt 的场景

### 4.4 CUDA Graph

- 在 warmup 阶段预先捕获不同 batch size（1, 2, 4, 8, ...）的 decode 计算图
- 运行时根据实际 batch size 向上取整到最近的预录 size，用 graph replay 执行
- 跳过 kernel launch 开销，decode 阶段提速约 10-20%
- `enforce_eager=True` 可禁用

### 4.5 张量并行 (Tensor Parallelism)

- **ColumnParallelLinear**：沿输出维度切分（QKV 投影、MLP gate/up）
- **RowParallelLinear**：沿输入维度切分 + all-reduce（MLP down、O 投影）
- **Embedding/Head**：词表按 rank 均匀分片
- 多进程协调：Rank 0 为主进程，其他 rank 通过 SharedMemory + Event 同步

### 4.6 Fused 算子优化

| 优化 | 实现方式 | 效果 |
|------|---------|------|
| RMSNorm + Residual | `torch.compile` 自动融合 | 减少显存读写 |
| SiLU × Mul (Gate) | 自定义 fused 实现 | 减少中间张量 |
| KV Cache Store | Triton kernel | block 化写入 |
| Gumbel-max Sampling | 数值稳定的温度采样 | 避免 softmax 精度问题 |

---

## 5. 推理流程 (端到端)

```
1. 用户调用 llm.generate(prompts, sampling_params)
       │
2. LLMEngine.add_request()
   • Tokenizer 编码 prompt
   • 创建 Sequence 对象 → 加入 waiting 队列
       │
3. 主循环 LLMEngine.step():
   ┌──────────────────────────────────────┐
   │ 3a. Scheduler.schedule()             │
   │     • 从 waiting/running 选择 seqs    │
   │     • BlockManager 分配/复用 blocks   │
   │     • 返回 SchedulerOutputs           │
   │                                      │
   │ 3b. ModelRunner.execute()            │
   │     • 构造 input_ids, positions 等    │
   │     • Prefill: 模型前向(变长)          │
   │     • Decode: CUDA Graph replay      │
   │     • 返回 logits                     │
   │                                      │
   │ 3c. Sampler 采样                      │
   │     • temperature 缩放               │
   │     • Gumbel-max 采样下一个 token      │
   │                                      │
   │ 3d. 更新 Sequence                     │
   │     • append token                   │
   │     • 检查终止条件 (EOS / max_tokens)  │
   │     • 已结束 → 移出 running            │
   └──────────────────────────────────────┘
       │
4. 所有请求完成 → Detokenize → 返回结果
```

---

## 6. 核心配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | 16384 | 一次调度最大 token 数 |
| `max_num_seqs` | 512 | 同时处理的最大序列数 |
| `max_model_len` | 4096 | 最大序列长度 |
| `gpu_memory_utilization` | 0.9 | GPU 显存使用比例 |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数 |
| `enforce_eager` | False | 禁用 CUDA Graph |
| `kvcache_block_size` | 256 | KV Cache 块大小（tokens） |

---

## 7. 依赖

| 包 | 版本要求 | 用途 |
|----|---------|------|
| torch | ≥2.4.0 | 深度学习框架 |
| triton | ≥3.0.0 | 自定义 GPU kernel (KV Cache store) |
| transformers | ≥4.51.0 | Tokenizer & 模型配置加载 |
| flash-attn | — | Flash Attention 2 实现 |
| xxhash | — | Prefix Caching 哈希 |

---

## 8. 性能数据

**测试环境**：RTX 4070 Laptop (8GB) / Qwen3-0.6B / 256 并发请求 / 输入输出长度 100-1024 随机

| 引擎 | 输出 Tokens | 耗时 (s) | 吞吐 (tokens/s) |
|------|------------|----------|-----------------|
| vLLM | 133,966 | 98.37 | 1,361.84 |
| **Nano-vLLM** | 133,966 | 93.41 | **1,434.13** |
| 提升 | — | -5.0s | **+5.3%** |

---

## 9. 设计亮点总结

1. **极简代码量 (~1200 行)** — 实现了 vLLM 的核心功能，可作为学习 LLM 推理引擎的最佳参考
2. **生产级性能** — 连续批处理 + PagedAttention + CUDA Graph + Prefix Caching，缺一不可
3. **API 兼容** — 接口设计与 vLLM 保持一致，降低迁移成本
4. **可扩展性** — 张量并行支持多卡推理，模型层可复用至其他架构
5. **零依赖自研** — 除 flash-attn/triton 外，调度器、block 管理器、采样器全部自研
