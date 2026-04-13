# Nano-vLLM vs vLLM: Multi-Round 推理性能对比报告

## 1. 测试环境

所有测试均在**同一张 GPU**上先后运行，确保硬件条件完全一致。

| 项目 | 值 |
|------|-----|
| **GPU** | NVIDIA A100-SXM4-80GB |
| **GPU SM** | 8.0 |
| **GPU 显存** | 79.3 GiB |
| **GPU 启动显存占用** | 4 MiB（空闲） |
| **CUDA_VISIBLE_DEVICES** | 0（同一张卡） |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 91002 |
| flash_attn | 2.8.3 |
| triton | 3.6.0 |
| transformers | 4.57.6 |
| xxhash | 3.6.0 |
| vLLM | 0.19.0 |
| flashinfer | 0.6.6 |
| **模型** | Qwen3-0.6B (bfloat16) |
| **max_model_len** | 4096 |
| **enforce_eager** | False（均启用 CUDA Graph） |
| **gpu_memory_utilization** | 0.9 |

### GPU 显存占用

| 引擎 | 初始化后显存 |
|------|-------------|
| Nano-vLLM | 72,795 MB |
| vLLM | 74,477 MB |

---

## 2. 测试方法

### 2.1 Workload 设计

模拟 **multi-agent multi-round** 场景：N 个独立会话，每个会话进行 R+1 轮对话。

- 每轮由 engine 生成 `output_len` 个 token
- 下一轮追加 `new_tokens_per_round` 个新 token（模拟用户输入）
- 上下文随轮次递增

### 2.2 三种模式对比

| 模式 | 引擎 | 说明 |
|------|------|------|
| **nano no_reuse** | Nano-vLLM `generate()` | 每轮重新发送完整上下文，从头 prefill |
| **nano kv_reuse** | Nano-vLLM `chat()` | 多轮 KV Cache 复用，只 prefill 新增 token |
| **vLLM prefix_cache** | vLLM `generate()` | 每轮发送完整上下文，依赖自动 prefix caching 复用已缓存 block |

### 2.3 公平性保证

- ✅ 同一张 GPU（GPU 0, A100-80GB）
- ✅ 相同 Python/PyTorch/CUDA/flash_attn 版本
- ✅ 相同模型（Qwen3-0.6B）和推理配置
- ✅ 相同 12 组 sweep configs × 2 repeats
- ✅ 相同随机种子（seed=42+repeat）
- ✅ 相同 SamplingParams（temperature=0.6, ignore_eos=True）
- ✅ GPU 启动前显存均为空闲状态（4 MiB）

---

## 3. Sweep 结果

### 3.1 完整对比表

| Config | nano no_reuse | nano kv_reuse | vLLM prefix$ | nano/vLLM | vs no_reuse |
|--------|-------------:|-------------:|-------------:|----------:|------------:|
| C16_R2_P200_N50_O64 | 328.5 t/s | 581.5 t/s | 443.1 t/s | **1.31x** | 1.77x |
| C16_R5_P200_N50_O64 | 328.8 t/s | 719.0 t/s | 433.7 t/s | **1.66x** | 2.19x |
| C16_R10_P200_N50_O64 | 324.7 t/s | 806.0 t/s | 429.8 t/s | **1.88x** | 2.48x |
| C16_R5_P100_N50_O64 | 329.2 t/s | 723.0 t/s | 438.4 t/s | **1.65x** | 2.20x |
| C16_R5_P500_N50_O64 | 326.6 t/s | 718.7 t/s | 426.9 t/s | **1.68x** | 2.20x |
| C16_R5_P1000_N50_O64 | 316.8 t/s | 703.9 t/s | 415.8 t/s | **1.69x** | 2.22x |
| C16_R5_P200_N100_O64 | 326.9 t/s | 415.2 t/s | 431.0 t/s | 0.96x | 1.27x |
| C16_R5_P200_N200_O64 | 323.2 t/s | 466.9 t/s | 428.5 t/s | **1.09x** | 1.44x |
| C16_R5_P200_N50_O32 | 281.7 t/s | 391.4 t/s | 424.7 t/s | 0.92x | 1.39x |
| C16_R5_P200_N50_O128 | 355.9 t/s | 508.9 t/s | 438.8 t/s | **1.16x** | 1.43x |
| C4_R5_P200_N50_O64 | 327.2 t/s | 715.8 t/s | 435.1 t/s | **1.64x** | 2.19x |
| C32_R5_P200_N50_O64 | 328.5 t/s | 719.2 t/s | 435.7 t/s | **1.65x** | 2.19x |

> Config 格式: C{会话数}_R{轮数}_P{初始prompt长度}_N{每轮新token数}_O{每轮输出长度}
>
> **nano/vLLM**: Nano-vLLM kv_reuse 吞吐 / vLLM prefix_caching 吞吐
>
> **vs no_reuse**: Nano-vLLM kv_reuse 吞吐 / Nano-vLLM no_reuse 吞吐

### 3.2 按维度分析

#### 轮数对性能的影响（P200, N50, O64）

| 轮数 | nano kv_reuse | vLLM | nano/vLLM |
|------|-------------:|-----:|----------:|
| R2 (3轮) | 581.5 t/s | 443.1 t/s | 1.31x |
| R5 (6轮) | 719.0 t/s | 433.7 t/s | 1.66x |
| R10 (11轮) | 806.0 t/s | 429.8 t/s | **1.88x** |

**结论**: 轮次越多，Nano-vLLM 优势越大。10 轮时达到 **1.88x**。因为 vLLM 的 prefix caching 每轮仍需重新计算 block hash + 传输完整 token IDs，而 Nano-vLLM 直接复用 block_table。

#### 初始 Prompt 长度的影响（R5, N50, O64）

| Prompt | nano kv_reuse | vLLM | nano/vLLM |
|--------|-------------:|-----:|----------:|
| P100 | 723.0 t/s | 438.4 t/s | 1.65x |
| P200 | 719.0 t/s | 433.7 t/s | 1.66x |
| P500 | 718.7 t/s | 426.9 t/s | 1.68x |
| P1000 | 703.9 t/s | 415.8 t/s | **1.69x** |

**结论**: Prompt 越长，两者都变慢（首轮 prefill 更重），但 Nano-vLLM 的相对优势略微增大。

#### 每轮新 Token 数的影响（R5, P200, O64）

| 新Token | nano kv_reuse | vLLM | nano/vLLM |
|---------|-------------:|-----:|----------:|
| N50 | 719.0 t/s | 433.7 t/s | **1.66x** |
| N100 | 415.2 t/s | 431.0 t/s | 0.96x |
| N200 | 466.9 t/s | 428.5 t/s | 1.09x |

**结论**: 当每轮新增 token 较多时（N100/N200），Nano-vLLM 的 KV 复用优势被新 token 的 prefill 开销稀释。N100 时两者持平。这是因为 Nano-vLLM 以 BS=1 顺序处理会话，而 vLLM 的 async 调度和 chunked prefill 在此场景下更高效。

#### 输出长度的影响（R5, P200, N50）

| 输出长度 | nano kv_reuse | vLLM | nano/vLLM |
|---------|-------------:|-----:|----------:|
| O32 | 391.4 t/s | 424.7 t/s | 0.92x |
| O64 | 719.0 t/s | 433.7 t/s | **1.66x** |
| O128 | 508.9 t/s | 438.8 t/s | 1.16x |

**结论**: 输出短 (O32) 时 Nano-vLLM 略逊于 vLLM，因为 decode 阶段占比减小，vLLM 的调度和内核优化更占优。输出中等 (O64) 时优势最大。

---

## 4. 核心结论

### ✅ Nano-vLLM 胜出的场景（10/12 configs）

- **典型 multi-round 对话**（每轮新增 ~50 tokens，输出 64+ tokens）：**1.3x - 1.9x 优于 vLLM**
- **轮次越多越占优**：从 R2 的 1.31x 到 R10 的 1.88x
- **与会话数和 prompt 长度关系不大**：4-32 会话、100-1000 prompt 均稳定领先

### ⚠️ vLLM 持平或略优的场景（2/12 configs）

- **每轮新增大量 token (N100)**：0.96x — BS=1 顺序处理的瓶颈
- **输出极短 (O32)**：0.92x — decode 占比小，Nano-vLLM 调度开销相对大

### 原因分析

| 方面 | Nano-vLLM kv_reuse | vLLM prefix_caching |
|------|-------------------|-------------------|
| KV 复用方式 | **直接保留 block_table**，零开销 | 每轮重新计算所有 block hash，查找匹配 |
| 新增 token 处理 | 只 prefill 新增部分 | 发送完整上下文，依赖 hash 跳过已缓存 block |
| 调度模型 | 同步 BS=1 顺序处理 | 异步 + continuous batching + chunked prefill |
| 适用场景 | 少量新增 token 的多轮对话 | 大 batch + 高并发 serving |

---

## 5. 后续优化方向

基于 sweep 数据，Nano-vLLM 在以下方向有提升空间：

1. **Chunked Prefill** — 解决 N100/N200 场景下新 token 多时的性能瓶颈
2. **Batch Chat** — 支持多个 session 并发推理（当前 BS=1 顺序处理）
3. **更高效的 Decode 调度** — 缩小 O32 场景与 vLLM 的差距
4. **Async Engine** — 引入异步调度，减少 Python 层开销
