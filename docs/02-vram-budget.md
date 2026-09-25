# 02 · 显存账本：23 GB 花在哪了

vLLM 是"预分配"型引擎：`--gpu-memory-utilization` 的语义是**先把这块额度占住**，
权重 / 激活 / CUDA 图之外剩下的**全部变成 KV 池**。
所以 `nvidia-smi` 上看到接近满卡是**设计结果**，不是泄漏。

## 一次启动的完整分解

vLLM 自己会打印（`gpu_worker.py`，`Free memory on device ...` 那一行）。
实测（RTX 4090 24 GB，fp8 KV，`max_model_len=262144`，`max-num-batched-tokens=2048`）：

| 项 | 实测 | 说明 |
|---|---|---|
| checkpoint 盘上 | 9.42 GB | 1253 张量，3 分片 |
| 加载后权重 | 8.97 GiB | |
| weights + non-torch | **9.35 GiB** | non-torch ≈ 0.38（CUDA 上下文、通信缓冲等） |
| **peak activation** | **4.03 GiB** | ∝ `--max-num-batched-tokens` |
| **CUDA graph** | **1.75 GiB** | 34 档 piecewise + 19 档 full，每档一套静态缓冲 |
| KV 池 | **8.45 GiB** | → 265,157 token |
| 合计 | ~23.1 / 23.99 GiB | |
| 卡上其它占用 | ~1.5 GiB | 桌面的非计算进程 |

## 三个结论

1. **"引擎税" ≈ 5.8 GiB**（激活 4.03 + CUDA 图 1.75）。
   vLLM 自己给了换算：开着 CUDA graph 显存 profiling 时，
   `util 0.90` ≈ 关掉 profiling 时的 `0.8359`；想保持同样 KV 得提到 `0.9641`。
2. **`--max-num-batched-tokens` 是最灵敏的显存旋钮**：
   2048 → 8192 时激活+图多吃 ~0.9 GiB，
   KV 池 **8.45 → 7.56 GiB**，于是"单路 256K"（需要 8.33 GiB）**直接起不来**：
   ```
   ValueError: To serve at least one request with the model's max seq len (262144),
   (8.33 GiB KV cache is needed, which is larger than the available KV cache memory (7.56 GiB)
   ```
   → **动这个参数必须同步降 `max_model_len`**。
3. **`--gpu-memory-utilization` 上限约 0.93**：桌面非计算进程占着 ~1.5 GiB，
   0.94 直接报 `Free memory ... is less than desired GPU memory utilization`。

## KV 池大小怎么估

```
KV池 ≈ util × 卡容量 − 权重 − non-torch − 激活(∝batch_tokens) − CUDA图(∝capture档位)
```

- 想让 KV 更大：先降 `--max-num-batched-tokens`（最有效），再考虑降 `--max-num-seqs`
  （capture 档位变少 → 图内存变小），最后才是换低精度 KV。
- 本模型是混合架构（64 层里 48 层是线性注意力 GDN），每序列还要一份 SSM 状态，
  所以 vLLM 会把 attention block 撑到与 mamba page 齐平（fp8 下 1568 token，
  4-bit KV 下 3072）——这个 block 大小会直接影响**能并发多少路**，见
  [04-long-prefill-concurrency.md](04-long-prefill-concurrency.md)。

## 与 llama.cpp 的对比

llama.cpp 没有 CUDA graph 档位、没有 torch.compile workspace、没有 paged KV 的块表，
"引擎税"只有几十~几百 MB，所以同样 24 GB 能把更多显存放进上下文槽。
但它也没有连续批处理 / 前缀缓存。**同样 23 GB，花的地方完全不同**：
llama.cpp 大部分是上下文容量，vLLM 有 ~5.8 GiB 花在引擎侧。
