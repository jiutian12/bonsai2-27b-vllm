# 03 · KV 精度取舍：为什么同为 4-bit，两个后端差一倍

KV 精度决定两件事：**能塞多长/多少并发**（容量）与**深层解码多快**（速度）。
在我们的栈上实测了三档（RTX 4090 24 GB，`max_model_len=262144`，util 0.90）：

| KV dtype | 池大小 | 池 token | 256K 并发度 | 0 / 2.7k / 11k / 33k 解码 |
|---|---|---|---|---|
| `bf16`（auto） | — | — | **起不来** | — |
| **`fp8`** | 8.45 GiB | 265,157 | 1.01x → 1 路 | **86.8 / 84.5 / 81.4 / 74.1** |
| `turboquant_4bit_nc` | 8.77 GiB | 518,589 | 1.98x → ~2 路 | 85.2 / 73.3 / 51.8 / **27.7** |
| `kvarn_k4v4_g128` | — | — | **起不来**（0.90） | — |

## bf16 为什么不行

一路 256K（262,144 token）在 bf16 下需要 **~17.2 GiB** 的 KV，而 24 GB 卡在扣掉
权重 8.97 + 引擎税 5.8 之后只剩 ~8.5 GiB。vLLM 直接拒绝启动：

```
ValueError: To serve at least one request with the model's max seq len (262144),
(8.33 GiB KV cache is needed, which is larger than the available KV cache memory ...)
```

## fp8 是"双赢档"

fp8 不但省一半容量，**深层解码也比 bf16 更快**：解码的瓶颈是注意力读 KV 的**显存带宽**，
fp8 把 KV 字节数砍半，带宽压力直接减半（33k：fp8 74.1 vs bf16 63.9）。
再加上 **FlashAttention 3 原生支持 fp8(E4M3) KV，零转换开销** —— 所以它几乎没有代价。

## 4-bit 的代价：用算力换容量

`turboquant_4bit_nc` 的容量是 fp8 的 **1.96×**，但深层解码掉到 **1/2.7**。
原因在实现（源码级）：

1. KV 是 **4-bit 打包**存储，注意力不能直接吃。解码走 vLLM 的 Triton 融合内核
   `vllm/v1/attention/ops/triton_turboquant_decode.py`
   （docstring：*Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values*）；
   stage1 做分块打分、stage2 做 log-sum-exp 归约 —— **每一步都要把全部历史 KV 解出来**。
2. 它会**强制把 FlashAttention 退回 2 版**：
   ```
   WARNING TurboQuant is not yet compatible with FlashAttention >= 3.
           Overriding flash_attn_version to 2
   ```
3. 有一条 flydsl 快路径（`flydsl_turboquant_decode.py`），但注释写明**只在 gfx950（AMD）自动启用**
   → NVIDIA 卡只能走较慢的 SoA Triton 路径。

所以代价**随上下文线性放大**（每步过一遍全部历史 KV）：

| 上下文 | fp8 | 4-bit | 损失 |
|---|---|---|---|
| ~0 | 86.8 | 85.2 | −2% |
| ~2.7k | 84.5 | 73.3 | −13% |
| ~11k | 81.4 | 51.8 | −36% |
| ~33k | 74.1 | **27.7** | **−63%** |

反过来，**4-bit 的 prefill 更快**：26 万上下文的 prefill，fp8 要 221s，4-bit 只要 195s
（≈1344 vs 1176 tok/s）—— prefill 是算力受限的大矩阵乘，KV 读得更少、FA2 的 prefill 也不弱。
**慢的只是逐 token 的解码路径。**

## 一个容易踩的坑：同为 4-bit，后端开销可能吃掉精度优势

`kvarn_k4v4_g128` 的 **per-token 成本只有 bf16 的 1/3.7**（更省），
但它背后的 KVARN 后端**自己多吃约 4.5 GiB**，于是可用池反而更小：

- util 0.90 → 池只有 3.76 GiB → 一路 256K（4.62 GiB）**开不起来**，估出的模型上限只有 211,968；
- util 0.93 → 池 4.54 GiB，估值 259,072，仍然差一点点。

**结论：不要只看位宽，要看该后端在你的配置下的实际可用池。**

## 怎么选

| 目标 | 选择 |
|---|---|
| 单流长上下文、要最快 | **`fp8`**（33k 有 74 tok/s） |
| 同时塞两个长上下文（2 × ~258K） | **`turboquant_4bit_nc`**（深层 28 tok/s） |
| 短请求高并发 | **4-bit 反而更好**（32 路聚合 1027 tok/s，KV 池大意味着能同时排更多路） |

顺带一个反直觉结论：4-bit KV **基本不吃质量** —— 同一套 280 题，
fp8 88.2% vs 4-bit 87.9%，只差 1 题（GSM8K 完全持平）。
