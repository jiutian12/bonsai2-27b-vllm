# 04 · 长 prompt 的并发 prefill 为什么退化（以及代价）

## 现象

在**刚重启、KV 池干净**的实例上，用**互不相同**的 prompt 同时发两条 ~130K 的请求：

```
#0: prompt=129955  TTFT=81.30s
#1: prompt=129937  TTFT=162.34s     ← 正好是两倍
```

`/metrics` 显示 `peak num_requests_running = 2`、`num_requests_waiting = 1`，
而 **peak `kv_cache_usage_perc` 只有 0.62** —— 也就是说两条请求**从来没有同时满长**，
第二条是在第一条 prefill 完之后才真正开始。

对照组：**单条 260K 请求能跑通**（`kv_cache_usage_perc` 顶到 1.0）。
所以这**不是容量问题，是调度问题**。

## 根因链（源码级）

1. 这是混合 GDN 模型（64 层里 48 层线性注意力），vLLM 的 mamba cache mode 默认是 `align`，
   于是它会**把 attention block 撑到与 mamba page 等大**：
   ```
   Setting attention block size to 1568 tokens to ensure that attention page size >= mamba page size.
   Padding mamba page size by 0.13% to ensure that mamba page size and attention page size are exactly equal.
   ```
   （1568 是 fp8 KV 下的值；换 4-bit KV 后会变成 **3072**。）
2. 在 24 GB 的卡上用 `vllm serve`，`--max-num-batched-tokens` 的默认值是 **2048**
   （`arg_utils.py`：≥160 GiB 的卡给 16384，≥70 GiB 给 8192，其余给 **2048**）。
3. 调度器每步是**贪心**的：第一个请求直接拿走 `min(剩余, 整个预算)`。
4. `_mamba_block_aligned_split()`（`vllm/v1/core/sched/scheduler.py:412`）会把
   prefill 中间块的结束位置**向下对齐到 block 的整数倍**：
   ```python
   if end < prefill_end:
       aligned_end = end // block_size * block_size
       if aligned_end > start or block_size <= max_prefill_tokens:
           end = aligned_end
   ```
   对新请求 `start = 0`，于是 `aligned_end = 0`，而 `block_size (1568) <= max_prefill_tokens (2048)`
   为真 → **`end = 0`** → `num_new_tokens == 0` → 该请求这一步被跳过（`continue`）。

**合起来：2048 的预算刚好只够放一个 1568 的块（剩 480 < 1568），
第二个长请求永远拿不到一个完整块。** 这解释了"两个长 prompt 串行、短 prompt 却能几十路并发"：
短 prompt 的 chunk 恰好结束在 `prefill_end`，走 `end < prefill_end` 为假的分支，**不受对齐约束**。

## 让它真并行要改两个参数

```
--long-prefill-token-threshold 1568      # 把每个请求每步的 chunk 上限压到一个 block
--max-num-batched-tokens 3136            # ≥ 2 × block，才留得出第二个块的预算
```

**实测确实并行了**：`num_requests_running = 2`、`waiting = 0`、
peak `kv_cache_usage_perc = 0.994`（两条各自持 ~125K 同时存在）。

## 但在这台机器上**不划算**

| 方案 | 2 × 125K 总墙钟 | 聚合 prefill |
|---|---|---|
| 默认（长 prompt 串行） | **162s** | ~1600 tok/s |
| 调参（真并行） | **219s** | ~1140 tok/s |

原因：把 chunk 上限压到 1568 后，每步只有两个很小的块，逐步开销占比变高，
聚合 prefill 反而掉了 ~30%，总时间比串行更差。

而且 `--max-num-batched-tokens` 一提就会**吃掉 ~0.9 GiB 激活+图内存**，
KV 池从 8.45 → 7.56 GiB，**单路 256K 直接起不来**（见
[02-vram-budget.md](02-vram-budget.md)），所以还要同步降 `max_model_len`。

> 顺便：这个现象在旧的 W4A16 部署里就被记录过一句话 ——
> 「长 prefill 会被 `max_num_batched_tokens=2048` 串行化」。
> 当时只记了现象；这里是它背后的完整机制（mamba block 对齐那一层）。

## 结论

- **容量上限**：fp8 下双并发每路 ~130K；4-bit KV 下每路 ~258K。
- **并行性**：默认参数下长 prompt 的 prefill 会串行（等价于排队），
  短 prompt 不受影响，仍可 32+ 路并发。
- **要不要强开并行**：本机实测不值。正确的用法是**按容量设上限**（每路 ~130K），
  让调度器去排队，而不是把 prefill 拆碎去抢并行。
- 若你的场景确实是"多条长 prompt 必须同时在算"（例如离线批量重算），
  那就上大显存卡 + 保住 `max_num_batched_tokens ≥ 2 × block`。

## 一个测量陷阱

**不要用相同的 prompt 测并发容量**：前缀缓存的 radix tree 会让两条相同前缀的请求
**共享同一批 KV block**，于是"2 × 134K 也能并排跑"这种假象就出来了
（TTFT 5.6s、KV 占用只算一份）。测容量的 prompt 必须**各不相同**（至少头一个 block 不同）。
