# Bonsai 2 27B (ternary 2-bit) on vLLM

把 PrismML 的 **Bonsai 2 27B（ternary −1/0/+1，g128，2-bit）** 从 GGUF 移植到 **vLLM**，
用 `prism_ternary` 量化插件跑起来，并给出**可复现的实测数据**。

> **TL;DR** — 在一张 RTX 4090 24 GB 上，把本地 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
> 转成 vLLM checkpoint 后：
> 单流解码 **86.8 → 74.1 tok/s**（0 → 33k 上下文），**与 llama.cpp 原生实现同档**；
> **短请求并发聚合 1027 tok/s @ 32 路**（llama.cpp 3 路只有 ~101）；
> 质量 280 题 **88.2%**（llama.cpp 同权重 87.9%），且**评测墙钟快 3.4×**；
> **256K 单路可用**（实测 262,089 token 请求成功），配 4-bit KV 可**双路 ~258K**。
>
> 这个仓库只包含**转换/校验/压测工具与工程笔记**，不含任何模型权重，也不含插件源码
> （插件版权属于其作者，见文末 Credits）。

---

## 1. 为什么值得做

Bonsai 2 27B 的权重是 ternary（每权重 4 种取值）+ 一个折叠进去的 Hadamard 旋转。
原始只有 PrismML 派生的 llama.cpp / MLX 运行时能解。

llama.cpp 那条路的短板不是速度，而是**服务化能力**：固定槽位、没有连续批处理、
没有 Paged KV / 前缀缓存、没有 OpenAI 兼容 API 与工具调用协议。而 vLLM 恰好长在这些地方。

移植之后实测的收益集中在**并发与工程性**，单流速度基本持平（见第 4 节）。

## 2. 移植的 4 个关键判断（都不显然）

| # | 判断 | 为什么 |
|---|---|---|
| 1 | **checkpoint 存"折叠过"的权重**，不存还原后的 base 权重 | 插件在前向对激活做 `(x·signs)@H/√1024`，与权重里已折进去的旋转相乘正好约掉，还原成 base 前向。官方 `convert.py` 的 `verify` 就是在断言 `dequant(stored) ≈ rotate(base, signs)` |
| 2 | 用 **HF 分体名**（`q/k/v_proj`、`in_proj_qkv/z`、`in_proj_b/a`、`gate/up_proj`…） | vLLM 的 `WeightsMapper` 是纯字符串替换 + `shard_id`，会自己融成 `qkv_proj`/`in_proj_qkvz`/`gate_up_proj`；`.weight` 与 `.scales` 都会被一并融合 |
| 3 | **GemmaRMSNorm 类 norm 要减 1** | GGUF 里存的是"有效乘子"（≈1.0），而 vLLM 做 `x*(1+w)`；`linear_attn.norm` 例外，不减 |
| 4 | **GDN 需要 vperm 重排**（`linear_num_value_heads=48 ≠ key_heads=16`） | `attn_qkv/gate/alpha/beta/A_log/dt/conv1d` 要按 `reshape(3,16,u).transpose` 重排；`ssm_out` 因为 `gdn_v_grouped=True` 不排 |

这四条都是靠"同基座的参考 checkpoint（HF 上的 AWQ 版）当 oracle"反推出来的，
不是猜的 —— 校验脚本 `tools/validate_vs_reference.py` 就是干这个的。

## 3. 快速开始

```bash
# 0) 依赖：vLLM ≥0.28（需要支持 qwen3_5 / GDN 混合架构）、python ≥3.12、ninja、CUDA toolkit
#    以及插件的 src 目录（见 Credits 里的 bonsai-vllm）
export VENV=/path/to/venv                         # 已装好 vLLM 的环境
export PLUGIN_SRC=/path/to/bonsai-vllm/src        # prism_ternary 插件源码

# 1) 装插件（不联网、不需要 build backend）
VENV=$VENV PLUGIN_SRC=$PLUGIN_SRC bash tools/install_prism_ternary.sh

# 2) GGUF -> vLLM checkpoint（约 4~6 分钟，峰值内存 ~12 GB）
python tools/convert_gguf_to_vllm.py \
    Ternary-Bonsai-2-27B-PTQ1_0.gguf  ./bonsai_vllm \
    --reference /path/to/Qwen3.8-27B-AWQ-INT4     # 可选，但强烈建议给（会跑逐张量校验）

# 3) 起服务（默认 fp8 KV / 256K / 单卡）
VENV=$VENV CKPT=./bonsai_vllm PORT=8000 bash tools/serve_vllm_bonsai.sh

# 4) 冒烟 + 压测
python tools/smoke_quality.py            # 算术/常识/中文/工具调用
python tools/bench_speed.py              # 深度-速度曲线（流式，TTFT 与 decode 分离）
python tools/bench_concurrency.py        # 1→128 并发聚合吞吐 + 延迟分位
```

转换产物：`model.safetensors`（分片）+ `config.json`（含 `quant_method: prism_ternary`
与 base64 编码的 Hadamard `signs`）+ tokenizer。

## 4. 实测（RTX 4090 24 GB，单卡独占，vLLM 0.28）

### 4.1 权重与显存账本（KV dtype = fp8）

| 项 | 实测 |
|---|---|
| checkpoint 盘上体积 | 9.42 GB / 1253 张量（402 折叠权重 + 402 scales + 449 稠密） |
| 加载后权重 | 8.97 GiB |
| weights + non-torch | 9.35 GiB |
| **peak activation** | **4.03 GiB**（∝ `--max-num-batched-tokens`） |
| **CUDA graph** | **1.75 GiB**（34 档 piecewise + 19 档 full） |
| KV 池 | 8.45 GiB → **265,157 token** |
| 合计 | ~23.1 / 23.99 GiB |

`--gpu-memory-utilization` 是"吃满"语义：除权重/激活/图之外全变 KV 池。
所以 23 GB 占用是设计结果，不是泄漏。**激活 + 图 ≈ 5.8 GiB 是引擎税**，
也是调 `--max-num-batched-tokens` 时会动的部分（把它从 2048 提到 8192 会多吃 ~0.9 GiB，
KV 池掉到 7.56 GiB，256K 直接起不来）。

### 4.2 单流速度（流式测量，TTFT 与 decode 分离）

| 上下文 | fp8 KV | turboquant 4-bit KV |
|---|---|---|
| ~0 | **86.8** | 85.2 |
| ~2.7k | 84.5 | 73.3 |
| ~11k | **81.4** | 51.8 |
| ~33k | **74.1** | 27.7 |

对照 llama.cpp 原生 PTQ1_0 实现：80（0 ctx）/ 66（32k）→ vLLM 在 fp8 下**略快**。

### 4.3 并发（短 prompt + 256 token 输出 + `ignore_eos` 定长，kv4）

| 并发 | 聚合 tok/s | 单路均 | p50 延迟 |
|---|---|---|---|
| 1 | 74.6 | 74.6 | 3.43s |
| 8 | 486.0 | 60.8 | 4.21s |
| 16 | 800.9 | 50.1 | 5.11s |
| **32** | **1027.1** | 32.1 | 7.96s |
| 48 | 732.6 | 15.3 | 12.99s |
| 64 | 863.2 | 13.5 | 13.11s |
| 128 | 875.5 | 6.8 | 25.73s |

甜点 **32 路**；KV 池在 **~45–48 路**打满（`running=45 / waiting=83`），
之后只涨延迟不涨吞吐。`max_num_seqs=128` 只是名义值。

### 4.4 长上下文容量

| 场景 | 结果 |
|---|---|
| 单路 256K（262,089 token） | ✅ 成功，TTFT **194.98s**（prefill ≈1344 tok/s） |
| 4 路 × 32K（agent 形态） | ✅ 4 路全部并行完成，墙钟 55.6s，聚合 prefill ≈2.3k tok/s |
| 2 路 × 200K（kv4） | ✅ 都完成（fp8 下 400K > 265K 池，根本装不下） |
| 2 路 × 256K | 需 KV≈518K：`turboquant_4bit_nc` 给出 518,589 token → **约 2 路满长** |

### 4.5 质量（280 题混合基准，temperature=0，non-thinking）

| 部署 | 总分 | GSM8K | 备注 |
|---|---|---|---|
| vLLM + fp8 KV | **88.2%** (247/280) | 95.7 | 评测墙钟 296s |
| vLLM + turboquant 4-bit KV | 87.9% (246/280) | 95.7 | 墙钟 343s |
| llama.cpp（同权重） | 87.9% (246/280) | 95.7 | 墙钟 1013s |

**4-bit KV 基本不吃质量**（±1 题，属噪声）；而 vLLM 因为能连续批处理并发请求，
评测墙钟快 **3.4×**。

## 5. 踩过的坑（都在代码注释里留了）

1. **必须开 CUDA graph**。`--enforce-eager` 只有 ~9 tok/s，去掉后 7× 到 ~70：
   小 batch 解码被 kernel 启动开销拖死。
2. **混合 GDN 模型开图捕获要求 `max_num_seqs ≤ 可用 mamba 块数`**（本配置 ~170），
   否则报 `max_num_seqs (256) exceeds available Mamba cache blocks (170)`。
3. **工具调用必须显式开** `--enable-auto-tool-choice --tool-call-parser qwen3_xml`。
   不开的话带 `tools` 的请求会被 vLLM 直接 **400**（基准里表现成"全部答错"，很容易误判成模型不行）。
   注意模板输出的是 Qwen XML（`<tool_call><function=NAME>`），对应 `qwen3_xml`，不是 hermes。
4. **`embed_tokens` 不能带 `.scales`**：它是稠密的，转换器若残留 `.scales` 会报
   `no module named 'embed_tokens.scales'`。
5. **长 prompt 的并发 prefill 会被调度卡住**，不是被 KV 卡住 —— 详见
   `docs/04-long-prefill-concurrency.md`（混合 GDN 的 mamba "align" 让 block 变成 1568/3072，
   与默认 2048 的 `max-num-batched-tokens` 叠加后，第二个长请求的 chunk 会被 floor 成 0）。
6. **`--gpu-memory-utilization` 别设 ≥0.94**：桌面非计算进程占着 ~1.5 GiB，
   会直接报 `Free memory ... less than desired GPU memory utilization`。上限约 0.93。
7. **TurboQuant 会强制退回 FlashAttention 2**（`not yet compatible with FlashAttention >= 3`）
   并要求每步反量化 KV → 深层解码掉 2.7×（33k：74.1 → 27.7）。换容量可以，换速度不行。
8. **测长上下文必须用流式**：非流式把 prefill 和 decode 混在一起，数字会完全失真。
9. **测并发必须 `ignore_eos: true` + 固定 max_tokens**：否则各请求提前停、长度不一，
   聚合与延迟都不可比。
10. **不要用相同 prompt 测并发**：前缀缓存会共享 block，得到"能开 2 路 130K"的假象。

## 6. 兼容性

| 项 | 要求 |
|---|---|
| GPU | `prism_ternary` 声明 `get_min_capability() = 80`；内核用 bf16 tensor-core MMA |
| 已验证 | **RTX 4090（sm_89）** —— 插件作者只在 RTX PRO 6000 Blackwell 上测过，本仓库补上了 Ada 上的验证 |
| KV dtype | `fp8`（推荐，FA3 原生支持） / `turboquant_4bit_nc`（换容量） |
| 参考 checkpoint | 同基座的 HF AWQ 版（仅用于校验，不参与推理） |

## 7. 仓库结构

```
tools/
  convert_gguf_to_vllm.py     GGUF -> vLLM checkpoint（含 Hadamard/signs/vperm/norm 偏移）
  validate_vs_reference.py    逐张量与参考 checkpoint 对齐校验（folded 余弦 + 稠密余弦）
  install_prism_ternary.sh    免构建后端安装 vLLM 插件（.pth + dist-info entry point）
  serve_vllm_bonsai.sh        起服务（KV dtype / 上下文 / batch tokens 全部可用 env 覆盖）
  smoke_quality.py            质量冒烟（算术 / 常识 / 中文 / 工具调用）
  bench_speed.py              深度-速度曲线（流式，TTFT 与 decode 分离）
  bench_concurrency.py        1→128 并发扫描（聚合吞吐 + p50/p90 + KV 峰值）
  probe_long_context.py       指定长度 / 路数的长上下文探针（带 /metrics 实时判定）
docs/
  01-porting-notes.md         移植细节：折叠权重、名字映射、vperm、norm 偏移
  02-vram-budget.md           显存账本与"引擎税"
  03-kv-dtype-tradeoffs.md    fp8 vs 4-bit KV 的容量/速度取舍（含为什么同为 4-bit 差一倍）
  04-long-prefill-concurrency.md  长 prompt 并发 prefill 为什么退化，以及怎么改、代价多大
```

## 8. Credits & 授权

- **权重**：Bonsai 2 27B 由 **PrismML** 发布（ternary 2-bit + 折叠 Hadamard）。
  本仓库**不分发权重**，只提供从你本地 GGUF 生成 vLLM checkpoint 的工具。
- **插件**：`prism_ternary` 来自 **fraserprice/bonsai-vllm**，版权归其作者。
  本仓库**不包含**插件源码，只提供安装脚本与 Ada 架构上的验证结论。
- **参考 checkpoint**：校验步骤需要同基座的公开 checkpoint（用于对照），请自行获取。
- 本仓库自有代码 / 文档：**MIT**（见 `LICENSE`）。

> 说明：本仓库的基准只用到自建的少量题目与公开推理接口，不分发第三方基准数据集。
