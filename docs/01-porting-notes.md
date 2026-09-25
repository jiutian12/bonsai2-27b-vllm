# 01 · 移植细节

目标：把 `Ternary-Bonsai-2-27B-PTQ1_0.gguf` 转成 vLLM 能加载的 checkpoint。
参考实现是 MLX 侧的 loader（GGUF → MLX），本工具把它等价地落到 vLLM 的命名与布局上。

## GGUF 里有什么

- **402 个折叠权重**：类型 id `143`（`PTQ1_0`，自定义，超出 stock gguf 的枚举范围），
  每 128 个权重一个 fp16 scale，权重是 ternary（每个 2 bit，16 个塞进一个 int32）。
- **折叠**：线性层的权重里已经乘进了一个 blockwise signed Hadamard 旋转（block = 1024），
  `token_embd` 是唯一需要**逆变换**的（`inverse_weight_names`）。
- **Hadamard 元数据**：`prism.hadamard.sign_widths = [5120, 6144, 17408]`（合计 28672）、
  `prism.hadamard.sign_values`（28672 个 ±1 的 fp32）、`prism.hadamard.block_size = 1024`、
  `prism.hadamard.gdn_v_grouped = true`。
- 其余 449 个是稠密张量（norm / conv1d / A_log / dt_bias / in_proj_a / in_proj_b / embedding）。

工具需要给 `gguf` 库补上两个自定义量化类型（`PTQ1_0 = 143`、`PQ2_0 = 147`）及其
`GGML_QUANT_SIZES`。注意这个 dict 被 `constants` / `gguf_reader` / `quants` **三个命名空间各按值导入了一份**，
必须三处都补，否则报 `143 is not a valid GGMLQuantizationType`。

## 判断 1：存折叠权重，不存还原后的 base 权重

推理时的复合是：

```
rotated = signed_hadamard(x, signs)     # = (x * signs) @ H / sqrt(1024)
y       = ternary_gemv(rotated, W_stored)

(x*signs)@H  ·  ((base*signs)@H)^T  ==  x @ base^T      # signs 抵消，H 正交归一
```

所以 **checkpoint 里就该存 GGUF 里那份折叠权重**，插件在激活侧做一次 Hadamard，
两者相乘正好还原 base 前向。官方 `convert.py` 的 `verify` 断言的就是
`dequant(stored) ≈ rotate(base, signs)`。

> 换句话说：转换器**不做数学**，只做「解包 + 改名 + 重排 + 拼装」。

## 判断 2：用 HF 分体名，不要自己融合

vLLM 对 `qwen3_5` 这类混合架构在模型侧就把 `q_proj/k_proj/v_proj` 融成 `qkv_proj`、
`gate_proj/up_proj` 融成 `gate_up_proj`、`in_proj_qkv/z` 融成 `in_proj_qkvz`、
`in_proj_b/a` 融成 `in_proj_ba`。它是靠 `hf_to_vllm_mapper`（`WeightsMapper`）做的，
而 `WeightsMapper` 是**纯字符串替换 + 打 `shard_id` 标记** —— 所以 `.weight` 与 `.scales`
会被**同一条规则一起搬运**，不需要在转换器里手工拼字节。

名字映射（GGUF stem → vLLM）：

```
attn_norm → input_layernorm          post_attention_norm → post_attention_layernorm
ffn_gate  → mlp.gate_proj            ffn_up              → mlp.up_proj
ffn_down  → mlp.down_proj            attn_q/k/v          → self_attn.q/k/v_proj
attn_output → self_attn.o_proj       attn_qkv            → linear_attn.in_proj_qkv
attn_gate → linear_attn.in_proj_z    ssm_out             → linear_attn.out_proj
ssm_alpha → linear_attn.in_proj_a    ssm_beta            → linear_attn.in_proj_b
ssm_conv1d → linear_attn.conv1d
```

## 判断 3：GemmaRMSNorm 要减 1

vLLM 里这些 norm 走 `x * (1 + w)`（Gemma 式），而 GGUF 存的是**有效乘子**。
用同基座参考 checkpoint 对齐验证：GGUF 的值集中在 ≈1.0 附近（0.97 / 0.78 / 1.23 / 1.94），
参考 checkpoint 的对应值集中在 ≈0.2 —— 差的就是这个 1。

需要减 1：`input_layernorm`、`post_attention_layernorm`、`self_attn.q_norm`、`k_norm`、`model.norm`。
**不减**：`linear_attn.norm`。

## 判断 4：GDN 的 vperm 重排

`linear_num_value_heads (48) ≠ linear_num_key_heads (16)` 时，GGUF 里这些张量的
head 排布与 vLLM 期望的不同，需要按 `reshape(3, 16, u).transpose(...)` 重排：
`attn_qkv`、`attn_gate`、`ssm_alpha`、`ssm_beta`、`ssm_a`(A_log)、`ssm_dt`(dt_bias)、`ssm_conv1d`。

例外：`ssm_out` 不排 —— 因为 `prism.hadamard.gdn_v_grouped = true`，它已经按 grouped 布局存好了。

> 这一条最容易错，也最容易验证：**排错了余弦会掉到 ≈0**（不再相关），
> 而不是"稍微低一点"。所以验证脚本里只要看余弦的量级就能判定。

## 怎么确认自己转对了

`tools/validate_vs_reference.py` 拿一份**同基座的参考 checkpoint**（例如 HF 上的 AWQ 版）
做两件事：

1. **折叠线性层**：把参考权重按 `rotate(base, signs)` 折一遍，与生成的 int32→float 反量化结果比
   逐行余弦。实测 **0.71–0.76**（官方 `convert.py` 用的阈值是 0.6）。
   这部分差值来自"2-bit ternary + 参考自身 4-bit"的叠加误差。
2. **稠密张量**：直接比余弦。实测：

| 张量 | 余弦 |
|---|---|
| `model.norm` | 0.99996 |
| `q_norm` / `k_norm` | 0.9998 |
| `linear_attn.norm` | 0.9996 |
| `A_log` / `dt_bias` | 1.00000 |
| `conv1d` | 0.9950 |
| `in_proj_a` / `in_proj_b` | 0.9939 / 0.9715 |

> 一个反直觉现象：`input_layernorm` 对参考 checkpoint 的余弦并不高 —— 不是转换错了，
> 而是**参考的 AWQ 把 per-channel scale 折进了 layer norm**，两边基准本就不同。
> 判定要看"值域分布是否符合 −1 之后的预期"，而不是单看余弦。

最后当然还要看**端到端**：`tools/smoke_quality.py` 跑算术/常识/中文/工具调用四项。
