#!/usr/bin/env python3
"""Convert local Bonsai 2 27B (PTQ1_0) GGUF -> vLLM checkpoint (prism_ternary plugin).

Mirrors the reference MLX loader (_bonsai/runtime__runtime.py):
  * GGUF stem -> HF/vLLM param name mapping  (vLLM fuses q/k/v, gate/up,
    in_proj_qkv/z, in_proj_b/a itself)
  * GDN vperm reorder when linear_num_value_heads != linear_num_key_heads
  * unit-offset (+1) on input/post_attention/q/k/model norms (GemmaRMSNorm)
  * folded PTQ1_0 tensors -> int32 words + fp16 scales (transcode), stored folded
  * embedding stored dense (bf16) via inverse FWHT
Folded tensors are stored AS-IS; the plugin applies signs+Hadamard on the
activation side, which composes back to the base forward (verified vs AWQ).

usage:
  python convert_gguf_to_vllm.py Ternary-Bonsai-2-27B-PTQ1_0.gguf out_dir \
         [--reference /path/to/Qwen3.8-27B-AWQ-INT4] [--no-validate]
"""
import base64, json, os, shutil, sys, time
from collections import OrderedDict

import numpy as np
import torch
from safetensors.torch import save_file

import gguf
from gguf.constants import GGMLQuantizationType as _Base
import gguf.constants as gc, gguf.gguf_reader as gr, gguf.quants as qu

_members = [f"{m.name} = {m.value}" for m in _Base] + ["PTQ1_0 = 143", "PQ2_0 = 147"]
_ns = {}
exec("from enum import IntEnum\nclass ExtEnum(IntEnum):\n" + "\n".join("    " + d for d in _members), _ns)
ExtEnum = _ns["ExtEnum"]
gc.GGML_QUANT_SIZES[ExtEnum.PTQ1_0] = (128, 28)
gc.GGML_QUANT_SIZES[ExtEnum.PQ2_0] = (128, 34)
gr.GGML_QUANT_SIZES = gc.GGML_QUANT_SIZES
qu.GGML_QUANT_SIZES = gc.GGML_QUANT_SIZES
gr.GGMLQuantizationType = ExtEnum

# ---- paths ----------------------------------------------------------------
#   usage: convert_gguf_to_vllm.py <bonsai.gguf> <out_dir> [--reference <ckpt>] [--no-validate]
#   env  : BONSAI_GGUF / BONSAI_OUT / BONSAI_REF
#   --reference 指向同基座的参考 checkpoint（例如 HF 上的 AWQ 版）时会额外跑一遍
#   逐张量对齐校验；不提供就跳过校验（等价 --no-validate）。
_argv = sys.argv[1:]


def _opt(flag, default=""):
    if flag in _argv:
        i = _argv.index(flag)
        if i + 1 < len(_argv):
            return _argv[i + 1]
    return default


_ref = _opt("--reference")
_pos = [a for a in _argv if not a.startswith("--") and a != _ref]
GGUF = os.environ.get("BONSAI_GGUF") or (_pos[0] if _pos else "Ternary-Bonsai-2-27B-PTQ1_0.gguf")
OUT = os.environ.get("BONSAI_OUT") or (_pos[1] if len(_pos) > 1 else "bonsai_vllm")
AWQ = os.environ.get("BONSAI_REF") or _ref
BLOCK, GROUP, LANES = 1024, 128, 16
DO_VALIDATE = bool(AWQ) and "--no-validate" not in _argv


def transcode(raw, shape, source):
    rows, width = shape
    blocks = rows * width // 128
    sizes = {"PQ2_0": 34, "PTQ1_0": 28}
    data = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, sizes[source])
    scale_bytes = data[:, :2] if source == "PQ2_0" else data[:, 26:28]
    scales = scale_bytes.copy().view("<f2").reshape(rows, width // 128)
    if not np.isfinite(scales).all():
        raise ValueError("non-finite scale")
    if source == "PQ2_0":
        words = data[:, 2:].copy().view("<u4").reshape(rows, width // 16)
    else:
        pieces = []
        for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
            packed = data[:, lo:hi].astype(np.uint16)
            for trit in range(count):
                rem = (packed * (3 ** trit)) & 255
                pieces.append(((rem * 3) >> 8).astype(np.uint8))
        codes = np.concatenate(pieces, axis=1)     # (blocks,128) uint8, values 0..3
        cr = codes.reshape(rows, width // 16, 16)
        # pack 16 2-bit codes into one uint32 little-endian (lane i -> bits 2i)
        b = (cr[..., 0::4] | (cr[..., 1::4] << 2) | (cr[..., 2::4] << 4) | (cr[..., 3::4] << 6)).astype(np.uint8)
        words = b.reshape(rows, width // 16, 4).view("<u4").reshape(rows, width // 16)
    return np.ascontiguousarray(words), np.ascontiguousarray(scales), np.ascontiguousarray(-scales)


def dequant(words, scales, rows, width):
    codes = np.empty((rows, width // 16, 16), dtype=np.uint8)
    for lane in range(16):
        codes[:, :, lane] = ((words >> (2 * lane)) & 3).astype(np.uint8)
    f = codes.reshape(rows, width // 128, 128).astype(np.float32) - 1.0
    return (f * scales[:, :, None].astype(np.float32)).reshape(rows, width)


def hadamard(n):
    h = np.ones((1, 1), dtype=np.float32)
    two = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.kron(h, two)
    return h


H_RAW = hadamard(BLOCK)
H_NORM = (H_RAW / np.sqrt(BLOCK)).astype(np.float32)


t0 = time.time()
r = gguf.GGUFReader(GGUF)
fd = {k: v.contents() for k, v in r.fields.items()}
SIGNS, off = {}, 0
for w in [int(x) for x in fd["prism.hadamard.sign_widths"]]:
    SIGNS[w] = np.asarray(fd["prism.hadamard.sign_values"][off:off + w], dtype=np.float32)
    off += w
assert off == len(fd["prism.hadamard.sign_values"])
print("sign widths:", list(SIGNS), " block:", int(fd["prism.hadamard.block_size"]),
      " tensors:", len(r.tensors))


def rotate(w, signs):
    rows = w.shape[0]
    return ((w.astype(np.float32) * signs).reshape(rows, -1, BLOCK) @ H_NORM).reshape(rows, -1)


def fwht_inverse(v):
    shape = v.shape
    out = (v.astype(np.float32).reshape(-1, BLOCK) @ H_RAW) * (1.0 / np.sqrt(BLOCK))
    return (out.reshape(shape) * SIGNS[shape[-1]]).astype(np.float32)


G = lambda k, d=None: fd.get("qwen35." + k, d)
HID, FFN = int(G("embedding_length")), int(G("feed_forward_length"))
NV, NK = int(G("ssm.time_step_rank")), int(G("ssm.group_count"))
HD, HK = int(G("ssm.inner_size")) // NV, int(G("ssm.state_size"))
qk = 2 * NK * HK
print(f"HID={HID} FFN={FFN} NV={NV} NK={NK} HD={HD} HK={HK} qk={qk}")
T = {t.name: t for t in r.tensors}


def vperm(unit):
    return np.arange(NV * unit).reshape(NV // NK, NK, unit).transpose(1, 0, 2).reshape(-1)


V_HD, V_1 = vperm(HD), vperm(1)


def perm_for(rows, stem):
    """Row permutation GGUF->HF layout; returns int array (or None if identity)."""
    if NV == NK:
        return None
    if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
        idx = np.concatenate([np.arange(qk), qk + V_HD])
        return idx if len(idx) == rows else None
    if stem == "attn_gate.weight":
        return V_HD
    if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
        return V_1
    return None


BLK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
GLOBAL_MAP = {"output.weight": "lm_head.weight",
              "output_norm.weight": "model.norm.weight",
              "token_embd.weight": "model.embed_tokens.weight"}
OFFSET_NORMS = {"attn_norm.weight", "post_attention_norm.weight",
                "attn_q_norm.weight", "attn_k_norm.weight"}

tensors = OrderedDict()
stats = {"folded": 0, "dense": 0}

print("=== building tensors ===")
for tname, t in T.items():
    if tname in GLOBAL_MAP:
        base = GLOBAL_MAP[tname]
        stem = tname
    elif tname.startswith("blk."):
        _, lidx, stem = tname.split(".", 2)
        base = "model.layers.%s.%s" % (lidx, BLK_MAP[stem])
    else:
        raise SystemExit("unmapped tensor " + tname)

    if t.tensor_type.name in ("PTQ1_0", "PQ2_0"):
        out_dim, in_dim = int(t.shape[1]), int(t.shape[0])
        words, scales, _ = transcode(t.data.tobytes(), (out_dim, in_dim), t.tensor_type.name)
        perm = perm_for(out_dim, stem)
        if perm is not None:
            words, scales = words[perm], scales[perm]
        tensors[base] = torch.from_numpy(np.ascontiguousarray(words.astype(np.int32)))
        tensors[base.rsplit(".", 1)[0] + ".scales"] = torch.from_numpy(np.ascontiguousarray(scales.astype(np.float16)))
        stats["folded"] += 1
        continue

    # ---- dense ----
    if stem in ("ssm_alpha.weight", "ssm_beta.weight"):
        out_dim, in_dim = int(t.shape[1]), int(t.shape[0])
        a = torch.frombuffer(bytearray(t.data.tobytes()), dtype=torch.bfloat16).reshape(out_dim, in_dim)
        perm = perm_for(out_dim, stem)
        if perm is not None:
            a = a[torch.from_numpy(perm).long()]
        tensors[base] = a.contiguous()
        stats["dense"] += 1
        continue

    arr = np.asarray(t.data).copy()
    perm = perm_for(arr.shape[0], stem)
    if perm is not None:
        arr = arr[perm]
    if stem == "ssm_a":
        if not (arr < 0).all():
            raise SystemExit("ssm_a not all negative")
        arr = np.log(-arr).astype(np.float32)
    elif stem == "ssm_conv1d.weight":
        out_dim, in_dim = int(t.shape[1]), int(t.shape[0])
        arr = np.ascontiguousarray(arr).reshape(out_dim, in_dim)[:, None, :]   # (out,1,kernel)
    elif stem in OFFSET_NORMS or tname == "output_norm.weight":
        arr = arr.astype(np.float32) - 1.0
    tensors[base] = torch.from_numpy(np.ascontiguousarray(arr))
    stats["dense"] += 1

print("folded modules:", stats["folded"], " dense:", stats["dense"], " total:", len(tensors))

# ---- embedding (dense bf16, inverse FWHT of dequantised folded rows) ----
print("=== embedding (inverse FWHT) ===")
te = T["token_embd.weight"]
vout, vin = int(te.shape[1]), int(te.shape[0])
ew, es, _ = transcode(te.data.tobytes(), (vout, vin), "PTQ1_0")
emb = np.empty((vout, vin), dtype=np.float16)
step = 8192
for s in range(0, vout, step):
    e = min(vout, s + step)
    emb[s:e] = fwht_inverse(dequant(ew[s:e], es[s:e], e - s, vin)).astype(np.float16)
emb_t = torch.from_numpy(emb)
tensors["model.embed_tokens.weight"] = emb_t
# the folded pass above also emitted a (stale) .scales for the embedding; the
# embedding is unquantized in vLLM, so drop it.
tensors.pop("model.embed_tokens.scales", None)
print("embedding done  %.1fs" % (time.time() - t0))

# ---------------- save ----------------
os.makedirs(OUT, exist_ok=True)


def shard_of(name):
    if name.startswith("model.layers."):
        return 1 if int(name.split(".")[2]) < 32 else 2
    return 0


shard_names = {0: "model-00001-of-00003.safetensors",
               1: "model-00002-of-00003.safetensors",
               2: "model-00003-of-00003.safetensors"}
weight_map, total = {}, 0
for sh, fn in shard_names.items():
    sub = {k: v.contiguous() for k, v in tensors.items() if shard_of(k) == sh}
    if not sub:
        continue
    sz = sum(v.numel() * v.element_size() for v in sub.values())
    save_file(sub, os.path.join(OUT, fn), metadata={"format": "pt"})
    weight_map.update({k: fn for k in sub})
    total += sz
    print("  wrote %s  %d tensors  %.2f GB" % (fn, len(sub), sz / 1e9))
    del sub

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(OUT + "/model.safetensors.index.json", "w"), indent=2)

# ---------------- config.json ----------------
tc = dict(json.load(open(AWQ + "/config.json"))["text_config"])
tc["mtp_num_hidden_layers"] = 0
tc["mtp_use_dedicated_embeddings"] = False
tc.pop("dtype", None)
lt = ["linear_attention"] * 64
for i in range(3, 64, 4):
    lt[i] = "full_attention"
tc["layer_types"] = lt

cfg = OrderedDict()
cfg["architectures"] = ["Qwen3_5ForCausalLM"]
cfg["model_type"] = "qwen3_5_text"
cfg["tie_word_embeddings"] = False
for k in sorted(tc):
    cfg[k] = tc[k]
cfg["quantization_config"] = {
    "quant_method": "prism_ternary", "bits": 2, "group_size": GROUP,
    "hadamard_block": BLOCK,
    "signs": {str(w): base64.b64encode(np.packbits(v < 0).tobytes()).decode() for w, v in SIGNS.items()},
}
json.dump(cfg, open(OUT + "/config.json", "w"), indent=2)

for aux in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "vocab.json", "merges.txt"):
    src = os.path.join(AWQ, aux)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(OUT, aux))

print("WROTE %s  total %.2f GB  (%.1fs)" % (OUT, total / 1e9, time.time() - t0))

# ---------------- validation vs AWQ (same base model) ----------------
if DO_VALIDATE:
    try:
        from safetensors import safe_open
        import torch.nn.functional as Fn
        aidx = json.load(open(AWQ + "/model.safetensors.index.json"))["weight_map"]

        def awq_key(name):
            k = ("model.language_model." + name[len("model."):]) if name.startswith("model.") else name
            return k if k in aidx else (name if name in aidx else None)

        def awq_t(key):
            with safe_open(f"{AWQ}/{aidx[key]}", framework="pt") as f:
                return f.get_tensor(key).float().numpy()

        def awq_folded(name):
            key = awq_key(name)
            qw = awq_t(key + ".qweight").astype(np.int64)
            qz = awq_t(key + ".qzeros").astype(np.int64)
            sc = awq_t(key + ".scales").astype(np.float32)
            in_dim = qw.shape[0]
            g = in_dim // qz.shape[0]
            shift = np.arange(8, dtype=np.int64) * 4
            w = ((qw[:, :, None] >> shift) & 0xF).reshape(in_dim, -1)
            z = ((qz[:, :, None] >> shift) & 0xF).reshape(in_dim // g, -1)
            z = np.repeat(z, g, axis=0)
            s = np.repeat(sc, g, axis=0)
            return ((w - z).astype(np.float32) * s).T   # (out, in)

        print("=== VALIDATION vs AWQ base ===")
        ref = awq_t(awq_key("model.embed_tokens.weight"))
        n = min(4096, emb_t.shape[0])
        cos = Fn.cosine_similarity(emb_t[:n].float(), torch.from_numpy(ref[:n]), dim=1).mean().item()
        print(f"  embed_tokens  row-cos vs AWQ            = {cos:.4f}")

        for name in ["model.layers.0.mlp.gate_proj",
                     "model.layers.3.self_attn.q_proj",
                     "model.layers.3.self_attn.o_proj",
                     "model.layers.0.linear_attn.in_proj_qkv",
                     "model.layers.0.linear_attn.in_proj_z",
                     "model.layers.0.linear_attn.out_proj",
                     "model.layers.0.mlp.down_proj"]:
            mw = tensors[name + ".weight"]
            ms = tensors[name + ".scales"]
            rows, wid = mw.shape[0], mw.shape[1] * 16
            refr = rotate(awq_folded(name), SIGNS[wid])
            got = dequant(mw.numpy(), ms.numpy(), rows, wid)
            c = Fn.cosine_similarity(torch.from_numpy(got), torch.from_numpy(refr), dim=1).mean().item()
            print(f"  {name:44s} row-cos vs rotate(AWQ) = {c:.4f}")
    except Exception as e:
        import traceback
        print("VALIDATION FAILED:", e)
        traceback.print_exc()

