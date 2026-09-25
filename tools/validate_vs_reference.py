#!/usr/bin/env python3
"""Lean validation of the built vLLM checkpoint vs the AWQ base (compressed-tensors)."""
import base64, json, os, numpy as np, torch, torch.nn.functional as Fn
from safetensors import safe_open

# usage: validate_vs_reference.py <built_ckpt> <reference_ckpt>
#   reference 用同基座的 HF checkpoint（例如 AWQ 版，compressed-tensors 格式）
MINE = os.environ.get("MINE") or (sys.argv[1] if len(sys.argv) > 1 else "bonsai_vllm")
AWQ = os.environ.get("AWQ") or (sys.argv[2] if len(sys.argv) > 2 else "")
BLOCK = 1024

cfg = json.load(open(MINE + "/config.json"))
SIGNS = {}
for w, enc in cfg["quantization_config"]["signs"].items():
    bits = np.unpackbits(np.frombuffer(base64.b64decode(enc), dtype=np.uint8))[:int(w)]
    SIGNS[int(w)] = np.where(bits == 1, -1.0, 1.0).astype(np.float32)


def hadamard(n):
    h = np.ones((1, 1), np.float32)
    two = np.array([[1., 1.], [1., -1.]], np.float32)
    while h.shape[0] < n:
        h = np.kron(h, two)
    return h

H_RAW = hadamard(BLOCK)
H_NORM = (H_RAW / np.sqrt(BLOCK)).astype(np.float32)


def rotate(w, signs):
    rows = w.shape[0]
    return ((w * signs).reshape(rows, -1, BLOCK) @ H_NORM).reshape(rows, -1)


def dequant(words, scales, rows, width):
    codes = np.empty((rows, width // 16, 16), np.uint8)
    for lane in range(16):
        codes[:, :, lane] = ((words >> (2 * lane)) & 3).astype(np.uint8)
    f = codes.reshape(rows, width // 128, 128).astype(np.float32) - 1.0
    return (f * scales[:, :, None]).reshape(rows, width)


midx = json.load(open(MINE + "/model.safetensors.index.json"))["weight_map"]
aidx = json.load(open(AWQ + "/model.safetensors.index.json"))["weight_map"]


def mine(name):
    with safe_open(f"{MINE}/{midx[name]}", framework="pt") as f:
        return f.get_tensor(name)


def awq_t(key):
    with safe_open(f"{AWQ}/{aidx[key]}", framework="pt") as f:
        return f.get_tensor(key)


def cands(name):
    out = [name]
    if name.startswith("model."):
        out.append("model.language_model." + name[len("model."):])
    return out


def awq_dense(name):
    for c in cands(name):
        if c + ".weight" in aidx:
            return awq_t(c + ".weight").float().numpy()
    return None


def awq_quant(name):
    for c in cands(name):
        if c + ".weight_packed" in aidx:
            packed = awq_t(c + ".weight_packed").numpy().astype(np.int32)
            scale = awq_t(c + ".weight_scale").float().numpy().astype(np.float32)
            zp = awq_t(c + ".weight_zero_point").numpy().astype(np.int32)
            shape = awq_t(c + ".weight_shape").numpy()
            out_dim, in_dim = int(shape[0]), int(shape[1])
            g = in_dim // scale.shape[1]
            nib = np.arange(8, dtype=np.int32) * 4
            codes = ((packed[:, :, None] >> nib) & 0xF).astype(np.int32).reshape(out_dim, in_dim)
            zpv = ((zp[:, :, None] >> nib) & 0xF).astype(np.int32).reshape(zp.shape[0] * 8, scale.shape[1])[:out_dim]
            w = (codes.reshape(out_dim, in_dim // g, g) - zpv[:, :, None]) * scale[:, :, None]
            return w.reshape(out_dim, in_dim).astype(np.float32)
    return None


def report_folded(name):
    mw = mine(name + ".weight").numpy()
    ms = mine(name + ".scales").numpy()
    rows, wid = mw.shape[0], mw.shape[1] * 16
    ref = awq_quant(name)
    refr = rotate(ref, SIGNS[wid])
    got = dequant(mw, ms, rows, wid)
    r = min(rows, 4096)
    c = Fn.cosine_similarity(torch.from_numpy(got[:r]), torch.from_numpy(np.ascontiguousarray(refr[:r])), dim=1).mean().item()
    print(f"  {name:44s} rows={rows:6d} in={wid:5d}  row-cos={c:.4f}", flush=True)


print("=== FOLDED modules: dequant(mine) vs rotate(AWQ base) ===", flush=True)
for name in ["model.layers.3.self_attn.k_proj",
             "model.layers.0.linear_attn.in_proj_z",
             "model.layers.3.self_attn.v_proj",
             "model.layers.3.self_attn.q_proj",
             "model.layers.0.mlp.gate_proj"]:
    try:
        report_folded(name)
    except Exception as e:
        print(f"  {name} ERROR {type(e).__name__}: {e}", flush=True)

print("\n=== DENSE probes: mine vs AWQ ===", flush=True)
for name in ["model.layers.0.input_layernorm.weight",
             "model.layers.3.self_attn.q_norm.weight",
             "model.norm.weight",
             "model.layers.0.linear_attn.norm.weight",
             "model.layers.0.linear_attn.A_log",
             "model.layers.0.linear_attn.dt_bias",
             "model.layers.0.linear_attn.in_proj_a.weight",
             "model.layers.0.linear_attn.in_proj_b.weight",
             "model.layers.0.linear_attn.conv1d.weight"]:
    m = mine(name).float().numpy().ravel()
    r = awq_dense(name)
    if r is None:
        print(f"  {name:46s} (no AWQ dense)", flush=True); continue
    r = r.ravel()
    n = min(len(m), len(r))
    c = Fn.cosine_similarity(torch.from_numpy(m[:n]).unsqueeze(0), torch.from_numpy(r[:n]).unsqueeze(0)).item()
    print(f"  {name:46s} max|d|={np.abs(m[:n]-r[:n]).max():.4f} cos={c:.5f}", flush=True)

print("\n=== EMBEDDING ===", flush=True)
mw = mine("model.embed_tokens.weight").float().numpy()
rw = awq_dense("model.embed_tokens")
n = min(4096, mw.shape[0])
c = Fn.cosine_similarity(torch.from_numpy(mw[:n]), torch.from_numpy(rw[:n]), dim=1).mean().item()
print(f"  dense embed row-cos vs AWQ = {c:.4f}", flush=True)
print("DONE", flush=True)
