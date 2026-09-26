#!/usr/bin/env python3
"""Build a vision+video-capable Qwen3.5 checkpoint (e.g. bonsai_vllm_vl) from:
   - an existing *text-only* vLLM checkpoint  (model.* / lm_head.*, from
     convert_gguf_to_vllm.py)   <-- "src"
   - the llama.cpp mmproj (vision tower + merger): *.mmproj-Q8_0.gguf  <-- "mmproj"

Usage:
    python tools/vis_convert.py <src_checkpoint_dir> <dst_checkpoint_dir> <mmproj.gguf>

    e.g. python tools/vis_convert.py ~/bonsai_vllm ~/bonsai_vllm_vl \
        ~/bonsai/models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf

Output layout follows HF Qwen3.5-VL naming so that vLLM 0.28
`Qwen3_5ForConditionalGeneration` loads it directly:
    model.language_model.*   (LLM, ternary packing preserved)
    lm_head.*
    model.visual.*           (vision tower + merger, bf16)

The resulting checkpoint supports BOTH image and video input via the
OpenAI-compatible API (see README, "多模态（图像 / 视频）支持").  Video is
enabled by the `temporal_patch_size` field and the `model.visual.*` tower;
the vLLM Qwen3.5-VL processor handles frame sampling (fps / max frames)
out of the box.
"""
import argparse
import json
import mmap
import os
import shutil
import struct

import numpy as np
import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------- GGUF v3

def _u32(d, p):
    return struct.unpack_from("<I", d, p)[0], p + 4


def _u64(d, p):
    return struct.unpack_from("<Q", d, p)[0], p + 8


def _str(d, p):
    n, p = _u64(d, p)
    return d[p:p + n].decode("utf-8", "replace"), p + n


def _val(d, p, t):
    if t == 0:
        return struct.unpack_from("<B", d, p)[0], p + 1
    if t == 1:
        return struct.unpack_from("<b", d, p)[0], p + 1
    if t == 2:
        return struct.unpack_from("<H", d, p)[0], p + 2
    if t == 3:
        return struct.unpack_from("<h", d, p)[0], p + 2
    if t == 4:
        return struct.unpack_from("<I", d, p)[0], p + 4
    if t == 5:
        return struct.unpack_from("<i", d, p)[0], p + 4
    if t == 6:
        return struct.unpack_from("<f", d, p)[0], p + 4
    if t == 7:
        return struct.unpack_from("<B", d, p)[0] != 0, p + 1
    if t == 8:
        return _str(d, p)
    if t == 10:
        return struct.unpack_from("<Q", d, p)[0], p + 8
    if t == 11:
        return struct.unpack_from("<q", d, p)[0], p + 8
    if t == 12:
        return struct.unpack_from("<d", d, p)[0], p + 8
    if t == 9:
        et, p = _u32(d, p)
        cnt, p = _u64(d, p)
        out = []
        for _ in range(cnt):
            v, p = _val(d, p, et)
            out.append(v)
        return out, p
    raise ValueError("unsupported ggml type %d" % t)


def dequant(raw, qtype, n):
    """Return float32 ndarray of n elements."""
    if qtype == 0:  # F32
        return np.frombuffer(raw[: 4 * n], dtype=np.float32).copy()
    if qtype == 1:  # F16
        return np.frombuffer(raw[: 2 * n], dtype=np.float16).astype(np.float32)
    if qtype == 8:  # Q8_0: blocks of 32 -> fp16 scale + 32 int8
        bs = 32
        nb = (n + bs - 1) // bs
        out = np.empty(nb * bs, dtype=np.float32)
        for b in range(nb):
            off = b * 34
            d = np.frombuffer(raw[off:off + 2], dtype=np.float16)[0]
            q = np.frombuffer(raw[off + 2:off + 34], dtype=np.int8)
            out[b * bs:(b + 1) * bs] = q.astype(np.float32) * float(d)
        return out[:n]
    raise ValueError("unhandled ggml type %d" % qtype)


def load_gguf(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    assert mm[:4] == b"GGUF", "not a gguf file"
    _, p = _u32(mm, 4)
    n_t, p = _u64(mm, p)
    n_kv, p = _u64(mm, p)
    meta = {}
    for _ in range(n_kv):
        k, p = _str(mm, p)
        t, p = _u32(mm, p)
        v, p = _val(mm, p, t)
        meta[k] = v
    infos = []
    for _ in range(n_t):
        name, p = _str(mm, p)
        nd, p = _u32(mm, p)
        dims = []
        for _ in range(nd):
            x, p = _u64(mm, p)
            dims.append(x)
        ty, p = _u32(mm, p)
        off, p = _u64(mm, p)
        infos.append((name, tuple(dims), ty, off))
    align = meta.get("general.alignment", 32)
    data_start = (p + align - 1) // align * align
    tensors = {}
    for name, dims, ty, off in infos:
        n = int(np.prod(dims)) if dims else 1
        start = data_start + off
        if ty == 0:
            nb = 4 * n
        elif ty == 1:
            nb = 2 * n
        elif ty == 8:
            nb = ((n + 31) // 32) * 34
        else:
            raise ValueError("type %d on %s" % (ty, name))
        tensors[name] = (dequant(mm[start:start + nb], ty, n).reshape(dims), ty)
    mm.close()
    return meta, tensors


# ------------------------------------------------------- vision renaming

def build_vision(meta, gg):
    """gg: name -> (float32 ndarray, qtype). Returns dict of torch bf16 tensors."""
    out = {}
    V = "model.visual."

    def t(name, arr):
        out[name] = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)

    def tr(name, gname, shape):
        """mmproj stores weight data in torch (out,in) row-major order but
        declares dims swapped -> reshape (NOT transpose) to the torch shape.
        Verified against the original W4A16 HF checkpoint: cosine == 1.0000."""
        t(name, gg[gname][0].reshape(shape))

    # patch_embed: two temporal slices -> Conv3d (1152, 3, 2, 16, 16)
    w0 = gg["v.patch_embd.weight"][0].reshape(1152, 3, 16, 16)
    w1 = gg["v.patch_embd.weight.1"][0].reshape(1152, 3, 16, 16)
    t(V + "patch_embed.proj.weight", np.stack([w0, w1], axis=2))
    t(V + "patch_embed.proj.bias", gg["v.patch_embd.bias"][0])

    # pos_embed: Embedding (2304, 1152)
    tr(V + "pos_embed.weight", "v.position_embd.weight", (2304, 1152))

    H = int(meta.get("clip.vision.embedding_length", 1152))
    F = int(meta.get("clip.vision.feed_forward_length", 4304))
    depth = int(meta.get("clip.vision.block_count", 27))
    for i in range(depth):
        b = "v.blk.%d." % i
        p = V + "blocks.%d." % i
        tr(p + "attn.qkv.weight", b + "attn_qkv.weight", (3 * H, H))
        t(p + "attn.qkv.bias", gg[b + "attn_qkv.bias"][0])
        tr(p + "attn.proj.weight", b + "attn_out.weight", (H, H))
        t(p + "attn.proj.bias", gg[b + "attn_out.bias"][0])
        tr(p + "mlp.linear_fc1.weight", b + "ffn_up.weight", (F, H))
        t(p + "mlp.linear_fc1.bias", gg[b + "ffn_up.bias"][0])
        tr(p + "mlp.linear_fc2.weight", b + "ffn_down.weight", (H, F))
        t(p + "mlp.linear_fc2.bias", gg[b + "ffn_down.bias"][0])
        t(p + "norm1.weight", gg[b + "ln1.weight"][0])
        t(p + "norm1.bias", gg[b + "ln1.bias"][0])
        t(p + "norm2.weight", gg[b + "ln2.weight"][0])
        t(p + "norm2.bias", gg[b + "ln2.bias"][0])

    # merger (projector): mm.0 -> linear_fc1, mm.2 -> linear_fc2
    OUT = int(meta.get("clip.vision.projection_dim", 5120))
    MS = int(meta.get("clip.vision.spatial_merge_size", 2))
    MG = H * MS * MS
    tr(V + "merger.linear_fc1.weight", "mm.0.weight", (MG, MG))
    t(V + "merger.linear_fc1.bias", gg["mm.0.bias"][0])
    tr(V + "merger.linear_fc2.weight", "mm.2.weight", (OUT, MG))
    t(V + "merger.linear_fc2.bias", gg["mm.2.bias"][0])
    # 1152-dim norm from llama.cpp's post_ln == HF merger.ln_q
    t(V + "merger.norm.weight", gg["v.post_ln.weight"][0])
    t(V + "merger.norm.bias", gg["v.post_ln.bias"][0])
    return out


# --------------------------------------------------------------- config

def build_config(meta, src_cfg, tok_cfg):
    from transformers.models.qwen3_5 import Qwen3_5Config

    def tid(name):
        dec = tok_cfg.get("added_tokens_decoder", {})
        for k, v in dec.items():
            if v.get("content") == name:
                if "id" in v:
                    return int(v["id"])
                try:
                    return int(k)
                except (TypeError, ValueError):
                    pass
        return None

    text_cfg = dict(src_cfg)
    vision_cfg = {
        "model_type": "qwen3_5_vision",
        "depth": int(meta.get("clip.vision.block_count", 27)),
        "hidden_size": int(meta.get("clip.vision.embedding_length", 1152)),
        "intermediate_size": int(meta.get("clip.vision.feed_forward_length", 4304)),
        "num_heads": int(meta.get("clip.vision.attention.head_count", 16)),
        "in_channels": 3,
        "patch_size": int(meta.get("clip.vision.patch_size", 16)),
        "spatial_merge_size": int(meta.get("clip.vision.spatial_merge_size", 2)),
        "temporal_patch_size": 2,
        "out_hidden_size": int(meta.get("clip.vision.projection_dim", 5120)),
        "num_position_embeddings": 2304,
        "image_size": int(meta.get("clip.vision.image_size", 768)),
        "layer_norm_eps": float(meta.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
        "deepstack_visual_indexes": [],
    }
    cfg = Qwen3_5Config(
        text_config=text_cfg,
        vision_config=vision_cfg,
        image_token_id=tid("mask"),
        video_token_id=tid("frame"),
        vision_start_token_id=tid(""),
        vision_end_token_id=tid(""),
        tie_word_embeddings=False,
    )
    cfg.architectures = ["Qwen3_5ForConditionalGeneration"]
    return cfg


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Build a vision+video-capable Qwen3.5 vLLM checkpoint "
                    "(from a text-only checkpoint + the mmproj GGUF).")
    ap.add_argument("src", help="existing text-only vLLM checkpoint dir (model.* / lm_head.*)")
    ap.add_argument("dst", help="output vision+video checkpoint dir")
    ap.add_argument("mmproj", help="path to *.mmproj-Q8_0.gguf (vision tower + merger)")
    args = ap.parse_args()
    SRC, DST, MMP = args.src, args.dst, args.mmproj

    os.makedirs(DST, exist_ok=True)

    print("[1/4] reading mmproj gguf ...")
    meta, gg = load_gguf(MMP)
    print("      tensors:", len(gg), "projector:", meta.get("clip.projector_type"))

    print("[2/4] building vision tensors (bf16) ...")
    vis = build_vision(meta, gg)
    total = sum(int(v.numel()) for v in vis.values())
    print("      vision tensors:", len(vis), "params: %.1fM" % (total / 1e6))

    print("[3/4] copying + renaming text weights ...")
    src_cfg = json.load(open(os.path.join(SRC, "config.json")))
    tok_cfg = json.load(open(os.path.join(SRC, "tokenizer_config.json")))
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wmap = idx["weight_map"]

    shards = {}
    for name, shard in wmap.items():
        shards.setdefault(shard, []).append(name)

    new_wmap = {}
    n_renamed = 0
    n_total = len(shards) + 1
    for i, shard in enumerate(sorted(shards), start=1):
        sd = load_file(os.path.join(SRC, shard))
        out = {}
        for k, v in sd.items():
            if k.startswith("model."):
                nk = "model.language_model." + k[len("model."):]
                n_renamed += 1
            else:
                nk = k
            out[nk] = v
        oname = "model-%05d-of-%05d.safetensors" % (i, n_total)
        save_file(out, os.path.join(DST, oname), metadata={"format": "pt"})
        for k in out:
            new_wmap[k] = oname
        print("      wrote", oname, len(out), "tensors")
        del sd, out

    print("      renamed %d model.* -> model.language_model.*" % n_renamed)

    vshard = "model-%05d-of-%05d.safetensors" % (n_total, n_total)
    save_file(vis, os.path.join(DST, vshard), metadata={"format": "pt"})
    for k in vis:
        new_wmap[k] = vshard
    print("      wrote", vshard, len(vis), "vision tensors")

    idx_out = {"metadata": {"total_size": 0}, "weight_map": new_wmap}
    json.dump(idx_out, open(os.path.join(DST, "model.safetensors.index.json"), "w"),
              indent=2)

    print("[4/4] writing config + tokenizer ...")
    cfg = build_config(meta, src_cfg, tok_cfg)
    cfg.save_pretrained(DST)
    # keep vLLM-friendly raw json too
    with open(os.path.join(DST, "config.json"), "w") as f:
        json.dump(json.loads(cfg.to_json_string()), f, indent=2)

    for fn in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "chat_template.jinja"):
        sp = os.path.join(SRC, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(DST, fn))

    print()
    print("DONE ->", DST)
    print("  total tensors:", len(new_wmap))
    print("  vision tensors:", len(vis))
    vc = cfg.vision_config
    print("  vision: depth=%d hidden=%d heads=%d out=%d patch=%d merge=%d "
          "temporal=%d img=%d" % (
        vc.depth, vc.hidden_size, vc.num_heads, getattr(vc, "out_hidden_size", -1),
        vc.patch_size, vc.spatial_merge_size, getattr(vc, "temporal_patch_size", -1),
        getattr(vc, "image_size", -1)))
    print("  image_token_id=%s vision_start=%s vision_end=%s video=%s" % (
        cfg.image_token_id, cfg.vision_start_token_id,
        cfg.vision_end_token_id, cfg.video_token_id))


if __name__ == "__main__":
    main()
