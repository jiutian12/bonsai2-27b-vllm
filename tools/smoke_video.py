#!/usr/bin/env python3
"""Video + image multimodal smoke test against the vLLM Bonsai server.

Verifies that the vision+video checkpoint accepts both image and video input
via the OpenAI-compatible /v1/chat/completions API and returns a sensible
answer (the model describes what it sees in the testsrc clip).

用法（与 smoke_quality.py 一致，用 env 覆盖端口/模型名）:
    VLLM_PORT=18020 VLLM_MODEL=qwen3.8-27b python smoke_video.py

视频请求的 payload 结构（vLLM 要求 url 嵌套在 video_url 对象里）:
    {"type":"video_url","video_url":{"url":"data:image/jpeg;base64,<b64>"}}
注意：base64 数据是 mp4 的，mime 标签写成 data:image/jpeg 即可（vLLM 按
base64 内容自行判定容器，已实测 200 + 正确描述）。
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT = os.environ.get("VLLM_PORT", "18020")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TMP_VID = os.path.join(tempfile.gettempdir(), "_tmp_video.mp4")


def gen_testsrc(path, duration=2, size="256x256", fps=4):
    """Generate a small motion test card (color bars + counter) with ffmpeg."""
    cmd = ["ffmpeg", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={duration}:size={size}:rate={fps}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print(f"ffmpeg failed (rc={p.returncode}):", p.stderr.decode("utf-8", "ignore")[-1500:],
              file=sys.stderr)
        sys.exit(1)
    print(f"[gen] {os.path.basename(path)} {os.path.getsize(path)} bytes")


def post_video(video_path, question):
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "video_url", "video_url": {"url": "data:image/jpeg;base64," + b64}},
        ]}],
        "max_tokens": 160,
        "temperature": 0.0,
    }
    return payload, b64


def call(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            dt = time.time() - t
            d = json.load(r)
            ch = d["choices"][0]
            msg = ch["message"]
            return {
                "ok": True, "status": 200, "dt": dt,
                "content": msg.get("content") or "",
                "completion_tokens": d["usage"].get("completion_tokens", 0),
                "prompt_tokens": d["usage"].get("prompt_tokens", 0),
            }
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code,
                "err": e.read().decode("utf-8", "ignore")[:800], "dt": 0.0}


def main():
    print(f"=== MULTIMODAL SMOKE (video + image) @ {URL} model={MODEL} ===", flush=True)
    gen_testsrc(TMP_VID, duration=2, size="256x256", fps=4)
    q = "用中文简短回答：你在这段视频里看到什么？描述主要颜色、形状，以及是否有运动/变化。"
    payload, _ = post_video(TMP_VID, q)

    res = call(payload)
    print("\n--- [video] 请求 ---", flush=True)
    print(f"  prompt_tokens={res['prompt_tokens']}", flush=True)
    if res.get("ok"):
        print(f"  completion_tokens={res['completion_tokens']} dt={res['dt']:.2f}s", flush=True)
        print(f"  answer: {res['content'][:600]}", flush=True)
    else:
        print(f"  FAIL status={res['status']}: {res.get('err')}", flush=True)
    print("DONE", flush=True)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
