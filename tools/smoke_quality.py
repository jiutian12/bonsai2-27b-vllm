#!/usr/bin/env python3
"""Smoke-test + throughput test against the vLLM Bonsai server.

端口/模型名改成可覆盖（历史上写死 :18030 + 旧别名，切到 18020 后一直超时）:
  VLLM_PORT=18020 VLLM_MODEL=qwen3.8-27b python _bs_test.py
"""
import json, os, time, urllib.request

PORT = os.environ.get("VLLM_PORT", "18020")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def chat(prompt, max_tokens=256, temperature=0.0, thinking=False):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t
    ch = r["choices"][0]
    msg = ch["message"]
    txt = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = r["usage"]
    ct = u.get("completion_tokens", 0)
    return txt, reasoning, ct, dt


PROMPTS = [
    ("gsm8k", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"),
    ("fact", "What is the capital of France? Answer in one short sentence."),
    ("zh", "用一句话解释什么是量化推理。"),
]

print("=== SMOKE TEST (vLLM Bonsai PTQ1_0 @ :18030) ===", flush=True)
for tag, p in PROMPTS:
    try:
        txt, reason, ct, dt = chat(p, max_tokens=200)
        tps = ct / dt if dt > 0 else 0
        print(f"\n--- [{tag}] {p[:60]}...", flush=True)
        if reason:
            print(f"  reasoning: {reason[:300]}", flush=True)
        print(f"  answer   : {txt[:400]}", flush=True)
        print(f"  tokens={ct} time={dt:.2f}s -> {tps:.1f} tok/s", flush=True)
    except Exception as e:
        print(f"\n--- [{tag}] ERROR {type(e).__name__}: {e}", flush=True)
print("\nDONE")
