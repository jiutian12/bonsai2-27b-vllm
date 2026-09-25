#!/usr/bin/env python3
"""Speed validation for the vLLM+Bonsai backend on :18020 (streaming TTFT vs decode)."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18020"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"


def filler(n):
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()
    return "\n".join(f"Entry {i}: value {i*37%991} tag {words[i%12]}{i}." for i in range(n))


def stream(prompt, max_tokens=80):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                break
            try:
                j = json.loads(p)
            except Exception:
                continue
            d = j["choices"][0].get("delta", {})
            piece = d.get("content") or d.get("reasoning_content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    total = time.time() - t0
    dec = (total - ttft) if ttft else total
    return n, ttft, total, (n / dec if dec > 0 else 0)


print(f"=== SPEED  base={BASE} model={MODEL} ===", flush=True)
for entries in (0, 250, 1000, 3000):
    prompt = filler(entries) + "\n\nIgnore the list. Write a detailed 3-sentence description of a sunset over the sea."
    try:
        n, ttft, total, tps = stream(prompt)
        print(f"  ctx~{entries*11:6d} tok  gen={n:3d}  TTFT={ttft:6.2f}s  decode={tps:6.1f} t/s", flush=True)
    except Exception as e:
        print(f"  entries={entries} ERROR {type(e).__name__}: {e}", flush=True)
print("DONE", flush=True)
