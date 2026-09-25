#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发扫描（扩展版）：1 → 128 并发，报聚合吞吐 + 每请求延迟分位 + KV 峰值占用。

与 concurrency_vllmbonsai.py 的区别：
  - 档位扩到 max_num_seqs(128)，用来找饱和点
  - 记录每请求延迟的 p50/p90/max（长尾比均值重要）
  - 顺带采样 /metrics 的 KV 占用峰值与 running/waiting
用法: python concurrency_sweep2.py [base] [model] [levels...]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18020"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
LEVELS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]

_peak = {"running": 0.0, "waiting": 0.0, "kv": 0.0}


def poll_metrics(stop):
    while not stop.is_set():
        try:
            with urllib.request.urlopen(BASE + "/metrics", timeout=4) as r:
                txt = r.read().decode("utf-8", "ignore")
            for line in txt.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    _peak["running"] = max(_peak["running"], float(line.rsplit(" ", 1)[1]))
                elif line.startswith("vllm:num_requests_waiting{"):
                    _peak["waiting"] = max(_peak["waiting"], float(line.rsplit(" ", 1)[1]))
                elif line.startswith("vllm:kv_cache_usage_perc"):
                    _peak["kv"] = max(_peak["kv"], float(line.rsplit(" ", 1)[1]))
        except Exception:
            pass
        time.sleep(2)


def one(i, max_tokens=256):
    body = {"model": MODEL,
            "messages": [{"role": "user",
                          "content": f"Write a short paragraph about topic number {i}."}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True,   # 必须：否则各请求提前停、生成长度不一，延迟不可比
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    return d["usage"].get("completion_tokens", 0), time.time() - t


def sweep(n, max_tokens=256):
    res = [None] * n

    def w(i):
        try:
            res[i] = one(i, max_tokens)
        except Exception:
            res[i] = (0, 0.0)

    t0 = time.time()
    ths = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    lat = sorted(r[1] for r in res)
    tot = sum(r[0] for r in res)
    return tot, wall, lat


print(f"=== CONCURRENCY SWEEP  base={BASE}  model={MODEL} ===", flush=True)
_stop = threading.Event()
threading.Thread(target=poll_metrics, args=(_stop,), daemon=True).start()
print(f"{'并发':>4}  {'聚合 tok/s':>10}  {'单路均 t/s':>10}  {'p50 延迟':>9}  "
      f"{'p90 延迟':>9}  {'max':>8}  {'KV峰值':>7}", flush=True)
for n in LEVELS:
    try:
        tot, wall, lat = sweep(n)
        agg = tot / wall if wall > 0 else 0
        p50 = statistics.median(lat)
        p90 = lat[min(len(lat) - 1, int(len(lat) * 0.9))]
        print(f"{n:>4}  {agg:>10.1f}  {agg / n:>10.1f}  {p50:>8.2f}s  "
              f"{p90:>8.2f}s  {lat[-1]:>7.2f}s  {_peak['kv']:>7.3f}", flush=True)
    except Exception as e:
        print(f"{n:>4}  ERROR {type(e).__name__}: {e}", flush=True)
print(f"peak running={_peak['running']:.0f}  waiting={_peak['waiting']:.0f}", flush=True)
print("DONE", flush=True)
