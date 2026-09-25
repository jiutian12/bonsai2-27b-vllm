#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双并发 + /metrics 实时探针：在请求在飞期间采样 vLLM 的 running/waiting/KV 占用，
用来判定「两个请求到底有没有同时在飞」，而不是靠 TTFT 猜。

用法: python dual_concurrency_probe.py <target_tokens> [n_conc] [seed]
"""
import json
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:18020"
MODEL = "qwen3.8-27b"

FILLER = (
    "考虑一个分布式推理集群的容量规划问题。我们有三类请求：短问答、长文档摘要、"
    "以及多轮智能体工具调用。每类请求的上下文长度分布不同，KV 缓存占用的方差也很大。"
    "运维的目标是在不触发显存溢出的前提下，让平均排队时延尽可能低，同时保证长请求"
    "不被反复抢占。请逐步分析这个权衡，并给出一个可执行的调度策略。"
)
WORDS = ("量子 梯度 缓存 调度 带宽 张量 流形 频谱 拓扑 熵增 粒子 相位 谐振 卷积 稀疏 "
         "岩层 洋流 季风 苔原 珊瑚 陨石 极光 潮汐 冰川 熔岩 沙丘 峡湾 洞穴 泽地 草原 "
         "丝绸 瓷器 篆刻 壁画 编钟 榫卯 漆器 织锦 玉琮 拓片 青瓷 竹简 石经 藻井 斗拱").split()

GAUGES = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
          "num_requests_preempted_total")


def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def token_count(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}) as r:
        d = json.loads(r.read())
    return d.get("count") or len(d.get("tokens", []))


def build_unique_prompt(target, rng, head_tokens=4000):
    head = " ".join(rng.choices(WORDS, k=1200))
    t = token_count(head)
    head = " ".join(rng.choices(WORDS, k=max(1, round(head_tokens * 1200 / max(1, t)))))
    t_head = token_count(head)
    t_fill = token_count(FILLER)
    m = max(1, (target - t_head) // t_fill)
    cand = head + "\n" + FILLER * m
    tot = token_count(cand)
    while tot > target and m > 1:
        m -= 1
        cand = head + "\n" + FILLER * m
        tot = token_count(cand)
    return cand, tot


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for g in GAUGES:
            if line.startswith(f"vllm:{g}{{") or line.startswith(f"vllm:{g} "):
                try:
                    out[g] = float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
        # KV 池 token 数：从 cache_config_info 读，别写死（换 KV dtype 后会变）
        if line.startswith("vllm:cache_config_info"):
            m = re.search(r'kv_cache_size_tokens="(\d+)"', line)
            if m:
                out["kv_cache_size_tokens"] = float(m.group(1))
    return out


def poller(stop, peak, samples):
    while not stop.is_set():
        try:
            with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
                m = parse_metrics(r.read().decode("utf-8", "ignore"))
            for k, v in m.items():
                peak[k] = max(peak.get(k, 0.0), v)
            samples.append(m)
        except Exception:
            pass
        time.sleep(1.5)


def one(prompt, idx, barrier, out):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": 16, "temperature": 0.0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.time()
    ttft = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    ev = json.loads(p)
                except Exception:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    if ch.get("text") and ttft is None:
                        ttft = time.time() - t0
        out[idx] = {"ok": True, "ttft": ttft, "wall": time.time() - t0,
                    "pt": (usage or {}).get("prompt_tokens")}
    except Exception as e:
        out[idx] = {"ok": False, "err": f"{type(e).__name__}: {e}", "wall": time.time() - t0}


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 129976
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 101
    prompts, tots = [], []
    for i in range(n):
        p, tot = build_unique_prompt(target, random.Random(seed * 1000 + i))
        prompts.append(p)
        tots.append(tot)
        print(f"  #{i}: prompt_tokens={tot}", flush=True)

    out, stop, peak, samples = {}, threading.Event(), {}, []
    th = threading.Thread(target=poller, args=(stop, peak, samples), daemon=True)
    th.start()
    time.sleep(1)

    barrier = threading.Barrier(n)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        for i in range(n):
            ex.submit(one, prompts[i], i, barrier, out)
    wall = time.time() - t0
    stop.set()
    time.sleep(0.2)

    print(f"\n=== 结果 (总墙钟 {wall:.1f}s) ===", flush=True)
    for i in sorted(out):
        r = out[i]
        if r["ok"]:
            tt = f"{r['ttft']:.2f}s" if r["ttft"] else "n/a"
            print(f"  #{i}: OK prompt={r['pt']} TTFT={tt} wall={r['wall']:.2f}s", flush=True)
        else:
            print(f"  #{i}: FAIL {r['err']}", flush=True)
    print("\n=== 在飞期间 /metrics 峰值 ===", flush=True)
    for k in ("num_requests_running", "num_requests_waiting",
              "num_requests_preempted_total", "kv_cache_usage_perc"):
        print(f"  peak {k:28s} = {peak.get(k)}", flush=True)
    kv = peak.get("kv_cache_usage_perc", 0) or 0
    pool = peak.get("kv_cache_size_tokens")
    if pool:
        print(f"  -> 峰值 KV tokens ~= {kv * pool:,.0f}  (池 {pool:,.0f})", flush=True)
    run = peak.get("num_requests_running") or 0
    print(f"  -> {n} 路是否同时在飞? {'是' if run >= n else '否'}"
          f"（peak running={run:.0f}）", flush=True)


if __name__ == "__main__":
    main()
