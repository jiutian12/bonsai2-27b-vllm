#!/usr/bin/env bash
# Serve a locally converted Bonsai 2 27B (PTQ1_0) vLLM checkpoint with the
# prism_ternary plugin.
#
#   VENV=~/venv CKPT=~/bonsai_vllm ./serve_vllm_bonsai.sh
#
# Knobs (env):
#   PORT          default 8000
#   MAX_LEN       default 262144  (= 256K, the model's max_position_embeddings)
#   MAX_SEQS      default 128
#   BATCH_TOKENS  default 2048    (see docs/04-long-prefill-concurrency.md)
#   KV_DTYPE      default fp8     (fp8 | turboquant_4bit_nc | ...)
#   GPU_UTIL      default 0.90
#   ALIAS         default bonsai-ptq1
#
# NOTE: keep the venv's bin on PATH — torch needs `ninja` from there to JIT the
# plugin's CUDA kernels, and CUDA_HOME must point at a usable toolkit.
set -x

VENV="${VENV:-$HOME/venv}"
CKPT="${CKPT:-$HOME/bonsai_vllm}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-128}"
BATCH_TOKENS="${BATCH_TOKENS:-2048}"
KV_DTYPE="${KV_DTYPE:-fp8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
ALIAS="${ALIAS:-bonsai-ptq1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$VENV/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# venv may ship mismatched flashinfer / flashinfer-cubin: skip the sampler path
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

exec "$VENV/bin/vllm" serve "$CKPT" \
  --served-model-name "$ALIAS" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$BATCH_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype bfloat16 \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --trust-remote-code
