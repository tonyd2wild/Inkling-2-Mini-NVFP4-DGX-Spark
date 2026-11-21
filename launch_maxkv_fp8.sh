#!/usr/bin/env bash
# Inkling 2 Mini (official NVFP4 checkpoint) on one DGX Spark (GB10): MAX KV profile, FP8 KV.
# Full 1,048,576-token context, KV pool set in bytes, --kv-cache-dtype fp8 (E4M3, unit scales).
# Container inkling2mini, host port 18120 -> 8000 (same as the other profiles, override with NAME / PORT).
#
# KV_GIB defaults to 74, not 80. FP8 KV splits the cache into 99 KV groups (13 with BF16), and the
# per-group buffers cost about 5 GiB more memory outside the KV pool. Measured on GB10:
#   KV_GIB=80 -> 3,577,201 tokens (3.41x at 1M), but MemAvailable fell to 1.9 GiB during warmup (killed)
#   KV_GIB=74 -> 3,308,911 tokens (3.16x at 1M), MemAvailable low 7.3 GiB at startup, ~8.2 GiB idle
# The Inkling attention code allows fp8 KV only on the sm_12x family (GB10 is sm_121).
set -euo pipefail

NAME="${NAME:-inkling2mini}"
PORT="${PORT:-18120}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Inkling-2-Mini}"
# Tested image: vllm/vllm-openai:inkling-2-mini = vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
IMAGE="${IMAGE:-vllm/vllm-openai:inkling-2-mini}"
KV_GIB="${KV_GIB:-74}"
MAX_LEN="${MAX_LEN:-1048576}"
KV_BYTES=$(( KV_GIB * 1024 * 1024 * 1024 ))

# GB10 page-cache trap: drop clean page cache once before start. Skipped without passwordless sudo.
sync
if sudo -n true 2>/dev/null; then
  echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart no -p "$PORT:8000" \
  -v "$MODEL_DIR:/models/Inkling-2-Mini:ro" "$IMAGE" \
  /models/Inkling-2-Mini \
  --served-model-name inkling-2-mini thinkingmachines/Inkling-2-Mini \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --kv-cache-dtype fp8 \
  --reasoning-parser inkling \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'

echo "started $NAME (max KV FP8: ${KV_GIB} GiB = ${KV_BYTES} bytes, max-model-len $MAX_LEN) on :$PORT"
echo "watch: free -g ; docker logs -f $NAME | grep -E 'KV cache|concurrency|startup complete'"
