#!/usr/bin/env bash
# Inkling 2 Mini (official NVFP4 checkpoint) on one DGX Spark (GB10): MAX KV profile, BF16 KV.
# Full 1,048,576-token context, KV pool set in bytes (kv_cache_dtype auto = BF16).
# Container inkling2mini, host port 18120 -> 8000 (same as the other profiles, override with NAME / PORT).
#
# KV_GIB (default 80) is the KV pool in GiB. On GB10 the GPU and the OS share 121.6 GiB, so the
# pool must leave room for the CPU side. 80 GiB left about 7.2 GiB MemAvailable after warmup.
# Step down by 4 GiB (KV_GIB=76, then 72) if it OOMs or MemAvailable drops below ~6 GiB.
set -euo pipefail

NAME="${NAME:-inkling2mini}"
PORT="${PORT:-18120}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Inkling-2-Mini}"
# Tested image: vllm/vllm-openai:inkling-2-mini = vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
IMAGE="${IMAGE:-vllm/vllm-openai:inkling-2-mini}"
KV_GIB="${KV_GIB:-80}"
MAX_LEN="${MAX_LEN:-1048576}"
KV_BYTES=$(( KV_GIB * 1024 * 1024 * 1024 ))

# GB10 page-cache trap: clean page cache counts as used unified memory until the kernel
# reclaims it. Drop it once before start (clean pages only, safe). Skipped without passwordless sudo.
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
  --reasoning-parser inkling \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'

echo "started $NAME (max KV: ${KV_GIB} GiB = ${KV_BYTES} bytes BF16 KV, max-model-len $MAX_LEN) on :$PORT"
echo "watch: free -g ; docker logs -f $NAME | grep -E 'KV cache|concurrency|startup complete'"
