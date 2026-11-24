#!/usr/bin/env bash
# Inkling 2 Mini (official NVFP4 checkpoint) on one DGX Spark (GB10) with the DFlash2 drafter
# (incoai/Inkling-2-Mini-DFlash2, k=7) instead of DSpark. Otherwise the STANDARD profile:
# 262,144-token context, gpu-memory-utilization 0.80, BF16 KV.
#
# Before the first run: ./prepare_dflash2.sh (builds dflash2/*.py from the image + patches)
# and download the drafter:  hf download incoai/Inkling-2-Mini-DFlash2 --local-dir ~/models/Inkling-2-Mini-DFlash2
#
# usage: ./launch_dflash2.sh [extra vLLM flags]
# env:   NAME=inkling2mini PORT=18120 K=7 MAXLEN=262144 UTIL=0.80 RESTART=no|unless-stopped
#        MODEL_DIR=~/models/Inkling-2-Mini DRAFT_DIR=~/models/Inkling-2-Mini-DFlash2
set -euo pipefail

NAME="${NAME:-inkling2mini}"
PORT="${PORT:-18120}"
K="${K:-7}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Inkling-2-Mini}"
DRAFT_DIR="${DRAFT_DIR:-$HOME/models/Inkling-2-Mini-DFlash2}"
# Tested image: vllm/vllm-openai:inkling-2-mini = vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
IMAGE="${IMAGE:-vllm/vllm-openai:inkling-2-mini}"
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HERE/dflash2
VL=/usr/local/lib/python3.14/dist-packages/vllm
[ -f "$DRAFT_DIR/model.safetensors" ] || { echo "drafter not found in $DRAFT_DIR (set DRAFT_DIR=...)"; exit 1; }
[ -f "$D/qwen3_dflash2.py" ] || { echo "run ./prepare_dflash2.sh first"; exit 1; }

# GB10 page-cache trap: drop clean page cache once before start (skipped without passwordless sudo).
sync
if sudo -n true 2>/dev/null; then echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart "${RESTART:-no}" -p "$PORT:8000" \
  -v "$MODEL_DIR:/models/Inkling-2-Mini:ro" \
  -v "$DRAFT_DIR:/models/dflash2:ro" \
  -v "$D/qwen3_dflash.py:$VL/model_executor/models/qwen3_dflash.py:ro" \
  -v "$D/qwen3_dflash2.py:$VL/model_executor/models/qwen3_dflash2.py:ro" \
  -v "$D/dflash2_speculator.py:$VL/v1/worker/gpu/spec_decode/dflash2/speculator.py:ro" \
  -v "$D/eagle3_utils.py:$VL/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:ro" \
  "$IMAGE" /models/Inkling-2-Mini \
  --served-model-name inkling-2-mini-nvfp4 inkling-2-mini \
  --tensor-parallel-size 1 \
  --max-model-len "${MAXLEN:-262144}" \
  --gpu-memory-utilization "${UTIL:-0.80}" \
  --reasoning-parser inkling \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":$K}" "$@"

echo "started $NAME (DFlash2 k=$K, ${MAXLEN:-262144} ctx, util ${UTIL:-0.80}) on :$PORT; ready in about 5 to 6 min"
echo "watch: docker logs -f $NAME | grep -E 'KV cache|concurrency|startup complete'"
