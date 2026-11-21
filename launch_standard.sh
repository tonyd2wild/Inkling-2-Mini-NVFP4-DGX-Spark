#!/usr/bin/env bash
# Inkling 2 Mini (official NVFP4 checkpoint) on one DGX Spark (GB10): STANDARD profile.
# 262,144-token context, gpu-memory-utilization 0.80, extensible KV cache (vLLM sizes KV after warmup).
# Container inkling2mini, host port 18120 -> 8000 (override with NAME / PORT).
set -euo pipefail

NAME="${NAME:-inkling2mini}"
PORT="${PORT:-18120}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Inkling-2-Mini}"
# Tested image: vllm/vllm-openai:inkling-2-mini = vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
IMAGE="${IMAGE:-vllm/vllm-openai:inkling-2-mini}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart no -p "$PORT:8000" \
  -v "$MODEL_DIR:/models/Inkling-2-Mini:ro" "$IMAGE" \
  /models/Inkling-2-Mini \
  --served-model-name inkling-2-mini thinkingmachines/Inkling-2-Mini \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.80 \
  --reasoning-parser inkling \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'

echo "started $NAME (standard: 262144 ctx, util 0.80) on :$PORT; ready in about 5 to 6 min"
echo "watch: docker logs -f $NAME | grep -E 'KV cache|concurrency|startup complete'"
