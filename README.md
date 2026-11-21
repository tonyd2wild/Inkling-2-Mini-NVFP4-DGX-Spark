# Inkling 2 Mini (NVFP4) on one NVIDIA DGX Spark: vLLM, DSpark, up to 1M context

A tested recipe for serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single DGX Spark (GB10) with the official vLLM image, DSpark speculative decoding and three launch tiers: the vLLM recipe's 262K setup, and two 1,048,576-token (full 1M) setups, one with BF16 KV and one with FP8 KV.

Unofficial community recipe. Not affiliated with Thinking Machines Lab, vLLM or NVIDIA. This repo has no weights.

Measured on one Spark on 2026-11-20 and 2026-11-21:

| | standard | max BF16 | **max FP8** |
|---|---|---|---|
| Script | `launch_standard.sh` | `launch_maxkv.sh` | `launch_maxkv_fp8.sh` |
| Context per request | 262,144 | 1,048,576 | 1,048,576 |
| KV dtype and size | BF16, `--gpu-memory-utilization 0.80` (68.89 GiB) | BF16, 80 GiB | FP8 (E4M3), 74 GiB |
| KV pool | 1,346,671 tokens | 2,075,828 tokens | **3,308,911 tokens** |
| Full-length requests at once | 5.14x at 262K | 1.98x at 1M | **3.16x at 1M** |
| MemAvailable after warmup | 20.2 GiB | 7.2 to 7.4 GiB | 8.2 to 8.4 GiB |
| Decode, one user | 40 to 45 tok/s (short chats) | 38 to 73 tok/s short, 28.3 tok/s at 200K | 38 to 73 tok/s short, 47.2 tok/s at 200K, 18.4 tok/s at 950K |
| Prefill | not measured at length | ~1,680 tok/s over 200K | ~1,690 tok/s over 200K, 598 tok/s over 950K |
| 200K needle | not run | PASS (TTFT 119 s) | PASS (TTFT 118 s) |
| ~1M needle (949,953 tokens) | n/a | not run | **PASS** (TTFT 26.5 min) |
| 69-scenario eval: quality / responsiveness / deployability | **88.4** / 92.1 / 89.5 (56/10/3) | same weights and KV dtype as standard | **87.0** / 92.7 / 88.7 (55/10/4) |
| Eval decode | 47.7 tok/s | | 47.3 tok/s |

Details: [memory](#6-memory-budget-unified-memory-1216-gib-total), [KV math](#7-kv-pool-math), [speed and checks](#9-speed-and-sanity-checks), [eval](#10-69-scenario-tool-eval).

**Which tier to use**

- **standard**: prompts up to 262K, and 20 GiB of headroom for the OS and anything else on the box. The right default for chat, agents and tool use.
- **max BF16**: prompts past 262K with the KV cache unquantized. Holds one full 1M request plus shorter ones. Leaves only about 7 GiB for the OS.
- **max FP8**: the most context. A 3.3M-token pool holds three full 1M requests at once. It passed the ~1M needle and scored within 1.4 points of BF16 on our eval. The KV is unit-scale FP8 with no calibrated scales, so check it on your own long-context work before you rely on it.
- A near-1M prompt takes about 26 minutes to its first token on either max tier, so the 1M tiers suit long-document and long-session work, not latency-sensitive traffic.

> **Unified-memory warning.** On a DGX Spark the GPU and the operating system share the same 121.6 GiB. Every GiB you give to the KV cache is taken from the OS. The max tiers leave about 7 to 8 GiB of MemAvailable after warmup. Do not run other big jobs (downloads, builds, a second model) next to them, and do not raise `KV_GIB` past 80 (BF16) or 74 (FP8). An out-of-memory on GB10 can hang the whole box until a power cycle. `watch_start.sh` kills the container if MemAvailable drops under 3 GiB.

## 1. The model

- **Inkling 2 Mini**, Thinking Machines Lab, released 2026-11-20 under Apache 2.0. Text and image in, text out.
- **Size:** 26.4B total parameters, 3.4B active per pass. It is a looped MoE: 24 layers, and the middle 12 (layers 6 to 17) run 3 times with shared weights, so there are 48 layer applications per token (~5.4B effective active).
  - Each layer routes to 8 of 128 experts.
  - Attention is 20 sliding-window layers (window 1024) plus 4 global layers (5, 11, 17, 23), with 24 query heads, 8 KV heads and head_dim 128.
  - Every looped layer keeps its own KV cache and conv-state cache per pass, so there are 48 KV caches and 48 conv-state caches.
- **Checkpoint:** about 20 GB.
  - Experts in layers 0 to 19 are NVFP4. Experts in layers 20 to 23 are MXFP8 (`quant_method: inkling2`, modelopt).
  - Attention, embeddings and the unembed are BF16.
  - Files: `model-00001-of-00001.safetensors` (19.5 GB) and `mtp.safetensors` (0.64 GB). The second file is the **DSpark** speculator, a 3-layer BF16 drafter with block size 8.
- **Context:** up to 1,048,576 tokens. The model card says needle-in-a-haystack was tested to 1M. The official vLLM recipe caps DGX Spark at 262,144 "to fit in memory". The max tiers here run the full 1M on one Spark.

## 2. Hardware and software (as tested)

| Item | Value |
|---|---|
| Box | 1x NVIDIA DGX Spark, GB10 (sm_121), 121.6 GiB unified memory visible to Linux (MemTotal 127,533,040 kB) |
| OS | DGX OS 7.2.3 (OTA 7.6.0), kernel 7.0.0-1025-nvidia |
| Driver / CUDA | 580.178.04 / CUDA 13.0 |
| Image | `vllm/vllm-openai:inkling-2-mini`, digest `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (built 2026-11-20, 27.1 GB) |
| vLLM inside | `0.1.dev25855+g7cb5dae22` (pre-release of 0.35), torch 2.14.0+cu130, Python 3.14 |
| Attention backends | FLASH_ATTN (FA2) for the 48 attention caches, INKLING_SCONV_SWA for the 24 conv layers, FLASH_ATTN x3 for the DSpark drafter |
| Weights | Hugging Face revision `1fb4134b9a74f9681054f4ef82e4bf1f458a2337` ("upload model"). The next commit, `ed40e7da`, only declares the dtype at the top level of `config.json`. |
| Weights path | `~/models/Inkling-2-Mini` on the host, mounted read-only at `/models/Inkling-2-Mini` |

## 3. Download

```bash
pip install -U "huggingface_hub[cli]"        # gives the `hf` command
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/models/Inkling-2-Mini
docker pull vllm/vllm-openai:inkling-2-mini
# or the exact image we tested:
docker pull vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
```

Check that `mtp.safetensors` is present. Without it there is no DSpark, and decode drops a lot.

While a big download runs on a Spark, the page cache fills unified memory. If a GPU job is running at the same time, drop the clean cache with `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches` (the "GB10 page-cache trap").

## 4. Quick start

```bash
git clone https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark
cd Inkling-2-Mini-NVFP4-DGX-Spark

./launch_standard.sh            # 262K context, util 0.80, BF16 KV
./launch_maxkv_fp8.sh           # 1M context, 74 GiB FP8 KV (the most context)
./launch_maxkv.sh               # 1M context, 80 GiB BF16 KV
KV_GIB=76 ./launch_maxkv.sh     # step down 4 GiB if you want more OS headroom
./watch_start.sh                # optional: memory watch while it starts (kills at MemAvailable < 3 GiB)

python3 sanity.py chat                      # 2 short chats
python3 sanity.py needle --tokens 200000    # one long-context needle
./stop.sh
```

Every launcher uses the same container name (`inkling2mini`) and host port (`18120`), so clients do not change when you switch tiers. Override with `NAME=`, `PORT=`, `MODEL_DIR=` or `IMAGE=` in the environment. Starting a tier replaces whatever tier is running.

Startup takes about 5 to 6 minutes:
- weights load: ~105 to 121 s
- profile and JIT warmup: ~65 s
- CUDA graph capture: ~100 s (51 piecewise + 48 full + 48 DSpark graphs)

Ready when the log shows `Application startup complete`.

The max-KV BF16 docker command (what `launch_maxkv.sh` runs, with `KV_GIB=80`):

```bash
docker run -d --name inkling2mini --gpus all --ipc=host --restart no -p 18120:8000 \
  -v ~/models/Inkling-2-Mini:/models/Inkling-2-Mini:ro vllm/vllm-openai:inkling-2-mini \
  /models/Inkling-2-Mini \
  --served-model-name inkling-2-mini thinkingmachines/Inkling-2-Mini \
  --tensor-parallel-size 1 \
  --max-model-len 1048576 \
  --kv-cache-memory-bytes 85899345920 \
  --reasoning-parser inkling \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

- The standard tier uses `--max-model-len 262144 --gpu-memory-utilization 0.80` in place of the two max-KV lines.
- The FP8 tier (`launch_maxkv_fp8.sh`) is the max-KV command with `--kv-cache-memory-bytes 79456894976` (74 GiB) and `--kv-cache-dtype fp8` added.

## 5. Every flag explained

| Flag | Why |
|---|---|
| `--gpus all` | Exposes the GB10 to the container. |
| `--ipc=host` | Shared memory for PyTorch workers (the container default /dev/shm is 64 MB). |
| `--restart no` | Start and stop it on purpose. Nothing restarts behind your back after an OOM. |
| `-p 18120:8000` | Host port. vLLM listens on 8000 inside the container. |
| `-v ...:/models/Inkling-2-Mini:ro` | Local weights, read-only. No HF token or network needed at start. |
| `/models/Inkling-2-Mini` | Model path (positional, the image entrypoint is `vllm serve`). |
| `--served-model-name inkling-2-mini thinkingmachines/Inkling-2-Mini` | Either name works in the `model` field. |
| `--tensor-parallel-size 1` | One GPU. The official recipe says TP1 is the most efficient setup for this model. |
| `--max-model-len 262144` / `1048576` | Per-request context cap. 1M is the model's native maximum. The DSpark drafter follows it automatically. |
| `--gpu-memory-utilization 0.80` | (standard) Caps vLLM at 97.3 GiB of the 121.6 GiB. The KV cache is then sized from memory measured after warmup (the "extensible KV cache"). |
| `--kv-cache-memory-bytes N` | (max tiers) Sets an exact KV pool in bytes. It ignores gpu-memory-utilization and turns the extensible KV cache off. This build's log line suggests `--kv-cache-memory=...`, but the real flag is `--kv-cache-memory-bytes`. |
| `--kv-cache-dtype fp8` | (FP8 tier) FP8 E4M3 KV with unit scales (no calibrated scales). The Inkling code allows it only on sm_12x, and GB10 is sm_121. The default `auto` is BF16. |
| `--reasoning-parser inkling` | Splits thinking into `reasoning_content`, so `content` holds only the answer. |
| `--enable-auto-tool-choice --tool-call-parser inkling` | OpenAI-style `tools` / `tool_calls`. |
| `--speculative-config {"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}` | DSpark speculative decoding from `mtp.safetensors`. `num_speculative_tokens` must be 7 (checkpoint block size 8 minus 1, enforced). With spec decode on, vLLM caps scheduled tokens per step at 2048. |

Defaults left on: prefix caching, chunked prefill, CUDA graphs (FULL_AND_PIECEWISE). Sampling defaults come from `generation_config.json` (temperature 1.0). Pass `temperature` per request if you want something else.

## 6. Memory budget (unified memory, 121.6 GiB total)

On GB10, "GPU memory" and the OS share the same 121.6 GiB, so every GiB given to KV is taken away from the CPU side. vLLM's own suggestion for the standard run, `--kv-cache-memory=95187425792` (88.65 GiB) "to fully utilize gpu memory", would leave the OS about 0 to 2 GiB. Do not use it.

| Item | Standard (262K, BF16) | Max BF16 (80 GiB) | Max FP8 (74 GiB) |
|---|---|---|---|
| Weights (main + DSpark), as logged by "Model loading took" | 19.74 GiB | 20.11 GiB | 20.11 GiB |
| KV groups | 13 | 13 | 99 |
| CUDA graphs | 2.91 GiB | 2.83 GiB | 2.99 GiB |
| KV cache | 68.89 GiB (31,352 blocks) | 80.00 GiB (36,408 blocks) | 74.00 GiB |
| Lowest MemAvailable during startup | not logged | 6.3 GiB | 7.3 GiB |
| **MemAvailable after warmup** | **20.2 GiB** | **7.2 to 7.4 GiB** | **8.2 to 8.4 GiB** |
| Lowest MemAvailable during the checks in section 9 | not logged | 7.0 GiB | 7.5 GiB |
| Swap in use | not logged | 1.4 GiB (left from startup, stable) | 3.0 GiB (left from startup, stable) |

FP8 at the same 80 GiB as BF16 did not fit: it reported 3,577,201 tokens (3.41x at 1M), then MemAvailable fell to 1.9 GiB during the JIT warmup and `watch_start.sh` killed it. FP8 KV costs about 5 GiB more memory outside the KV pool (section 7), so the FP8 tier uses 74 GiB.

Why 80 GiB for the max BF16 tier: the standard tier left 20.2 GiB MemAvailable with 68.89 GiB of KV. Adding 11.11 GiB of KV (to 80 GiB) plus the 1M-length overhead was planned to leave about 8.3 GiB. The measured result was about 1 GiB lower, 7.2 to 7.4 GiB at idle. That is above a 6 GiB floor, so 80 GiB stays. If you want 8 to 10 GiB of OS headroom, use `KV_GIB=78` (about 2.02M BF16 tokens) or `KV_GIB=76` (about 1.97M). 84 GiB would land near 3 GiB, which is too tight.

## 7. KV pool math

These numbers come from the startup logs and the vLLM source in this image.

- **Hybrid KV, one shared block pool.** The 13 KV groups are:
  - 1 full-attention group: 8 global caches plus the DSpark global layer, so 9 layers
  - 5 sliding-window groups (40 caches, padded to 45)
  - 1 DSpark sliding-window group
  - 6 conv-state groups (48 caches, padded to 54)

  Group block sizes are 64 tokens (attention) and 16 (conv).
- **BF16: one block id = 2.25 MiB** (9 layers x 64 tokens x 4 KiB/token of BF16 K+V). So `blocks = floor(KV_bytes / 2,359,296)`.
  - Check: 68.89 GiB gives 31,352 blocks, which matches the log.
- **Blocks per request at max length:**
  - Full attention needs `ceil(max_model_len / 64)` blocks.
  - Every sliding-window and conv group is capped by window + in-flight tokens. That cap does not depend on context length and adds up to a fixed **2,007 blocks**.
  - At 262,144 that is 4,096 + 2,007 = 6,103 blocks per request. 31,352 / 6,103 = 5.137x, and 5.137 x 262,144 = 1,346,671 tokens, exactly what vLLM logged.
  - At 1,048,576 it is 16,384 + 2,007 = 18,391 blocks per request.
- **"GPU KV cache size: N tokens"** is `max_concurrency x max_model_len`. Because the fixed 2,007-block overhead is spread over a longer request at 1M, the same memory reports more tokens per GiB at 1M (~25.9K tok/GiB) than at 262K (~19.5K tok/GiB).

| KV GiB (BF16) | Bytes | Blocks | Pool tokens @1M | Max concurrency @1M |
|---|---|---|---|---|
| 72 | 77,309,411,328 | 32,768 | 1,868,290 (est.) | 1.78x (est.) |
| 76 | 81,604,378,624 | 34,588 | 1,972,059 (est.) | 1.88x (est.) |
| **80** | **85,899,345,920** | **36,408** | **2,075,828 (measured)** | **1.98x (measured)** |
| 84 | 90,194,313,216 | 38,229 | 2,179,653 (est.) | 2.08x (est.) |

The block model reproduced both measured BF16 runs exactly, so the estimated rows should be close.

In practice, one 1M-token request needs 18,391 BF16 blocks (about 40.4 GiB), so 80 GiB of BF16 KV holds one full 1M session with room left for shorter ones.

### FP8 KV

- FP8 stores attention K and V in 1 byte (E4M3, unit scales) instead of 2. The conv-state caches stay BF16.
- The attention and conv page sizes no longer line up, so vLLM's hybrid KV manager cannot merge layers the way it does for BF16. The log shows **99 KV groups**: 48 conv-state groups (block 16), 48 attention groups (block 128) and 3 DSpark groups (block 64). BF16 has 13.
- vLLM keeps block tables per group, sized `[max_num_seqs, blocks for max_model_len]` (`vllm/v1/worker/gpu/block_table.py`), and `max_num_seqs` defaults to 256 on GB10. With 99 groups at 1M, this and the other per-group buffers took about 5 GiB more memory outside the KV pool: 4.7 to 5.9 GiB less MemAvailable than BF16 at the same startup stages.
- Measured pools: 80 GiB gives 3,577,201 tokens (3.41x at 1M, but did not fit in memory, section 6). 74 GiB gives **3,308,911 tokens (3.16x at 1M)**.
- That is 44,715 tokens per GiB at 1M. One full 1M request takes about 23.5 GiB of FP8 KV, against 40.4 GiB of BF16 KV: **1.72x the tokens per GiB**.

| KV GiB (FP8) | Pool tokens @1M | Max concurrency @1M |
|---|---|---|
| 66 | 2,951,191 (est.) | 2.81x (est.) |
| 70 | 3,130,051 (est.) | 2.99x (est.) |
| **74** | **3,308,911 (measured)** | **3.16x (measured)** |
| 78 | 3,487,771 (est.) | 3.33x (est.), likely too tight on memory |
| 80 | 3,577,201 (measured) | 3.41x (measured), killed in warmup at 1.9 GiB MemAvailable |

The estimates are linear between the two measured points. Not tested: a lower `--max-num-seqs` (for example 32) should shrink those per-group block tables and might let FP8 use more of the 80 GiB.

## 8. Using it

OpenAI-compatible at `http://localhost:18120/v1`. Model name: `inkling-2-mini` (or `thinkingmachines/Inkling-2-Mini`).

**Reasoning effort** is passed as `chat_template_kwargs.reasoning_effort`, and the chat template turns it into a system line ("Thinking effort level: X").
- Named levels: none = 0.0, minimal = 0.1, low = 0.2, medium = 0.7, high = 0.9, xhigh = 0.99, max = 0.99.
- You can also pass a number from 0.0 to 0.99.
- The default is 0.9 (high). `none` still thinks briefly.

```bash
curl -s http://localhost:18120/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
  "max_tokens": 1024,
  "chat_template_kwargs": {"reasoning_effort": "low"}
}' | python3 -c 'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print("THINK:", (m.get("reasoning_content") or "")[:200]); print("ANSWER:", m["content"])'
```

With the OpenAI Python client, pass it through `extra_body`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18120/v1", api_key="none")
r = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "Summarize the plot of Hamlet in two sentences."}],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(r.choices[0].message.content)
```

**Tool calling** (auto tool choice, inkling parser):

```bash
curl -s http://localhost:18120/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "What is the weather in Atlanta right now?"}],
  "tools": [{"type": "function", "function": {
     "name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "tool_choice": "auto",
  "chat_template_kwargs": {"reasoning_effort": "low"}
}' | python3 -m json.tool
```

The reply has `choices[0].message.tool_calls[0].function` = `get_weather` with `{"city": "Atlanta"}`. Send the result back as a `{"role": "tool", "tool_call_id": ..., "content": ...}` message.

**Images:** send standard OpenAI `image_url` content parts. Each image dimension should be between 40 and 4096 px. The DSpark drafter only sees text, so the decode speed-up is lower on image-heavy turns.

## 9. Speed and sanity checks

**Standard tier** (measured 2026-11-20):
- About 40 to 45 tok/s for one user with DSpark on, and 50 to 62 tok/s in the vLLM log during the 69-scenario eval.
- DSpark mean acceptance length over 37 log windows was 3.1 tokens per step (range 2.0 to 4.3).

**Max BF16 tier, 80 GiB** (measured 2026-11-21):

| Check | Result |
|---|---|
| KV pool | **2,075,828 tokens**, **1.98x at 1,048,576** (80.0 GiB, 36,408 blocks, matches the block model exactly) |
| Startup | 5.5 min: weights 104.8 s, init engine 193.5 s (graphs 100 s, 2.83 GiB) |
| MemAvailable | 6.3 GiB lowest during the JIT warmup, 7.2 to 7.4 GiB idle, 7.0 GiB lowest during the checks |
| Chat 1: "17 * 23" (effort low) | PASS, "391", 51 tokens, TTFT 1.47 s, 72.8 tok/s |
| Chat 2: 8 planets with facts (effort low) | PASS, 254 tokens, TTFT 1.45 s, 37.9 tok/s |
| Needle, 199,989-token prompt, needle at 50% depth | **PASS**, TTFT 119.1 s (~1,680 tok/s prefill), decode 28.3 tok/s (149 tokens) |

**Max FP8 tier, 74 GiB** (measured 2026-11-21):

| Check | Result |
|---|---|
| KV pool | **3,308,911 tokens**, **3.16x at 1,048,576** (74.0 GiB, 99 KV groups) |
| Startup | 5.8 min: weights 115.2 s, init engine 201.5 s (graphs 103 s, 2.99 GiB) |
| MemAvailable | 7.3 GiB lowest during startup, 8.2 to 8.4 GiB idle, 7.5 GiB lowest during the checks |
| Chat 1: "17 * 23" (effort low) | PASS, "391", 51 tokens, TTFT 1.79 s, 72.9 tok/s |
| Chat 2: 8 planets with facts (effort low) | PASS, 270 tokens, TTFT 1.73 s, 37.8 tok/s |
| Needle, 199,989-token prompt, needle at 50% depth | **PASS**, TTFT 118.2 s (1,693 tok/s prefill), decode 47.2 tok/s (77 tokens) |
| Needle, **949,953-token prompt**, needle at 50% depth | **PASS**, answered "VIOLET-HARBOR-7392". **TTFT 1,588 s (26.5 min)**, 598 tok/s average prefill, decode 18.4 tok/s at 950K context (91 tokens) |

Notes:
- **Prefill slows down as the context grows.** It averages about 1,690 tok/s over a 200K prompt and 598 tok/s over a 950K prompt. Between about 700K and 950K it ran at roughly 350 tok/s (from vLLM's KV usage counter). Budget about 2 minutes to the first token for 200K and about 27 minutes for a near-1M prompt. Prefix caching is on, so a follow-up turn on the same long prompt does not pay that again.
- The decode numbers on the needle rows come from short answers (77 to 149 tokens) and depend on how many DSpark drafts get accepted, so treat them as rough. Short-chat decode is about the same on all tiers (roughly 35 to 75 tok/s).
- We ran the 1M needle on the FP8 tier only. The BF16 max tier has the same 1M limit but was checked at 200K.

`sanity.py` (Python stdlib only) runs the light checks, one request at a time:

```bash
python3 sanity.py chat                       # 2 short chats: correctness, TTFT, decode tok/s
python3 sanity.py needle --tokens 200000     # ~200K filler + one needle at 50% depth, reports TTFT
python3 sanity.py needle --tokens 950000     # the 1M check (max tiers only, about 27 minutes)
BASE=http://other-host:18120 python3 sanity.py chat
```

For throughput numbers, use `vllm bench serve` against `/v1/chat/completions` with realistic text (the official recipe uses SPEED-Bench). Random-token prompts accept almost no DSpark drafts.

## 10. 69-scenario tool eval

Our 69-scenario tool-calling eval ([2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval); multi-turn tool loops with mock tool results, Pass/Partial/Fail = 2/1/0) against the server: temperature 0, max 8,192 tokens per turn, `reasoning_effort: "none"` (the lowest; the model still thinks briefly), tools through `--tool-call-parser inkling`, one request at a time. Scorecards: [`results/`](results/).

| | BF16 KV (standard tier, 262K) | FP8 KV (max FP8 tier, 1M, 74 GiB) |
|---|---|---|
| Quality | **88.4** / 100 (122 of 138 points) | **87.0** / 100 (120 of 138 points) |
| Result | 56 pass, 10 partial, 3 fail | 55 pass, 10 partial, 4 fail |
| Responsiveness | 92.1 (median turn latency 2.04 s) | 92.7 (median turn latency 1.93 s) |
| Deployability | 89.5 | 88.7 |
| Decode | 47.7 tok/s | 47.3 tok/s |
| Tokens | 82,637 for the whole run | 77,954 for the whole run |

The max BF16 tier uses the same weights and the same BF16 KV as the standard tier, so it was not scored separately.

**BF16 vs FP8:** 68 of the 69 scenarios scored the same. The one change is TC-22 (Toolset Scale: "Translate 'good morning, my friend' into Italian" with 52 tools offered). BF16 called `translate_text`. FP8 answered the translation itself, correctly, without calling a tool, and did so 3 times out of 3 on re-runs at temperature 0. That is a small behavior shift, not broken output.

| category | BF16 pass / partial / fail | FP8 pass / partial / fail |
|---|---|---|
| Context Retention, Contradictory Parameters, Format Compliance, Omitted Required Parameter, Parameter Precision, Structured Output, Tool Selection | all pass | all pass |
| Toolset Scale | 4 / 0 / 0 | 3 / 0 / 1 (TC-22, above) |
| Multi-Step Chains | 4 / 1 / 0 | 4 / 1 / 0 |
| Safety & Boundaries | 5 / 1 / 0 | 5 / 1 / 0 |
| Instruction Following | 4 / 0 / 1 | 4 / 0 / 1 |
| Ambiguity Handling | 2 / 2 / 0 | 2 / 2 / 0 |
| Refusal Calibration | 3 / 1 / 1 | 3 / 1 / 1 |
| Error Recovery | 2 / 1 / 1 | 2 / 1 / 1 |
| Hallucinated Tools | 0 / 4 / 0 | 0 / 4 / 0 |

Apart from TC-22, the lost points are the same on both: four "hallucinated tools" partials (it never invented a tool, but did not clearly say what it could not do), error recovery (ignored a flight-search error, unclear handling of an empty result), refusals (it did not refuse one clearly malicious request, TC-58, and two refusals were unclear), one skipped tool call (TC-15), two ambiguity partials where it guessed instead of asking, and one chain that skipped the final comparison math.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Startup dies or the box gets sluggish while the KV allocates (max tiers) | `./stop.sh`, then step down 4 GiB: `KV_GIB=76 ./launch_maxkv.sh` (then 72), or `KV_GIB=70 ./launch_maxkv_fp8.sh`. Watch `free -g` and keep MemAvailable above ~6 GiB after warmup. |
| FP8 tier drops toward 2 GiB MemAvailable during the JIT warmup | You are running FP8 with the BF16 budget. FP8 KV needs about 5 GiB more outside the pool (section 7). Use the default `KV_GIB=74` or lower. |
| The box stops answering ssh during startup | Unified memory ran out. Power cycle it, then relaunch with a smaller `KV_GIB` and run `./watch_start.sh` alongside. |
| `unrecognized arguments: --kv-cache-memory=...` | Use `--kv-cache-memory-bytes`. The log message names the flag wrong. |
| KV smaller than expected on the standard tier | Expected. gpu-memory-utilization 0.80 keeps 16.6 GiB out of KV on purpose (vLLM warns about it). That headroom is what the OS lives on. |
| `num_speculative_tokens` error | It must be 7 for this checkpoint. |
| `Inkling2 KV cache dtype must be one of ...` | fp8 KV only works on sm_12x in this build, and fp16 KV is not allowed with spec decode. Use `auto` (BF16) or `fp8`. |
| First few requests have latency spikes, with `Triton kernel JIT compilation during inference` warnings | One-time JIT for shapes the warmup did not cover. Harmless, gone after the first requests. |
| Every answer starts with a long think | The default effort is 0.9. Pass `reasoning_effort` "low" or "none". |
| `content` is empty but tokens were used | The model spent `max_tokens` thinking. Raise `max_tokens` or lower the effort. |
| Low acceptance or slow decode in a benchmark | Use realistic text, not random tokens. Check that `mtp.safetensors` exists and the log shows `SpeculativeConfig(method='dspark' ...)`. |
| Allocation fails right after a large download or copy | GB10 page-cache trap: `sync; echo 1 \| sudo tee /proc/sys/vm/drop_caches`, then relaunch. The max-KV launchers do this once automatically (needs passwordless sudo, skipped otherwise). |
| Want to silence the 0.80 warning | Leave it. Using 1.0 on unified memory starves the OS. |
| Long prompts time out in the client | Prefill is the slow part and slows down with length: about 2 minutes for 200K and 26.5 minutes for 950K (section 9). Set client timeouts of 30+ minutes for near-1M prompts. |

## 12. Files

| File | What |
|---|---|
| `launch_standard.sh` | 262K context, util 0.80, BF16 KV |
| `launch_maxkv.sh` | 1M context, BF16 KV, `KV_GIB` (default 80) |
| `launch_maxkv_fp8.sh` | 1M context, FP8 KV, `KV_GIB` (default 74) |
| `stop.sh` | stops and removes the container (any tier) |
| `watch_start.sh` | logs MemAvailable every 5 s during startup, kills the container under 3 GiB |
| `sanity.py` | light checks: 2 chats, and a needle at any length |
| `results/` | the 69-scenario scorecards behind section 10 |

## Credits

- [Thinking Machines Lab](https://huggingface.co/thinkingmachines) for Inkling 2 Mini, its NVFP4/MXFP8 checkpoint and the DSpark drafter ([model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache 2.0).
- The [vLLM](https://github.com/vllm-project/vllm) project for Inkling 2 support, the [`vllm/vllm-openai:inkling-2-mini`](https://hub.docker.com/r/vllm/vllm-openai/tags?name=inkling-2-mini) image and the [Inkling 2 Mini recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) this one starts from.
- NVIDIA for the DGX Spark platform.
- The 69-scenario tool eval ([2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval)) follows the Pass/Partial/Fail method of wolttam's `tool-eval-bench`.

## License

The scripts and docs in this repository are MIT ([LICENSE](LICENSE)). The model weights are not included. Inkling 2 Mini is Apache 2.0 from Thinking Machines Lab, and vLLM is Apache 2.0.
