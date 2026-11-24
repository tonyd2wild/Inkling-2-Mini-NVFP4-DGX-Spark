# Inkling 2 Mini on one DGX Spark: vLLM vs TensorFold

TensorFold recipe: Mia's AI Lab (github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold @ 6061fa8): 262K x 8 streams, fp8 KV, DSpark drafts, FP8 attention, tensor-core prefill.
Engine: TensorFold by Ash Hart (ashhart) and contributors; Inkling 2 Mini CUDA engine by Jürgen Schmied (TensorFold PR #4441). vLLM: NVFP4, DSpark k7 / DFlash2 k7. GB10, 2026-11-24.

Thinking off, temperature 0, streaming; same scripts and prompts for all three.

## TTFT (streamed, fresh prompts) and prefill

| c | TTFT vLLM DSpark (s) | TTFT vLLM DFlash2 (s) | TTFT TensorFold DSpark (s) |
|---|---|---|---|
| c1 | 0.607 | 0.665 | 0.561 |
| c2 | 1.170 | 1.212 | 0.867 |
| c3 | 3.600 | 3.615 | 1.164 |
| c4 | 3.129 | 4.010 | 1.477 |
| c5 | 2.815 | 2.798 | 1.793 |
| c6 | 3.008 | 3.495 | 2.101 |

| prompt tokens (vLLM / TF) | prefill vLLM DSpark (tok/s) | prefill vLLM DFlash2 (tok/s) | prefill TensorFold DSpark (tok/s) |
|---|---|---|---|
| 974 / 974 | 2,086 | 2,077 | 2,460 |
| 7,670 / 7,670 | 2,802 | 2,825 | 3,354 |
| 30,710 / 30,710 | 2,729 | 2,713 | 3,127 |

## Per prompt type: aggregate tok/s (per-stream decode) [acceptance length]

| type | c | vLLM DSpark | vLLM DFlash2 | TensorFold DSpark | TF / DFlash2 | TF / DSpark |
|---|---|---|---|---|---|---|
| coding | c1 | 71.5 (71.5) [4.33] | 94.6 (94.6) [5.50] | 68.8 (68.8) [2.54] | 0.73x | 0.96x |
| coding | c2 | 106.4 (60.3) [4.59] | 132.8 (78.2) [5.69] | 90.9 (55.8) [2.75] | 0.68x | 0.85x |
| coding | c3 | 114.9 (45.5) [3.96] | 128.9 (62.6) [5.40] | 100.8 (44.4) [2.68] | 0.78x | 0.88x |
| coding | c4 | 106.8 (43.0) [4.01] | 168.5 (56.0) [5.22] | 116.1 (37.6) [2.79] | 0.69x | 1.09x |
| coding | c5 | 137.0 (42.1) [4.29] | 168.7 (52.7) [5.10] | 128.2 (35.2) [2.82] | 0.76x | 0.94x |
| coding | c6 | 156.3 (40.5) [4.34] | 204.2 (47.0) [4.96] | 155.3 (35.0) [2.88] | 0.76x | 0.99x |
| math | c1 | 55.9 (55.9) [3.44] | 82.2 (82.2) [4.83] | 67.5 (67.5) [2.36] | 0.82x | 1.21x |
| math | c2 | 89.9 (51.3) [3.81] | 130.0 (75.4) [5.47] | 93.3 (51.6) [2.58] | 0.72x | 1.04x |
| math | c3 | 107.4 (37.9) [3.44] | 156.0 (57.3) [5.17] | 104.8 (38.8) [2.29] | 0.67x | 0.98x |
| math | c4 | 113.9 (35.2) [3.38] | 158.8 (51.7) [4.88] | 119.5 (34.4) [2.65] | 0.75x | 1.05x |
| math | c5 | 140.2 (31.6) [3.39] | 196.7 (48.4) [4.98] | 141.3 (33.1) [2.55] | 0.72x | 1.01x |
| math | c6 | 151.4 (32.5) [3.48] | 236.3 (49.5) [5.13] | 163.3 (30.4) [2.53] | 0.69x | 1.08x |
| json | c1 | 75.8 (75.8) [4.54] | 91.2 (91.2) [5.33] | 74.2 (74.2) [2.75] | 0.81x | 0.98x |
| json | c2 | 134.3 (74.9) [5.67] | 137.4 (78.0) [5.74] | 85.7 (56.2) [2.49] | 0.62x | 0.64x |
| json | c3 | 122.2 (50.0) [4.15] | 140.0 (57.8) [5.19] | 107.5 (45.9) [2.56] | 0.77x | 0.88x |
| json | c4 | 175.7 (54.7) [4.83] | 202.3 (59.4) [5.67] | 117.8 (36.8) [2.75] | 0.58x | 0.67x |
| json | c5 | 200.0 (41.9) [4.72] | 248.3 (54.9) [5.88] | 131.9 (34.3) [2.72] | 0.53x | 0.66x |
| json | c6 | 238.1 (50.0) [5.12] | 236.6 (52.7) [5.30] | 146.3 (32.7) [2.54] | 0.62x | 0.61x |
| html | c1 | 68.7 (68.7) [4.01] | 83.9 (83.9) [4.89] | 72.3 (72.3) [2.70] | 0.86x | 1.05x |
| html | c2 | 84.1 (66.4) [4.80] | 99.9 (75.5) [5.58] | 96.8 (59.5) [2.82] | 0.97x | 1.15x |
| html | c3 | 113.2 (43.2) [4.20] | 142.7 (54.3) [5.27] | 131.0 (45.9) [3.16] | 0.92x | 1.16x |
| html | c4 | 162.9 (46.3) [4.50] | 200.8 (55.5) [5.53] | 137.5 (37.2) [2.79] | 0.68x | 0.84x |
| html | c5 | 148.9 (45.1) [4.31] | 175.4 (53.0) [5.12] | 138.8 (35.7) [2.71] | 0.79x | 0.93x |
| html | c6 | 135.7 (36.3) [3.91] | 191.7 (44.7) [4.96] | 148.5 (31.9) [2.71] | 0.77x | 1.09x |
| prose | c1 | 37.3 (37.3) [2.19] | 73.0 (73.0) [4.37] | 57.7 (57.7) [1.88] | 0.79x | 1.55x |
| prose | c2 | 51.4 (31.6) [2.31] | 99.4 (58.0) [4.36] | 65.5 (34.3) [1.27] | 0.66x | 1.27x |
| prose | c3 | 57.9 (28.2) [2.34] | 116.1 (52.9) [4.42] | 75.0 (30.8) [1.27] | 0.65x | 1.30x |
| prose | c4 | 73.6 (25.2) [2.26] | 141.0 (42.8) [4.26] | 84.8 (31.3) [1.32] | 0.60x | 1.15x |
| prose | c5 | 85.7 (21.9) [2.21] | 171.3 (41.0) [4.33] | 104.1 (26.1) [1.40] | 0.61x | 1.21x |
| prose | c6 | 99.6 (21.4) [2.27] | 179.1 (38.9) [4.13] | 115.5 (21.9) [1.49] | 0.64x | 1.16x |
| narrative | c1 | 36.6 (36.6) [2.25] | 71.8 (71.8) [4.08] | 52.7 (52.7) [1.77] | 0.73x | 1.44x |
| narrative | c2 | 53.5 (28.2) [2.15] | 112.4 (57.0) [4.36] | 65.7 (35.9) [1.21] | 0.58x | 1.23x |
| narrative | c3 | 65.8 (24.7) [2.27] | 128.6 (44.4) [4.17] | 80.7 (29.2) [1.32] | 0.63x | 1.23x |
| narrative | c4 | 84.9 (22.8) [2.28] | 144.1 (41.2) [3.98] | 92.5 (24.8) [1.09] | 0.64x | 1.09x |
| narrative | c5 | 95.2 (20.7) [2.23] | 159.5 (34.6) [3.92] | 103.7 (21.8) [1.11] | 0.65x | 1.09x |
| narrative | c6 | 107.3 (19.6) [2.27] | 165.4 (31.9) [3.68] | 119.3 (21.6) [1.11] | 0.72x | 1.11x |
| summary | c1 | 52.8 (52.8) [3.19] | 77.0 (77.0) [4.68] | 61.3 (61.3) [2.13] | 0.80x | 1.16x |
| summary | c2 | 63.2 (45.9) [3.20] | 110.9 (73.8) [5.29] | 71.7 (47.7) [2.18] | 0.65x | 1.13x |
| summary | c3 | 75.3 (37.7) [3.19] | 135.0 (48.4) [4.96] | 80.0 (42.6) [2.13] | 0.59x | 1.06x |
| summary | c4 | 83.9 (35.1) [3.18] | 144.5 (42.3) [4.57] | 86.0 (34.7) [1.92] | 0.60x | 1.03x |
| summary | c5 | 101.4 (31.0) [3.19] | 173.3 (42.7) [4.96] | 111.9 (30.5) [2.40] | 0.65x | 1.10x |
| summary | c6 | 116.4 (29.9) [3.16] | 198.2 (39.5) [4.59] | 123.4 (28.2) [2.31] | 0.62x | 1.06x |
| counting | c1 | 92.6 (92.6) [5.19] | 131.7 (131.7) [6.71] | 90.9 (90.9) [3.42] | 0.69x | 0.98x |
| counting | c2 | 156.6 (86.2) [5.45] | 206.0 (110.1) [6.89] | 151.8 (87.2) [4.00] | 0.74x | 0.97x |
| counting | c3 | 179.1 (61.3) [4.93] | 261.4 (93.2) [6.76] | 185.1 (63.9) [4.13] | 0.71x | 1.03x |
| counting | c4 | 199.4 (51.9) [4.54] | 283.1 (76.0) [6.38] | 209.7 (57.4) [3.99] | 0.74x | 1.05x |
| counting | c5 | 243.6 (52.1) [4.91] | 338.7 (75.0) [6.58] | 239.8 (54.0) [3.89] | 0.71x | 0.98x |
| counting | c6 | 289.0 (57.2) [5.13] | 400.6 (75.5) [6.62] | 215.4 (45.9) [3.27] | 0.54x | 0.75x |

## Mean aggregate tok/s over the 8 types

| c | vLLM DSpark | vLLM DFlash2 | TensorFold DSpark | TF / DFlash2 | TF / DSpark |
|---|---|---|---|---|---|
| c1 | 61.4 | 88.2 | 68.2 | 0.77x | 1.11x |
| c2 | 92.4 | 128.6 | 90.2 | 0.70x | 0.98x |
| c3 | 104.5 | 151.1 | 108.1 | 0.72x | 1.03x |
| c4 | 125.1 | 180.4 | 120.5 | 0.67x | 0.96x |
| c5 | 144.0 | 204.0 | 137.5 | 0.67x | 0.95x |
| c6 | 161.7 | 226.5 | 148.4 | 0.66x | 0.92x |

## Quality (auto-graded, c1) and reply length

| type | vLLM DSpark | vLLM DFlash2 | TensorFold DSpark | tokens (med) |
|---|---|---|---|---|
| coding | 1.00 | 1.00 | 1.00 | 281 / 281 / 292 |
| math | 1.00 | 1.00 | 1.00 | 315 / 315 / 337 |
| json | 1.00 | 1.00 | 1.00 | 158 / 158 / 166 |
| html | 0.86 | 0.86 | 1.00 | 417 / 417 / 416 |
| prose | 0.55 | 0.55 | 0.35 | 1024 / 1024 / 1024 |
| narrative | 0.30 | 0.30 | 0.33 | 1024 / 1024 / 1024 |
| summary | 0.80 | 0.80 | 0.80 | 484 / 484 / 474 |
| counting | 0.60 | 0.60 | 0.80 | 894 / 894 / 812 |

Identical c1 outputs (temperature 0): TF vs vLLM DFlash2 13/40, TF vs vLLM DSpark 13/40, vLLM DSpark vs DFlash2 40/40. Equal prompt_tokens TF vs vLLM: 40/40.

Foreign requests seen during the runs: vLLM DSpark 0, vLLM DFlash2 0, TensorFold DSpark 0.

## Mia's recipe checks (her tools, her defaults)

- `tools/exact.py`: PASS at pad 0 (6/6 drafted == `draft:false` == 6-way concurrent, 508 drafts accepted) and pad 3k (498 accepted). README: 494-503.
- `tools/needle.py` 200K (199,951 prompt tokens, needle at 60%): cold prefill 98.0 s (README ~106 s). As shipped the tool asks for `max_tokens=64`, which the reply spent on reasoning, so content was empty (found: no). At `max_tokens=512` (the README's setting) it found `warm-library-4557`, the README's own passphrase.
- Same-prompt check vs vLLM: prompt_tokens equal on 8/8 sanity prompts and 40/40 bench c1 prompts (patch 0010 renders thinking-off as effort 0.0, like vLLM). Text is not byte-identical across engines (fp8 KV, FP8 attention and tensor-core prefill are lossy defaults); math answers 5/5 equal.

## Her settings: approximation of the README's sparkDash numbers

Mia's `tools/bench_streams.py` (temperature 0.7, top_k 20, thinking on, DSpark, 256-token replies, median of 3). The repo does not ship the `tools/bench.py` it imports, so it ran through a stand-in (`spark_tf/bench.py`, built on her `tools/client.py`). This is an approximation, not sparkDash.

| decode | ours x1 | README x1 | ours x2 | README x2 | ours x4 | README x4 | ours x8 | README x8 |
|---|---|---|---|---|---|---|---|---|
| prose agg tok/s | 47.4 | 48.3 | 70.1 | 74.9 | 109.5 | 106.1 | 162.5 | 145.0 |
| code agg tok/s | 49.1 | 56.1 | 75.7 | 86.5 | 118.2 | 108.5 | 178.5 | 149.4 |

| prefill (ours) | 6,146 | 12,204 | 24,445 | 48,770 | 97,554 | 195,244 |
|---|---|---|---|---|---|---|
| tok/s | 3,363 | 3,312 | 3,220 | 2,998 | 2,590 | 2,049 |

README prefill: 4k 3,497 / 8k 3,344 / 16k 3,253 / 32k 3,072 / 64k 2,750 / 128k 2,267 tok/s (bench_streams' word-count sizes land at 6K-195K tokens, not 4K-128K).

## Notes

- Acceptance length is not like-for-like: vLLM verifies a fixed 7-token draft every round; TensorFold's DSpark drafts adaptively and verifies far fewer tokens a round at a high hit rate (c1: coding 1.9 drafted/round, 81% accepted; math 1.9 drafted/round, 73% accepted; json 2.2 drafted/round, 81% accepted; html 2.1 drafted/round, 80% accepted; prose 1.3 drafted/round, 65% accepted; narrative 1.3 drafted/round, 60% accepted; summary 1.6 drafted/round, 70% accepted; counting 2.7 drafted/round, 89% accepted).
- Prose/narrative auto-scores are dominated by the 1,024-token budget: at effort none the model still reasons, and replies that spend the budget reasoning score 0 on both engines (length-capped c1 replies: vLLM 12/40, TF 10/40).
- TF memory: engine 59.70 GiB + 20.3 GiB prompt cache (14 GiB reserve); device peak 88.1 GiB; lowest MemAvailable during all TF runs 26.5 GiB; guard never fired.
