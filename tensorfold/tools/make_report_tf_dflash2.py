#!/usr/bin/env python3
"""make_report_tf_dflash2.py TF_DSPARK.json VLLM_DFLASH2.json TF_DFLASH2.json OUT_PREFIX "Title" ["footer"] [--extra extra.json]

(make_report3.py with the three configs renamed and the heatmap as TF DFlash2 / TF DSpark.)

Three-config version of make_report.py (same charcoal-and-white house style), from three bench_inkling.py /
bench_inkling_tf.py result files: vLLM DSpark (gray), vLLM DFlash2 (white), TensorFold with Mia's AI Lab recipe,
DSpark drafts (amber).
  row 1: TTFT c1..c6 (streamed, fresh prompts) | prefill tok/s at 1K / 8K / 32K
  row 2: aggregate tok/s c1..c6 (mean over prompt types) | per-type decode tok/s at c1
  row 3: per-type acceptance length at c1 | TensorFold / vLLM DFlash2 aggregate ratio per type x concurrency
Writes OUT_PREFIX.png and OUT_PREFIX.md. --extra: a JSON with quality/exactness notes for the markdown
({"md": ["line", ...]}).
TensorFold recipe: Mia's AI Lab (MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold). Engine: TensorFold (ashhart),
Inkling 2 Mini CUDA engine by Juergen Schmied (TensorFold PR #4441).
"""
import json, statistics, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = [a for a in sys.argv[1:]]
extra = None
if "--extra" in args:
    i = args.index("--extra"); extra = json.load(open(args[i + 1])); del args[i:i + 2]
RUNS = [json.load(open(p)) for p in args[:3]]
OUT, TITLE = args[3], args[4]
FOOT = args[5] if len(args) > 5 else ""
NAMES = ["TensorFold DSpark", "vLLM DFlash2", "TensorFold DFlash2"]
G = "#1B1C1F"; P = "#232428"; INK = "#F2F2F0"; MUT = "#9A9DA3"; RULE = "#3A3C41"; WH = "#F2F2F0"; GY = "#8F9297"; AM = "#E0A64B"
COLS = [GY, WH, AM]
CATS = [c for c in ["coding", "math", "json", "html", "prose", "narrative", "summary", "counting"]
        if all(c in d.get("types", {}) for d in RUNS)]
LEVELS = sorted({r["c"] for r in RUNS[0]["ttft"]["rows"]})


def lvl(d, cat, c, key):
    e = d["types"][cat]
    if c == 1:
        return {"agg": e["c1"]["decode_med_tok_s"], "per": e["c1"]["decode_med_tok_s"], "al": e["c1"]["spec"].get("acceptance_length")}[key]
    for r in e["levels"]:
        if r["c"] == c:
            return {"agg": r["agg_tok_s"], "per": r["per_stream_decode_med_tok_s"], "al": r["spec"].get("acceptance_length")}[key]
    return None


def mean_agg(d, c):
    v = [lvl(d, cat, c, "agg") for cat in CATS]; v = [x for x in v if x]
    return statistics.mean(v) if v else None


def ttft_rows(d):
    return {r["c"]: r for r in d["ttft"]["rows"]}


plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": RULE, "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.facecolor": P, "figure.facecolor": G, "savefig.facecolor": G,
                     "font.size": 12})
fig, axs = plt.subplots(3, 2, figsize=(15, 17.5), dpi=140)


def style(ax, title):
    ax.set_title(title, fontsize=14, loc="left", fontweight="bold", color=INK, pad=10)
    ax.grid(axis="y", color=RULE, lw=0.6); ax.spines[["top", "right"]].set_visible(False)


def bars(ax, labels, series, fmt, rot=0, fs=8.5):
    x = list(range(len(labels))); w = 0.27
    for k, (vals, col, name) in enumerate(zip(series, COLS, NAMES)):
        off = (k - 1) * w
        ax.bar([i + off for i in x], [v or 0 for v in vals], w, color=col, label=name)
        for i, v in enumerate(vals):
            if v: ax.text(i + off, v, fmt(v), ha="center", va="bottom", fontsize=fs, color=INK, rotation=90 if len(fmt(v)) >= 4 else 0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=11)
    ax.legend(frameon=False, fontsize=10.5, labelcolor=INK, loc="upper left", ncol=3)
    top = max([v or 0 for s in series for v in s] + [1]); ax.set_ylim(0, top * 1.28)


# 1 TTFT
ax = axs[0][0]
bars(ax, [f"c{c}" for c in LEVELS], [[ttft_rows(d)[c]["ttft_med_s"] * 1000 if c in ttft_rows(d) else None for c in LEVELS] for d in RUNS],
     lambda v: f"{v:.0f}")
style(ax, f"TTFT, ms (median, {RUNS[0]['ttft']['rows'][0]['prompt_tokens']:,}-token fresh prompt)")
# 2 prefill
ax = axs[0][1]
pr = [d["prefill"]["rows"] for d in RUNS]
bars(ax, [f"{r['prompt_tokens'] / 1000:.0f}K" for r in pr[0]], [[r["prefill_tok_s"] for r in p] for p in pr], lambda v: f"{v:,.0f}")
style(ax, "Prefill tok/s (cold prompt, c1)")
# 3 aggregate
ax = axs[1][0]
bars(ax, [f"c{c}" for c in LEVELS], [[mean_agg(d, c) for c in LEVELS] for d in RUNS], lambda v: f"{v:.0f}")
style(ax, "Aggregate tok/s by concurrency (mean of 8 types)")
# 4 per-type decode c1
ax = axs[1][1]
bars(ax, CATS, [[lvl(d, c, 1, "per") for c in CATS] for d in RUNS], lambda v: f"{v:.0f}", rot=30)
style(ax, "Decode tok/s per prompt type, one user (c1)")
# 5 acceptance
ax = axs[2][0]
bars(ax, CATS, [[lvl(d, c, 1, "al") for c in CATS] for d in RUNS], lambda v: f"{v:.1f}", rot=30)
style(ax, "Acceptance length per type, c1 (tokens per target step)")
# 6 ratio heatmap TF DFlash2 / TF DSpark
ax = axs[2][1]
M = [[(lvl(RUNS[2], cat, c, "agg") or 0) / (lvl(RUNS[0], cat, c, "agg") or 1) for c in LEVELS] for cat in CATS]
ax.imshow(M, cmap="Greys_r", vmin=0.3, vmax=max(1.6, max(max(r) for r in M)), aspect="auto")
for i, row in enumerate(M):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=10.5, color=G if v > 0.95 else INK, fontweight="bold")
ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels([f"c{c}" for c in LEVELS], fontsize=11)
ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS, fontsize=11)
ax.set_title("TensorFold DFlash2 / TensorFold DSpark (aggregate tok/s)", fontsize=14, loc="left", fontweight="bold", color=INK, pad=10)
for s in ax.spines.values(): s.set_visible(False)

fig.suptitle(TITLE, fontsize=19, fontweight="bold", color=INK, x=0.02, ha="left", y=0.995)
fig.text(0.02, 0.969, "Thinking off (reasoning_effort none), temperature 0, streaming. Same bench scripts and prompts; "
         "one config on the box at a time, nothing else on the endpoint.", fontsize=11.5, color=MUT)
nl = FOOT.count("\n")
if FOOT: fig.text(0.02, 0.005, FOOT, fontsize=10, color=MUT, va="bottom", linespacing=1.5)
fig.tight_layout(rect=(0, 0.02 + 0.011 * nl, 1, 0.963), h_pad=2.6, w_pad=2.2)
fig.savefig(OUT + ".png"); print("->", OUT + ".png")


# markdown
def f(v, nd=1): return "-" if v is None else f"{v:.{nd}f}"


md = [f"# {TITLE}", "", FOOT, "", "Thinking off, temperature 0, streaming; same scripts and prompts for all three.", "",
      "## TTFT (streamed, fresh prompts) and prefill", "", "| c | " + " | ".join(f"TTFT {n} (s)" for n in NAMES) + " |", "|---|---|---|---|"]
for c in LEVELS:
    md.append(f"| c{c} | " + " | ".join(f(ttft_rows(d).get(c, {}).get("ttft_med_s"), 3) for d in RUNS) + " |")
md += ["", "| prompt tokens (vLLM / TF) | " + " | ".join(f"prefill {n} (tok/s)" for n in NAMES) + " |", "|---|---|---|---|"]
for rows in zip(*pr):
    md.append(f"| {rows[0]['prompt_tokens']:,} / {rows[2]['prompt_tokens']:,} | " + " | ".join(f"{r['prefill_tok_s']:,}" for r in rows) + " |")
md += ["", "## Per prompt type: aggregate tok/s (per-stream decode) [acceptance length]", "",
       "| type | c | " + " | ".join(NAMES) + " | TF DFlash2 / vLLM DFlash2 | TF DFlash2 / TF DSpark |", "|---|---|---|---|---|---|---|"]
for cat in CATS:
    for c in LEVELS:
        a = [lvl(d, cat, c, "agg") for d in RUNS]
        cells = [f"{f(lvl(d, cat, c, 'agg'))} ({f(lvl(d, cat, c, 'per'))}) [{f(lvl(d, cat, c, 'al'), 2)}]" for d in RUNS]
        md.append(f"| {cat} | c{c} | " + " | ".join(cells) + f" | {f(a[2] / a[1] if a[1] and a[2] else None, 2)}x | {f(a[2] / a[0] if a[0] and a[2] else None, 2)}x |")
md += ["", "## Mean aggregate tok/s over the 8 types", "", "| c | " + " | ".join(NAMES) + " | TF DFlash2 / vLLM DFlash2 | TF DFlash2 / TF DSpark |", "|---|---|---|---|---|---|"]
for c in LEVELS:
    a = [mean_agg(d, c) for d in RUNS]
    md.append(f"| c{c} | " + " | ".join(f(x) for x in a) + f" | {f(a[2] / a[1], 2)}x | {f(a[2] / a[0], 2)}x |")
md += ["", "## Quality (auto-graded, c1) and reply length", "", "| type | " + " | ".join(NAMES) + " | tokens (med) |", "|---|---|---|---|---|"]
for cat in CATS:
    e = [d["types"][cat]["c1"] for d in RUNS]
    md.append(f"| {cat} | " + " | ".join(f(x["auto_score"], 2) for x in e) + " | " + " / ".join(str(x["tokens_med"]) for x in e) + " |")
items = [{it["id"]: it for cat in CATS for it in d["types"][cat]["c1"]["items"]} for d in RUNS]
same_tf_df = sum(items[2][k]["output"] == items[1][k]["output"] for k in items[2] if k in items[1])
same_tf_ds = sum(items[2][k]["output"] == items[0][k]["output"] for k in items[2] if k in items[0])
same_ds_df = sum(items[0][k]["output"] == items[1][k]["output"] for k in items[0] if k in items[1])
pt_eq = sum(items[2][k]["prompt_tokens"] == items[1][k]["prompt_tokens"] for k in items[2] if k in items[1])
md += ["", f"Identical c1 outputs (temperature 0): TF vs vLLM DFlash2 {same_tf_df}/{len(items[2])}, TF vs vLLM DSpark {same_tf_ds}/{len(items[2])}, "
       f"vLLM DSpark vs DFlash2 {same_ds_df}/{len(items[0])}. Equal prompt_tokens TF vs vLLM: {pt_eq}/{len(items[2])}.", ""]


def foreign(d):
    return d.get("ttft", {}).get("foreign_requests", 0) + d.get("prefill", {}).get("foreign_requests", 0) + sum(
        d["types"][c]["c1"]["foreign_requests"] + sum(r["foreign_requests"] for r in d["types"][c]["levels"]) for c in CATS)


md.append("Foreign requests seen during the runs: " + ", ".join(f"{n} {foreign(d)}" for n, d in zip(NAMES, RUNS)) + ".")
if extra:
    md += [""] + extra.get("md", [])
open(OUT + ".md", "w").write("\n".join(md) + "\n"); print("->", OUT + ".md")
