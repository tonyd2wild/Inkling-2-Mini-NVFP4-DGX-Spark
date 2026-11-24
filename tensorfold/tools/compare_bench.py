#!/usr/bin/env python3
"""compare_bench.py A.json B.json [labelA labelB]: per-type c1 decode and acceptance, aggregate c1..c6, TTFT, prefill
from two bench_inkling_tf.py result files, as a markdown table (B over A ratios)."""
import json, statistics, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
la, lb = (sys.argv[3:5] if len(sys.argv) > 4 else ("A", "B"))
cats = [c for c in ["coding", "math", "json", "html", "prose", "narrative", "summary", "counting"] if c in a["types"] and c in b["types"]]


def c1(d, c, k):
    e = d["types"][c]["c1"]
    return e["decode_med_tok_s"] if k == "dec" else e["spec"].get("acceptance_length")


def agg(d, c, lvl):
    if lvl == 1:
        return d["types"][c]["c1"]["decode_med_tok_s"]
    for r in d["types"][c]["levels"]:
        if r["c"] == lvl:
            return r["agg_tok_s"]
    return None


print(f"| type | {la} c1 tok/s | {lb} c1 tok/s | ratio | {la} AL | {lb} AL |")
print("|---|---|---|---|---|---|")
for c in cats:
    x, y = c1(a, c, "dec"), c1(b, c, "dec")
    print(f"| {c} | {x:.1f} | {y:.1f} | {y / x:.2f}x | {c1(a, c, 'al') or 0:.2f} | {c1(b, c, 'al') or 0:.2f} |")
mx = statistics.mean(c1(a, c, "dec") for c in cats)
my = statistics.mean(c1(b, c, "dec") for c in cats)
print(f"| **mean** | {mx:.1f} | {my:.1f} | {my / mx:.2f}x | | |")
levels = sorted({r["c"] for r in a["ttft"]["rows"]})
print()
print(f"| concurrency | {la} agg tok/s | {lb} agg tok/s | ratio |")
print("|---|---|---|---|")
for lvl in levels:
    xs = [agg(a, c, lvl) for c in cats]
    ys = [agg(b, c, lvl) for c in cats]
    if None in xs or None in ys:
        continue
    x, y = statistics.mean(xs), statistics.mean(ys)
    print(f"| c{lvl} | {x:.1f} | {y:.1f} | {y / x:.2f}x |")
print()
for k in ("ttft", "prefill"):
    if k in a and k in b:
        print(k, la, json.dumps(a[k].get("rows", a[k]))[:400])
        print(k, lb, json.dumps(b[k].get("rows", b[k]))[:400])
