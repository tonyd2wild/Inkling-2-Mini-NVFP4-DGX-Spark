#!/usr/bin/env python3
"""Recompute Inkling 2 Mini's DFlash 2 drafts from an engine dump (TF_INKLING2_DFLASH2_DUMP=<dir>) and compare.

A CPU torch reference of incoai/Inkling-2-Mini-DFlash2 as vLLM's qwen3_dflash.py / qwen3_dflash2.py run it with the
Inkling fixes (target_layer_ids used as visits, draft inputs through the target's embed_norm, soft-cap after the
selector correction): fc and hidden_norm over the dumped tapped features, k_norm and RoPE'd context keys, an 8-row
block [pending, mask x 7] through five layers with grouped dynamic convs and sliding non-causal attention, the
target's head over rows 1..7, the 16 likeliest real tokens, and the selector's greedy walk.

usage: inkling2_dflash2_ref.py DUMP_DIR DRAFTER_DIR TARGET_DIR
Reports per hold: cosine of the engine's final hidden rows vs the reference, top-16 overlap, and whether the greedy
drafts match (also when the head and walk start from the engine's own hidden rows, to separate layer from head drift).
"""
import json
import mmap
import os
import re
import struct
import sys

import torch

torch.set_grad_enabled(False)
H, NL, QH, KVH, HD, INTER = 2304, 5, 32, 8, 128, 6912
BLOCK, WINDOW, GROUP, RANK, TOPK = 8, 2048, 16, 256, 16


class Safetensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, name):
        e = self.header[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        a, b = e["data_offsets"]
        count = (b - a) // 2
        t = torch.frombuffer(self.mm, dtype=torch.bfloat16, count=count, offset=self.base + a)
        return t.view(e["shape"])


def bf(x):
    return x.to(torch.bfloat16)


def linear(x, w):
    """bf16 x [m, k] times bf16 w [n, k]^T: fp32 sums, bf16 out (as a BF16 GEMM)."""
    return bf(x.float() @ w.float().t())


def rms(x, w, eps):
    xf = x.float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return bf(bf(xf * r).float() * w.float())


def add_rms(x, resid, w, eps):
    s = x.float() + resid.float()
    r = torch.rsqrt(s.square().mean(-1, keepdim=True) + eps)
    new_resid = bf(s)
    return bf(bf(new_resid.float() * r).float() * w.float()), new_resid


def rope(x, pos, theta):
    """neox RoPE on [rows, heads, 128] at positions pos [rows]."""
    inv = theta ** (-torch.arange(0, 64, dtype=torch.float32) / 64.0)
    ang = pos.float()[:, None] * inv[None, :]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    xf = x.float()
    a, b = xf[..., :64], xf[..., 64:]
    return bf(torch.cat((a * c - b * s, b * c + a * s), dim=-1))


def conv(x, dyn, side, base):
    """DFlashGroupedConv, two taps, one block of 8 rows: fp32 sums, one bf16 rounding."""
    g = H // GROUP
    d = dyn.float().view(-1, 2, 2, g)[:, side]  # [rows, taps, groups]
    coef = base.float()[side].view(1, 2, g, GROUP) + d.unsqueeze(-1)  # [rows, taps, groups, 16]
    xb = x.float().view(-1, g, GROUP)
    out = coef[:, 0] * xb
    out[1:] += coef[1:, 1] * xb[:-1]
    return bf(out.reshape(-1, H))


def main():
    dump, ddir, tdir = sys.argv[1:4]
    dcfg = json.load(open(os.path.join(ddir, "config.json")))
    eps = float(dcfg.get("rms_norm_eps", 1e-6))
    theta = float(dcfg["rope_parameters"]["rope_theta"])
    mask_id = int(dcfg["dflash_config"]["mask_token_id"])
    cap = float(dcfg["dflash_config"]["final_logit_softcapping"])
    tcfg = json.load(open(os.path.join(tdir, "config.json")))
    tt = tcfg.get("text_config", tcfg)
    teps = float(tt.get("rms_norm_eps", 1e-6))
    vocab = None
    D = Safetensors(os.path.join(ddir, "model.safetensors"))
    T = Safetensors(os.path.join(tdir, "model-00001-of-00001.safetensors"))
    embed, embed_norm, unembed = T.get("model.llm.embed.weight"), T.get("model.llm.embed_norm.weight"), T.get("model.llm.unembed.weight")
    vocab = int(open(os.path.join(dump, "vocab")).read()) if os.path.exists(os.path.join(dump, "vocab")) else 200058
    L = []
    for i in range(NL):
        p = f"layers.{i}."
        L.append({k: D.get(p + k) for k in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
            "attention_conv.kernel_projection.weight", "attention_conv.base_kernel",
            "mlp_conv.kernel_projection.weight", "mlp_conv.base_kernel")})
    fc, hnorm, norm = D.get("fc.weight"), D.get("hidden_norm.weight"), D.get("norm.weight")
    hproj = D.get("candidate_selector.hidden_projection.weight")
    pred, succ = D.get("candidate_selector.predecessor_codebook"), D.get("candidate_selector.successor_codebook")

    feats = {}  # position -> bf16 [4 * H], the latest absorb's
    files = sorted(os.listdir(dump))
    for name in files:
        m = re.match(r"absorb_(\d+)_(\d+)_(\d+)\.bin$", name)
        if not m:
            continue
        start, rows = int(m[2]), int(m[3])
        raw = open(os.path.join(dump, name), "rb").read()
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(rows, 4 * H)
        for r in range(rows):
            feats[start + r] = t[r]

    def embed_rows(ids):
        return rms(embed[torch.tensor(ids)], embed_norm, teps)

    def head_walk(hidden, pending):
        """hidden bf16 [7, H] (block rows 1..7): top-16 ids/vals over the real vocab, greedy selector drafts."""
        logits = torch.cat([bf(hidden.float() @ unembed[a:a + 32768].float().t()).float()
                            for a in range(0, vocab, 32768)], dim=1)[:, :vocab]
        vals, ids = torch.topk(logits, TOPK, dim=-1)
        hid = linear(hidden, hproj)
        prev, drafts = pending, []
        for e in range(hidden.shape[0]):
            ah = bf(pred[prev].float() * hid[e].float())
            edges = succ[ids[e]].float() @ ah.float()
            s = vals[e] + edges
            s = cap * torch.tanh(s / cap)
            prev = int(ids[e][int(torch.argmax(s))])
            drafts.append(prev)
        return ids, vals, drafts

    for name in files:
        m = re.match(r"hold_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_x\.bin$", name)
        if not m:
            continue
        k, A, lo, pending, n = (int(v) for v in m.groups())
        tag = name[:-len("_x.bin")]
        ex = torch.frombuffer(bytearray(open(os.path.join(dump, tag + "_x.bin"), "rb").read()), dtype=torch.bfloat16).view(BLOCK, H)
        eids = torch.frombuffer(bytearray(open(os.path.join(dump, tag + "_ids.bin"), "rb").read()), dtype=torch.int32).view(BLOCK - 1, TOPK)
        ewalk = torch.frombuffer(bytearray(open(os.path.join(dump, tag + "_walk.bin"), "rb").read()), dtype=torch.int32)
        missing = [p for p in range(lo, A) if p not in feats]
        if missing:
            print(f"hold {k}: anchor {A}: {len(missing)} context rows not in the dump (first {missing[0]}); skipped")
            continue
        pos_c = torch.arange(lo, A)
        if A > lo:
            ctx = linear(torch.stack([feats[p] for p in range(lo, A)]), fc)
            hn = rms(ctx, hnorm, eps)
        x = embed_rows([pending] + [mask_id] * (BLOCK - 1))
        pos_b = torch.arange(A, A + BLOCK)
        resid = None
        for i, l in enumerate(L):
            if resid is None:
                resid = x
                h = rms(x, l["input_layernorm.weight"], eps)
            else:
                h, resid = add_rms(x, resid, l["input_layernorm.weight"], eps)
            dyn = linear(h, l["attention_conv.kernel_projection.weight"])
            hc = conv(h, dyn, 0, l["attention_conv.base_kernel"])
            q = linear(hc, l["self_attn.q_proj.weight"]).view(BLOCK, QH, HD)
            kb = linear(hc, l["self_attn.k_proj.weight"]).view(BLOCK, KVH, HD)
            vb = linear(hc, l["self_attn.v_proj.weight"]).view(BLOCK, KVH, HD)
            q = rope(rms(q, l["self_attn.q_norm.weight"], eps), pos_b, theta)
            kb = rope(rms(kb, l["self_attn.k_norm.weight"], eps), pos_b, theta)
            if A > lo:
                kc = rope(rms(linear(hn, l["self_attn.k_proj.weight"]).view(-1, KVH, HD), l["self_attn.k_norm.weight"], eps), pos_c, theta)
                vc = linear(hn, l["self_attn.v_proj.weight"]).view(-1, KVH, HD)
                K, V = torch.cat((kc, kb)), torch.cat((vc, vb))
                kpos = torch.cat((pos_c, pos_b))
            else:
                K, V, kpos = kb, vb, pos_b
            # row t sees block keys and context keys p > A + t - WINDOW
            live = (kpos[None, :] >= A) | (kpos[None, :] > (pos_b[:, None] - WINDOW))
            Kq = K.float().repeat_interleave(QH // KVH, dim=1)  # [keys, 32, 128]
            Vq = V.float().repeat_interleave(QH // KVH, dim=1)
            s = torch.einsum("thd,khd->htk", q.float(), Kq) / (HD ** 0.5)
            s = s.masked_fill(~live[None], float("-inf"))
            p = torch.softmax(s, dim=-1)
            att = bf(torch.einsum("htk,khd->thd", p, Vq).reshape(BLOCK, QH * HD))
            y = linear(att, l["self_attn.o_proj.weight"])
            y = conv(y, dyn, 1, l["attention_conv.base_kernel"])
            h, resid = add_rms(y, resid, l["post_attention_layernorm.weight"], eps)
            dyn = linear(h, l["mlp_conv.kernel_projection.weight"])
            hc = conv(h, dyn, 0, l["mlp_conv.base_kernel"])
            g, u = linear(hc, l["mlp.gate_proj.weight"]), linear(hc, l["mlp.up_proj.weight"])
            act = bf(torch.nn.functional.silu(g.float()) * u.float())
            y = linear(act, l["mlp.down_proj.weight"])
            x = conv(y, dyn, 1, l["mlp_conv.base_kernel"])
        out, _ = add_rms(x, resid, norm, eps)
        cos = torch.nn.functional.cosine_similarity(out.float(), ex.float(), dim=-1)
        ids, vals, drafts = head_walk(out[1:1 + n], pending)
        ids_e, _, drafts_e = head_walk(ex[1:1 + n], pending)
        overlap = [len(set(ids[e].tolist()) & set(eids[e].tolist())) for e in range(n)]
        engine = ewalk[1:1 + n].tolist()
        print(f"hold {k}: anchor {A} lo {lo} pending {pending} n {n}: hidden cos min {cos.min():.5f} mean {cos.mean():.5f}; "
              f"top16 overlap {overlap}; drafts ref {drafts} engine {engine} "
              f"{'MATCH' if drafts == engine else 'differ'}; from engine hidden {'MATCH' if drafts_e == engine else drafts_e}")


if __name__ == "__main__":
    main()
