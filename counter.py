#!/usr/bin/env python3
"""counter.py -- closed-form params / MACs / bytes-read per token. No torch needed.

Definitions (all per token, decode at batch=1, weights streamed once from RAM):
  total params     every nn.Parameter (embedding + LM head + norms + gate + cache keys/values ...)
  active core      params actually touched per token, excluding embedding table and LM head
  MACs             multiply-accumulates of weight matmuls + LM head. Attention *scores*
                   (QK^T, AV) are excluded unless --ctx > 0.
  bytes read       active weights * bytes/weight  (+ gathered cache rows * bytes/value)
                   (+ KV-cache read if --ctx > 0) + embedding row + LM head
Zombie expectations use --p-b = fraction of tokens routed to Branch B at inference.
Numbers are *expected* values under that routing fraction, not measurements.

    python counter.py                  # default tables, fp16 weights
    python counter.py --wb 4 --vb 4    # fp32
    python counter.py --ctx 512        # include attention scores + KV-cache reads
    python counter.py --json out.json
"""
import argparse
import json
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Spec:
    name: str = ""
    d: int = 384
    L: int = 12
    H: int = 8
    vocab: int = 8192
    n_kv: int = 8
    ffn_mult: float = 4.0          # dense FFN hidden = ffn_mult * d ; 0 => no FFN
    mem_layers: tuple = ()         # layers whose FFN is replaced by a Zombie block
    n_groups: int = 2              # shared caches (layer l -> group l*n_groups//L)
    n_sub: int = 128
    pk_heads: int = 2
    half: int = 96
    topk: int = 4
    gate_hidden: int = 32
    gate_mode: str = "learned"     # learned | fixed_half | mem_only
    qnorm: bool = True
    p_b: float = 0.5
    moe_layers: tuple = ()         # layers whose FFN is a top-k mixture of experts
    n_experts: int = 12
    top_k: int = 2
    exp_hidden: int = 164

    @property
    def hd(self):
        return self.d // self.H

    @property
    def qdim(self):
        return self.pk_heads * 2 * self.half

    @property
    def n_slots(self):
        return self.n_sub ** 2

    def group_of(self, l):
        return l * self.n_groups // self.L


def attn_params(s):
    return s.d * s.H * s.hd + 2 * s.d * s.n_kv * s.hd + s.H * s.hd * s.d


def dense_ffn_params(s):
    return 2 * s.d * int(s.ffn_mult * s.d)


def mem_path(s):
    """Memory path (query proj + product-key scoring + value gather) for ONE layer."""
    proj = s.d * s.qdim
    score = s.pk_heads * 2 * s.n_sub * s.half
    gather = s.pk_heads * s.topk * s.d
    return dict(proj=proj, score=score, gather=gather, macs=proj + score + gather)


def b_params(s):
    return 2 * s.d * s.d


def gate_params(s):
    return s.d * s.gate_hidden + s.gate_hidden + s.gate_hidden + 1


def keys_per_group(s):
    return s.pk_heads * 2 * s.n_sub * s.half


def n_groups_used(s):
    return len({s.group_of(l) for l in s.mem_layers})


def ffn_costs(s, l):
    """Cost of the FFN slot of layer l.
    returns dict(params, macs, act_w (weight params touched), act_v (value rows touched))"""
    if l in s.moe_layers:              # router + top_k experts touched per token (batch-1 decode)
        a = 2 * s.d * s.exp_hidden
        act = s.d * s.n_experts + s.top_k * a
        return dict(params=s.d * s.n_experts + s.n_experts * a, macs=act, act_w=act, act_v=0)
    if l not in s.mem_layers:
        n = dense_ffn_params(s)
        return dict(params=n, macs=n, act_w=n, act_v=0)
    m = mem_path(s)
    qn = 2 * s.qdim if s.qnorm else 0
    mem_w = s.d * s.qdim + qn + s.d + keys_per_group(s)    # qproj + BN + scale + sub-keys (read in full)
    mem_v = s.pk_heads * s.topk * s.d                       # gathered value rows
    own = s.d * s.qdim + qn + s.d                           # per-layer (non-shared) memory params
    B = b_params(s)
    G = gate_params(s)
    p = s.p_b
    if s.gate_mode == "mem_only":
        return dict(params=own, macs=m["macs"], act_w=mem_w, act_v=mem_v)
    if s.gate_mode == "fixed_half":
        return dict(params=own + B, macs=m["macs"] + B, act_w=mem_w + B, act_v=mem_v)
    gm = s.d * s.gate_hidden + s.gate_hidden
    return dict(params=own + B + G,
                macs=gm + p * B + (1 - p) * m["macs"],
                act_w=G + p * B + (1 - p) * mem_w,
                act_v=(1 - p) * mem_v)


def model_costs(s, wb=2.0, vb=2.0, ctx=0):
    hd = s.hd
    attn = attn_params(s)
    n_norm = sum(1 + (1 if (l in s.mem_layers or l in s.moe_layers or s.ffn_mult > 0) else 0) for l in range(s.L))
    tot = 2 * s.vocab * s.d + (n_norm + 1) * s.d             # embed + head + all RMSNorm (+final)
    tot += s.L * attn
    macs = s.d * s.vocab                                      # LM head
    act_core = 0.0
    byts = (s.d + s.vocab * s.d + s.d) * wb                   # embed row + head + final norm
    for l in range(s.L):
        f = ffn_costs(s, l)
        tot += f["params"]
        macs += attn + f["macs"]
        act_core += attn + f["act_w"] + f["act_v"]
        nn_l = 1 + (1 if (l in s.mem_layers or l in s.moe_layers or s.ffn_mult > 0) else 0)
        byts += (attn + nn_l * s.d + f["act_w"]) * wb + f["act_v"] * vb
        if ctx:
            macs += 2 * ctx * s.H * hd                         # QK^T + AV
            byts += 2 * ctx * s.n_kv * hd * wb                 # KV-cache read
    gs = n_groups_used(s)
    tot += gs * (keys_per_group(s) + s.n_slots * s.d)
    return dict(name=s.name, total_params=tot, active_core=act_core, macs=macs,
                bytes=byts, cache_params=gs * (keys_per_group(s) + s.n_slots * s.d))


def default_specs(p_b):
    base = Spec(p_b=p_b)
    z = dict(ffn_mult=4.0, n_kv=2)
    return [
        replace(base, name="MHA + 4x FFN", n_kv=8, ffn_mult=4.0),
        replace(base, name="GQA + 4x FFN", n_kv=2, ffn_mult=4.0),
        replace(base, name="GQA + 1x FFN (control)", n_kv=2, ffn_mult=1.0),
        replace(base, name="GQA attn-only", n_kv=2, ffn_mult=0.0),
        replace(base, name="Zombie v6 (12 mem layers)", mem_layers=tuple(range(12)), **z),
        replace(base, name="Zombie v6 (mem layers 6-11)", mem_layers=tuple(range(6, 12)), **z),
        replace(base, name="Zombie mem-only (no B/gate)", mem_layers=tuple(range(12)),
                gate_mode="mem_only", **z),
        replace(base, name="Zombie fixed_half (B+mem)", mem_layers=tuple(range(12)),
                gate_mode="fixed_half", **z),
        # index 8: plain MoE sized to match Zombie v6 on total params AND active MACs (see moe.py)
        replace(base, name="MoE top-2 (12 experts, matched)", n_kv=2, ffn_mult=0.0, moe_layers=tuple(range(12))),
    ]


def fmt_m(x):
    return f"{x / 1e6:8.2f}M"


def print_tables(specs, wb, vb, ctx):
    print(f"\nassumptions: weights {wb:g} B/param, cache values {vb:g} B/param, "
          f"p_b={specs[0].p_b}, ctx={ctx} (0 = attention scores & KV reads excluded)\n")

    # ---- Table 1: FFN slot only, one layer
    print("TABLE 1  FFN slot only (per layer, per token)")
    print(f"{'model':32s} {'MACs':>10s} {'weight bytes':>14s}")
    rows = []
    for s in specs:
        if s.ffn_mult == 0.0 and not s.mem_layers and not s.moe_layers:
            continue
        l = s.mem_layers[0] if s.mem_layers else (s.moe_layers[0] if s.moe_layers else 0)
        f = ffn_costs(s, l)
        by = f["act_w"] * wb + f["act_v"] * vb
        rows.append((s.name, f["macs"], by))
        print(f"{s.name:32s} {f['macs'] / 1e3:8.0f}k {by / 1e6:12.3f}MB")
    ref = rows[1]
    print("-- vs GQA+4xFFN: " + ", ".join(
        f"{n.split('(')[0].strip()} {ref[1] / m:.1f}x MACs / {ref[2] / b:.1f}x bytes"
        for n, m, b in rows if n.startswith("Zombie v6 (12")))

    # ---- Table 2: whole model
    print("\nTABLE 2  whole model, per token (decode, batch=1)")
    print(f"{'model':32s} {'total':>9s} {'active core':>11s} {'MACs':>9s} {'MB read':>9s}")
    res = [model_costs(s, wb, vb, ctx) for s in specs]
    for r in res:
        print(f"{r['name']:32s} {fmt_m(r['total_params'])} {fmt_m(r['active_core'])}"
              f" {fmt_m(r['macs'])} {r['bytes'] / 1e6:8.2f}M")
    gqa = res[1]
    z = res[4]
    ctrl = res[2]
    print(f"\nZombie v6 vs GQA+4xFFN    : {gqa['macs'] / z['macs']:.2f}x fewer MACs, "
          f"{gqa['bytes'] / z['bytes']:.2f}x fewer bytes")
    print(f"Zombie v6 vs GQA+1xFFN    : {ctrl['macs'] / z['macs']:.2f}x MACs, "
          f"{ctrl['bytes'] / z['bytes']:.2f}x bytes   <- the control that matters")
    if len(res) > 8:
        mo = res[8]
        print(f"Zombie v6 vs MoE (matched) : total {z['total_params'] / 1e6:.2f}M vs {mo['total_params'] / 1e6:.2f}M | "
              f"active core {z['active_core'] / 1e6:.2f}M vs {mo['active_core'] / 1e6:.2f}M | "
              f"MACs {z['macs'] / 1e6:.2f}M vs {mo['macs'] / 1e6:.2f}M | MB read {z['bytes'] / 1e6:.2f} vs {mo['bytes'] / 1e6:.2f}"
              f"  (batch-1 decode; MoE reads more experts as batch grows)")
    print(f"Zombie total params {z['total_params'] / 1e6:.2f}M vs GQA {gqa['total_params'] / 1e6:.2f}M "
          f"(cache = {z['cache_params'] / 1e6:.2f}M params; "
          f"{z['cache_params'] * 0.5 / 1e6:.1f}MB int4 / {z['cache_params'] * 2 / 1e6:.1f}MB fp16)")

    # ---- routing sweep
    print("\nTABLE 3  Zombie v6 (12 mem layers) vs fraction p_b routed to Branch B")
    print(f"{'p_b':>5s} {'MACs':>9s} {'MB read':>9s}")
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = model_costs(replace(specs[4], p_b=p), wb, vb, ctx)
        print(f"{p:5.2f} {fmt_m(r['macs'])} {r['bytes'] / 1e6:8.2f}M")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wb", type=float, default=2.0, help="bytes per weight (2=fp16, 4=fp32, 1=int8)")
    ap.add_argument("--vb", type=float, default=None, help="bytes per cache value (default = --wb; 0.5 = int4)")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--p-b", type=float, default=0.5)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    vb = a.wb if a.vb is None else a.vb
    res = print_tables(default_specs(a.p_b), a.wb, vb, a.ctx)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
