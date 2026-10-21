"""evaluate.py -- metrics used in the paper. All functions are model-agnostic unless noted.

  eval_loss        val/train loss + ppl on FIXED windows (same windows for every model and seed)
  dead_neurons     GELU hidden units with |act| < 0.01 for > 95% of tokens   (dense FFN and Zombie Branch B)
                   NOTE: for Zombie, Branch B only sees tokens routed to it, so its % is over routed tokens.
  routing_stats    Zombie only: fraction of tokens routed to B per layer, slot-usage entropy per cache,
                   dead sub-key rows, effective #slots.  (A different metric from dead neurons -- do not compare.)
  fakequant_       per-row symmetric PTQ (weights wbits, cache values cbits) on a COPY of the model.
                   int4 cache in practice also needs one fp16 scale per row (+0.5 B/value overhead per 384 values).
"""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import FFN


def windows(tokens, batch, seq, n_batches, seed, device):
    """Deterministic list-generator of (x, y). Same seed => same windows for every model."""
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        ix = rng.integers(0, len(tokens) - seq - 1, size=batch)
        x = np.stack([tokens[i:i + seq] for i in ix]).astype(np.int64)
        y = np.stack([tokens[i + 1:i + 1 + seq] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def eval_loss(model, batches, amp=None):
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        with (amp() if amp else contextlib.nullcontext()):
            _, loss, _ = model(x, y)
        tot += float(loss)
        n += 1
    model.train(was)
    loss = tot / max(n, 1)
    return dict(loss=loss, ppl=math.exp(min(loss, 50)), batches=n)


@torch.no_grad()
def dead_neurons(model, batches, thr=0.01, frac=0.95):
    """Neuron is dead if |gelu(pre-act)| < thr for > `frac` of the tokens it saw."""
    was = model.training
    model.eval()
    layers = []
    for i, b in enumerate(model.blocks):
        f = b.ffn
        if isinstance(f, FFN):
            layers.append((f"L{i}.ffn", f.w1))
        elif hasattr(f, "B"):
            layers.append((f"L{i}.B", f.B.w1))
        elif hasattr(f, "experts"):                      # MoE: every expert's hidden units (tokens = routed ones)
            layers += [(f"L{i}.e{j}", e.w1) for j, e in enumerate(f.experts)]
    if not layers:
        return None
    dev = next(model.parameters()).device
    stat = {n: [torch.zeros(m.out_features, dtype=torch.long, device=dev), 0] for n, m in layers}
    hooks = []

    def mk(n):
        def h(mod, inp, out):
            a = F.gelu(out.float()).abs().reshape(-1, out.shape[-1])
            stat[n][0] += (a >= thr).sum(0)
            stat[n][1] += a.shape[0]
        return h
    for n, m in layers:
        hooks.append(m.register_forward_hook(mk(n)))
    try:
        for x, _ in batches:
            model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    per, dead, tot = {}, 0, 0
    for n, (cnt, N) in stat.items():
        if N == 0:
            continue
        d = int(((cnt.float() / N) < (1 - frac)).sum())
        per[n] = 100.0 * d / cnt.numel()
        dead += d
        tot += cnt.numel()
    return dict(pct_dead=100.0 * dead / max(tot, 1), per_layer=per, neurons=tot,
                tokens_per_layer={n: s[1] for n, s in stat.items()}, thr=thr, frac=frac)


@torch.no_grad()
def routing_stats(model, batches):
    """Zombie only. Counts every memory lookup and every gate decision made during hard-routed inference."""
    if not hasattr(model, "caches") or not len(model.caches):
        return None
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    counts = [torch.zeros(c.n_slots, dtype=torch.long, device=dev) for c in model.caches]
    rows = [torch.zeros(c.heads, 2, c.n_sub, dtype=torch.long, device=dev) for c in model.caches]
    origs = []
    for k, c in enumerate(model.caches):
        orig = c.lookup
        origs.append(orig)

        def mk(k=k, c=c, orig=orig):
            def lookup(q):
                out, idx, top = orig(q)
                counts[k] += torch.bincount(idx.reshape(-1), minlength=c.n_slots)
                i, j = idx // c.n_sub, idx % c.n_sub
                for h in range(c.heads):
                    rows[k][h, 0] += torch.bincount(i[:, h].reshape(-1), minlength=c.n_sub)
                    rows[k][h, 1] += torch.bincount(j[:, h].reshape(-1), minlength=c.n_sub)
                return out, idx, top
            return lookup
        c.lookup = mk()
    gate_n = {}
    hooks = []
    for li, blk in enumerate(model.blocks):
        f = blk.ffn
        if hasattr(f, "gate"):
            gate_n[li] = [0, 0]

            def mkg(li=li):
                def h(mod, inp, out):
                    gate_n[li][0] += int((out > 0).sum())
                    gate_n[li][1] += out.numel()
                return h
            hooks.append(f.gate.register_forward_hook(mkg()))
    try:
        for x, _ in batches:
            model(x)
    finally:
        for c in model.caches:
            c.__dict__.pop("lookup", None)
        for h in hooks:
            h.remove()
        model.train(was)
    caches = []
    for k, c in enumerate(model.caches):
        cnt = counts[k].float()
        tot = cnt.sum()
        if tot > 0:
            p = cnt[cnt > 0] / tot
            H = float(-(p * p.log()).sum())
        else:
            H = 0.0
        caches.append(dict(slot_entropy=H, slot_entropy_norm=H / math.log(c.n_slots), eff_slots=math.exp(H),
                           used_frac=float((cnt > 0).float().mean()), dead_rows=int((rows[k] == 0).sum()),
                           total_rows=int(rows[k].numel()), lookups=int(tot)))
    p_b = {f"L{li}": (v[0] / v[1] if v[1] else None) for li, v in gate_n.items()}
    vals = [v for v in p_b.values() if v is not None]
    return dict(caches=caches, p_b=p_b, p_b_mean=(sum(vals) / len(vals)) if vals else None)


@torch.no_grad()
def moe_routing(model, batches):
    """MoE only: expert load over fixed eval windows (entropy_norm 1.0 = perfectly balanced, dead = never selected)."""
    was = model.training
    model.eval()
    model.moe_reset()
    for x, _ in batches:
        model(x)
    st = model.moe_stats(reset=True)
    model.train(was)
    return st


@torch.no_grad()
def fakequant_(model, wbits=8, cbits=None):
    """In-place fake quantisation (quantise -> dequantise), per output row, symmetric absmax. Use on a deepcopy."""
    def fq(w, bits):
        qmax = 2 ** (bits - 1) - 1
        s = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        return (w / s).round().clamp(-qmax - 1, qmax) * s
    if wbits:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.copy_(fq(m.weight, wbits))
        for c in getattr(model, "caches", []):
            c.keys.copy_(fq(c.keys, wbits))
    if cbits:
        for c in getattr(model, "caches", []):
            c.values.weight.copy_(fq(c.values.weight, cbits))
    return model
