#!/usr/bin/env python3
"""moe.py -- plain top-k Mixture-of-Experts transformer: the sparse-FFN baseline for Zombie-Brain.

Same GQA attention / RoPE / RMSNorm / embeddings as every other model (shared code in baseline.py). Only the FFN
differs: in EVERY layer, a router picks top_k of n_experts small GELU FFNs per token (Mixtral-style: softmax over the
selected logits), plus a Switch-style load-balancing loss. No capacity limit / token dropping (exact, deterministic).

Default sizing (12 experts, top-2, expert hidden 164) is chosen so that, on paper, it matches Zombie v6:
    total params   28.92M vs Zombie 28.88M   (+0.1%)
    active MACs    10.65M vs Zombie 10.68M   (-0.3%, batch-1, gate at 50%)
(see `python counter.py`). So the comparison is: same total capacity, same active compute -- does *learned memory +
gate + self-healing* beat *routed experts*?

Caveats to state in the paper: (1) at batch-1 decode MoE reads only top_k experts, but as batch/prefill grows it
touches most experts (more bytes per token than Zombie's 4 value rows); (2) the router can collapse onto a few experts --
`moe_stats()` logs load entropy and dead experts so it is visible; (3) tiny experts (hidden 164) are not how production
MoEs are shaped, but this is what "matched on paper at 25M scale" forces.

    python moe.py --selftest
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import Cfg, FFN, Transformer, count_params


@dataclass
class MCfg(Cfg):
    n_kv: int = 2
    ffn_mult: float = 0.0          # no dense FFN: every layer is MoE
    n_experts: int = 12
    top_k: int = 2
    exp_hidden: int = 164
    aux_w: float = 0.01            # weight of the load-balancing loss


class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.d, self.E, bias=False)
        self.experts = nn.ModuleList([FFN(cfg.d, cfg.exp_hidden) for _ in range(self.E)])
        self.aux = None
        self.register_buffer("counts", torch.zeros(self.E, dtype=torch.long), persistent=False)

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        logits = self.router(xf).float()
        topv, topi = logits.topk(self.k, dim=-1)                     # (N,k)
        w = torch.softmax(topv, dim=-1)                              # renormalised over the selected experts
        out = xf.new_zeros(xf.shape)
        for e in torch.unique(topi).tolist():                        # only experts that got tokens
            rows, slot = (topi == e).nonzero(as_tuple=True)
            y = self.experts[e](xf[rows]).to(out.dtype)
            out.index_add_(0, rows, y * w[rows, slot].unsqueeze(-1).to(out.dtype))
        with torch.no_grad():
            self.counts += torch.bincount(topi.reshape(-1), minlength=self.E)
        self.aux = None
        if self.training and torch.is_grad_enabled():
            P = torch.softmax(logits, dim=-1).mean(0)                # mean router prob per expert (differentiable)
            f = torch.bincount(topi.reshape(-1), minlength=self.E).float() / topi.numel()
            self.aux = self.E * (f.detach() * P).sum()               # =1.0 at perfectly uniform load
        return out.view(B, T, d)


class MoELM(Transformer):
    def __init__(self, cfg):
        super().__init__(cfg, ffn_factory=lambda l: MoEFFN(cfg))

    def moes(self):
        return [b.ffn for b in self.blocks]

    def aux_loss(self):
        v = [m.aux for m in self.moes() if m.aux is not None]
        if not v:
            return torch.zeros(())
        return self.cfg.aux_w * sum(v) / len(v)

    def moe_reset(self):
        for m in self.moes():
            m.counts.zero_()

    def moe_stats(self, reset=False):
        """per-layer expert load: normalised entropy (1 = uniform), dead experts (never selected), max/mean load."""
        layers = []
        for m in self.moes():
            c = m.counts.float()
            tot = c.sum()
            if tot > 0:
                p = c[c > 0] / tot
                H = float(-(p * p.log()).sum()) / math.log(m.E)
            else:
                H = 0.0
            layers.append(dict(entropy_norm=H, dead=int((c == 0).sum()),
                               max_over_mean=float(c.max() / c.mean()) if tot > 0 else 0.0))
        out = dict(layers=layers,
                   entropy_norm_mean=sum(l["entropy_norm"] for l in layers) / len(layers),
                   dead_experts=sum(l["dead"] for l in layers),
                   max_over_mean=max(l["max_over_mean"] for l in layers))
        if reset:
            self.moe_reset()
        return out


# --------------------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    import counter

    # 1. params == closed form (default config, and a variant)
    for kw in (dict(), dict(n_experts=8, top_k=1, exp_hidden=200)):
        m = MoELM(MCfg(**kw))
        c = MCfg(**kw)
        s = counter.Spec(n_kv=2, ffn_mult=0.0, moe_layers=tuple(range(12)), n_experts=c.n_experts, top_k=c.top_k,
                         exp_hidden=c.exp_hidden)
        a, b = count_params(m), counter.model_costs(s)["total_params"]
        assert a == b, (kw, a, b)
        print(f"[ok] params {kw or 'default'}: {a / 1e6:.3f}M == counter.py")

    # 2. routed output == naive per-token reference
    cfg = MCfg(vocab=50, d=64, n_layers=2, n_heads=4, n_kv=2, ctx=64, n_experts=6, top_k=2, exp_hidden=24)
    m = MoELM(cfg).eval()
    f = m.blocks[0].ffn
    x = torch.randn(3, 7, 64)
    got = f(x)
    ref = torch.zeros_like(x)
    for b in range(3):
        for t in range(7):
            lg = f.router(x[b, t])
            v, i = lg.topk(2)
            w = torch.softmax(v, -1)
            ref[b, t] = sum(w[j] * f.experts[int(i[j])](x[b, t]) for j in range(2))
    err = (got - ref).abs().max().item()
    assert err < 1e-5, err
    print(f"[ok] sparse dispatch == naive per-token reference (max err {err:.1e})")

    # 3. KV-cache decode == full forward; causality
    x = torch.randint(0, 50, (2, 20))
    full = m(x)[0]
    lg, _, pst = m(x[:, :11], use_cache=True)
    outs = [lg]
    for t in range(11, 20):
        lg, _, pst = m(x[:, t:t + 1], past=pst, use_cache=True)
        outs.append(lg)
    err = (full - torch.cat(outs, 1)).abs().max().item()
    assert err < 1e-4, err
    y = x.clone()
    y[0, 12] = (y[0, 12] + 1) % 50
    dd = (m(x)[0] - m(y)[0]).abs().amax(-1)[0]
    assert dd[:12].max() < 1e-5 and dd[12:].max() > 0
    print(f"[ok] KV-cache decode == full forward (max err {err:.1e}); causal")

    # 4. stats + aux loss
    m.train()
    m.moe_reset()
    _, ce, _ = m(torch.randint(0, 50, (8, 32)), torch.randint(0, 50, (8, 32)))
    aux = m.aux_loss()
    st = m.moe_stats()
    assert all(int(mm.counts.sum()) == 8 * 32 * 2 for mm in m.moes())
    assert 0.5 * cfg.aux_w < float(aux) < 3 * cfg.aux_w, float(aux)     # ~aux_w * 1.0 at init
    print(f"[ok] counts sum to tokens x top_k; aux {float(aux) / cfg.aux_w:.2f} x aux_w (1.0 = balanced); "
          f"entropy {st['entropy_norm_mean']:.3f}, dead experts {st['dead_experts']}")

    # 5. learns, and the aux loss keeps load spread
    g = torch.Generator().manual_seed(1)
    T = torch.softmax(torch.randn(50, 50, generator=g) * 3, -1)
    seq = [0]
    for _ in range(40000):
        seq.append(torch.multinomial(T[seq[-1]], 1, generator=g).item())
    data = torch.tensor(seq)
    m = MoELM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for step in range(200):
        ix = torch.randint(0, len(data) - 65, (16,))
        xb = torch.stack([data[i:i + 64] for i in ix])
        yb = torch.stack([data[i + 1:i + 65] for i in ix])
        _, ce, _ = m(xb, yb)
        loss = ce + m.aux_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(ce) if first is None else first
    st = m.moe_stats()
    assert float(ce) < first - 0.5, (first, float(ce))
    print(f"[ok] trains: ce {first:.2f} -> {float(ce):.2f}; load entropy {st['entropy_norm_mean']:.3f}, "
          f"dead experts {st['dead_experts']}, max/mean load {st['max_over_mean']:.2f}")
    print("moe selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.count:
        print(f"moe: {count_params(MoELM(MCfg())):,} params")
