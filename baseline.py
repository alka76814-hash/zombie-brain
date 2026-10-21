#!/usr/bin/env python3
"""baseline.py -- MHA / GQA / GQA+1xFFN / attn-only decoder-only transformers.

Shared building blocks (RMSNorm, RoPE, Attention, FFN, bench) are imported by zombie_v6.py,
so both architectures use *identical* attention code, init, norm and decode path.

    python baseline.py --selftest
    python baseline.py --model gqa1x --bench --threads 6
    python baseline.py --model mha --count        # real sum(p.numel())

Presets (d=384, 12 layers, 8 Q heads, head_dim 48, vocab 8192, untied embed/head):
    mha        8 KV heads, FFN 4x
    gqa        2 KV heads, FFN 4x
    gqa1x      2 KV heads, FFN 1x   <- FLOP-matched control for Zombie
    attn_only  2 KV heads, no FFN
"""
import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 8192
    d: int = 384
    n_layers: int = 12
    n_heads: int = 8
    n_kv: int = 8
    ffn_mult: float = 4.0
    ctx: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def hd(self):
        return self.d // self.n_heads


PRESETS = {
    "mha": dict(n_kv=8, ffn_mult=4.0),
    "gqa": dict(n_kv=2, ffn_mult=4.0),
    "gqa1x": dict(n_kv=2, ffn_mult=1.0),
    "attn_only": dict(n_kv=2, ffn_mult=0.0),
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        n = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * n * self.weight


def rope_cache(hd, max_len, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(max_len).float(), inv)  # (T, hd/2)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x (B,H,T,hd); cos/sin (T,hd/2)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.Hkv, self.hd = cfg.n_heads, cfg.n_kv, cfg.hd
        assert self.H % self.Hkv == 0
        self.wq = nn.Linear(cfg.d, self.H * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d, self.Hkv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d, self.Hkv * self.hd, bias=False)
        self.wo = nn.Linear(self.H * self.hd, cfg.d, bias=False)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.Hkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.Hkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        new = (k, v) if use_cache else None
        P = k.shape[2] - T
        if self.Hkv != self.H:
            r = self.H // self.Hkv
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        if P == 0:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            m = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1)), new


class FFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    def __init__(self, cfg, ffn):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn = ffn
        if ffn is not None:
            self.norm2 = RMSNorm(cfg.d, cfg.norm_eps)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        a, new = self.attn(self.norm1(x), cos, sin, past, use_cache)
        x = x + a
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x, new


def init_weights(model, n_layers):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    std = 0.02 / math.sqrt(2 * n_layers)
    for n, p in model.named_parameters():
        if n.endswith("wo.weight") or n.endswith("w2.weight"):
            nn.init.normal_(p, std=std)


class Transformer(nn.Module):
    """ffn_factory(layer_idx) -> nn.Module | None  lets Zombie reuse everything else."""

    def __init__(self, cfg, ffn_factory=None):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        blocks = []
        for l in range(cfg.n_layers):
            if ffn_factory is not None:
                ffn = ffn_factory(l)
            else:
                ffn = FFN(cfg.d, int(cfg.ffn_mult * cfg.d)) if cfg.ffn_mult > 0 else None
            blocks.append(Block(cfg, ffn))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d, cfg.norm_eps)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        cos, sin = rope_cache(cfg.hd, cfg.ctx, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        init_weights(self, cfg.n_layers)

    def forward(self, idx, targets=None, past=None, use_cache=False):
        """returns (logits, ce_loss | None, new_past | None)"""
        B, T = idx.shape
        P = 0 if past is None else past[0][0].shape[2]
        assert P + T <= self.cfg.ctx, f"sequence {P + T} exceeds ctx {self.cfg.ctx}"
        cos, sin = self.cos[P:P + T], self.sin[P:P + T]
        x = self.tok(idx)
        new = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, cos, sin, None if past is None else past[i], use_cache)
            new.append(kv)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, (new if use_cache else None)


def make_baseline(name, **over):
    kw = dict(PRESETS[name])
    kw.update(over)
    return Transformer(Cfg(**kw))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def bench(model, prompt_len=128, gen=64, batch=1, threads=None, reps=3):
    """prefill tok/s and batch-1 decode tok/s (KV-cache), best of `reps`; plus peak RSS in MB."""
    if threads:
        torch.set_num_threads(threads)
    model.eval()
    V = model.cfg.vocab
    x = torch.randint(0, V, (batch, prompt_len))
    best_pre, best_dec = 0.0, 0.0
    for r in range(reps + 1):  # first rep = warmup
        t0 = time.perf_counter()
        logits, _, past = model(x, use_cache=True)
        t1 = time.perf_counter()
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(gen):
            logits, _, past = model(nxt, past=past, use_cache=True)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
        t2 = time.perf_counter()
        if r > 0:
            best_pre = max(best_pre, batch * prompt_len / (t1 - t0))
            best_dec = max(best_dec, batch * gen / (t2 - t1))
    try:
        import resource
        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        ram = float("nan")
    return dict(prefill_tok_s=best_pre, decode_tok_s=best_dec, peak_rss_mb=ram)


def selftest():
    torch.manual_seed(0)
    # 1. real param counts vs closed-form counter
    from counter import Spec, model_costs
    for name, kw in PRESETS.items():
        m = make_baseline(name)
        s = Spec(n_kv=kw["n_kv"], ffn_mult=kw["ffn_mult"])
        a, b = count_params(m), model_costs(s)["total_params"]
        assert a == b, (name, a, b)
        print(f"[ok] {name:9s} params {a / 1e6:.3f}M == counter.py")
    # 2. KV-cache decode == full forward (MHA and GQA), incl. multi-token continuation
    for name in ("mha", "gqa"):
        m = make_baseline(name, d=64, n_layers=2, n_heads=4, vocab=100, ctx=64,
                          **({"n_kv": 4} if name == "mha" else {"n_kv": 2})).eval()
        x = torch.randint(0, 100, (2, 20))
        full, _, _ = m(x)
        lg, _, pst = m(x[:, :11], use_cache=True)
        outs = [lg]
        lg, _, pst = m(x[:, 11:15], past=pst, use_cache=True)   # T>1 with past (mask path)
        outs.append(lg)
        for t in range(15, 20):                                  # T==1 decode path
            lg, _, pst = m(x[:, t:t + 1], past=pst, use_cache=True)
            outs.append(lg)
        inc = torch.cat(outs, 1)
        err = (full - inc).abs().max().item()
        assert err < 1e-4, err
        print(f"[ok] {name} KV-cache incremental == full forward (max err {err:.1e})")
    # 3. causality: changing a future token must not change earlier logits
    m = make_baseline("gqa", d=64, n_layers=2, n_heads=4, vocab=100, ctx=64, n_kv=2).eval()
    x = torch.randint(0, 100, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 100
    d = (m(x)[0] - m(y)[0]).abs().amax(-1)[0]
    assert d[:10].max() < 1e-6 and d[10:].max() > 0
    print("[ok] causal")
    # 4. learns: markov data, loss must drop
    g = torch.Generator().manual_seed(1)
    T = torch.softmax(torch.randn(50, 50, generator=g) * 3, -1)
    seq = [0]
    for _ in range(40000):
        seq.append(torch.multinomial(T[seq[-1]], 1, generator=g).item())
    data = torch.tensor(seq)
    m = make_baseline("gqa1x", d=64, n_layers=2, n_heads=4, vocab=50, ctx=64, n_kv=2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for step in range(150):
        ix = torch.randint(0, len(data) - 65, (16,))
        xb = torch.stack([data[i:i + 64] for i in ix])
        yb = torch.stack([data[i + 1:i + 65] for i in ix])
        _, loss, _ = m(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first - 0.5, (first, last)
    print(f"[ok] trains: loss {first:.2f} -> {last:.2f}")
    print("baseline selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gqa", choices=list(PRESETS))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.count or a.bench:
        m = make_baseline(a.model)
        print(f"{a.model}: {count_params(m):,} params")
        if a.bench:
            print(bench(m, a.prompt, a.gen, threads=a.threads))
