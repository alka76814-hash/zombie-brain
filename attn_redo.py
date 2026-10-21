"""attn_redo.py -- dormant-attention-head measurement and ReDo-style recycling (Sokar et al. 2023, adapted to attention).

An attention head h of layer l is DORMANT when its activity is tiny relative to the other heads of the same layer:
    score(l, h) = mean|a_h| / mean over heads of mean|a_h'|        a_h = the head's output (the input of wo, its hd channels)
ReDo: heads with score < tau are recycled -- incoming weights (the head's rows of wq) are re-initialised, outgoing weights
(the head's columns of wo) are set to zero (so the reset never changes the function), and the optimizer moments for those
slices are cleared.  k/v projections are shared between several query heads (GQA), so they are left alone.

Everything is deterministic given (seed, step), so a resumed run does exactly what an uninterrupted run would have done.
"""
import torch


def _attn_modules(model):
    return [(n, m) for n, m in model.named_modules() if n.endswith(".attn") and hasattr(m, "wo")]


@torch.no_grad()
def head_scores(model, x, amp_ctx):
    """-> list over layers of a (H,) tensor of dormancy scores (1.0 = average head, ~0 = dormant), and per-channel scores"""
    mods = _attn_modules(model)
    cap = {}
    hooks = [m.wo.register_forward_hook(lambda mod, inp, out, n=n: cap.__setitem__(n, inp[0].detach())) for n, m in mods]
    was_training = model.training
    model.eval()
    try:
        with amp_ctx():
            model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    heads, chans = [], []
    for n, m in mods:
        a = cap[n].float().abs()                                  # (B, T, H*hd)
        ch = a.mean(dim=(0, 1))                                   # (H*hd,)
        hd = ch.view(m.H, m.hd).mean(1)                           # (H,)
        heads.append(hd / hd.mean().clamp_min(1e-12))
        chans.append(ch / ch.mean().clamp_min(1e-12))
    return heads, chans


def summarize(heads, chans, tau):
    nh = sum(h.numel() for h in heads)
    dh = sum(int((h < tau).sum()) for h in heads)
    nc = sum(c.numel() for c in chans)
    dc = sum(int((c < tau).sum()) for c in chans)
    return dict(tau=tau, heads_dormant=dh, heads_total=nh, heads_frac=dh / nh, chan_dormant=dc, chan_total=nc,
                chan_frac=dc / nc, min_head_score=float(min(h.min() for h in heads)),
                per_layer=[int((h < tau).sum()) for h in heads])


@torch.no_grad()
def redo_reset(model, opts, heads, tau, seed, step, init_std=0.02):
    """recycle dormant heads; returns the number of heads reset"""
    gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
    dense = opts.get("dense")
    n_reset = 0
    for (name, m), sc in zip(_attn_modules(model), heads):
        for h in torch.nonzero(sc < tau).flatten().tolist():
            lo, hi = h * m.hd, (h + 1) * m.hd
            new = torch.randn(m.hd, m.wq.weight.shape[1], generator=gen) * init_std
            m.wq.weight[lo:hi] = new.to(m.wq.weight.device, m.wq.weight.dtype)
            m.wo.weight[:, lo:hi] = 0.0
            if dense is not None:
                for p, sl in ((m.wq.weight, (slice(lo, hi),)), (m.wo.weight, (slice(None), slice(lo, hi)))):
                    st = dense.state.get(p, {})
                    for k in ("exp_avg", "exp_avg_sq"):
                        if k in st:
                            st[k][sl] = 0.0
            n_reset += 1
    return n_reset
