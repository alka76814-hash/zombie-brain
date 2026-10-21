#!/usr/bin/env python3
"""neuron_keeper.py -- detect and revive DEAD FFN neurons during training (opt-in: train.py --revive-neurons).

Dead neuron = a hidden unit whose output is ~0 on (almost) every input: |GELU(w1_j . x)| < thr on all but a tiny
fraction (`dead_frac`, default 0.1%) of the tokens seen in a window. A single weight cannot be "dead" on its own; the
unit (row of w1 + column of w2) is the thing that dies, so that is what is tracked and revived.

Revive (no gradient, every `every` steps, only if the window saw >= `min_tokens` tokens):
  w1[j]   <- unit-norm copy of a RECENT INPUT vector, scaled to the mean norm of the live rows (so it fires on that kind
             of input and starts learning right away)
  w2[:,j] <- tiny noise (0.01 x std of live columns): it joins the layer without perturbing the loss
  Adam moments of exactly those row/column entries are zeroed.
Applies to ANY FFN module (dense baselines' FFN and Zombie's Branch B) -> identical treatment keeps the comparison fair.
Not covered: attention weights, embeddings, the gate (monitor those from log.jsonl), and the memory (Gravekeeper does that).

    python neuron_keeper.py --selftest
"""
import torch
import torch.nn.functional as F

from baseline import FFN


class NeuronKeeper:
    def __init__(self, model, thr=0.01, dead_frac=0.001, every=500, min_tokens=500_000, buf=512, noise=0.01):
        self.thr, self.dead_frac, self.every, self.min_tokens, self.noise = thr, dead_frac, every, min_tokens, noise
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, FFN)]
        self.st = {}
        self.handles = []
        for name, m in self.layers:
            dev = m.w1.weight.device
            self.st[name] = dict(active=torch.zeros(m.w1.out_features, dtype=torch.long, device=dev), n=0,
                                 buf=torch.zeros(buf, m.w1.in_features, device=dev), bp=0, fill=0)
            self.handles.append(m.w1.register_forward_hook(self._hook(name)))
        self.total_revived = 0

    def _hook(self, name):
        def h(mod, inp, out):
            if not mod.training or not torch.is_grad_enabled():
                return
            s = self.st[name]
            with torch.no_grad():
                a = F.gelu(out.detach().float()).abs().reshape(-1, out.shape[-1])
                s["active"] += (a >= self.thr).sum(0)
                s["n"] += a.shape[0]
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                m = min(16, x.shape[0])
                pick = x[torch.randint(0, x.shape[0], (m,), device=x.device)].float()
                pos = (torch.arange(m, device=x.device) + s["bp"]) % s["buf"].shape[0]
                s["buf"][pos] = pick
                s["bp"] = (s["bp"] + m) % s["buf"].shape[0]
                s["fill"] = min(s["buf"].shape[0], s["fill"] + m)
        return h

    def remove(self):
        for h in self.handles:
            h.remove()

    @torch.no_grad()
    def step(self, step, opts=None):
        """Call after the optimizer step. Returns dict(dead=..., revived=..., layers=...) or None if nothing was checked."""
        if step <= 0 or step % self.every != 0:
            return None
        opt = None if opts is None else opts.get("dense")
        out = dict(dead=0, revived=0, checked=0, per_layer={})
        for name, m in self.layers:
            s = self.st[name]
            if s["n"] < self.min_tokens:
                continue
            out["checked"] += 1
            frac = s["active"].float() / s["n"]
            dead = frac < self.dead_frac
            k = int(dead.sum())
            out["dead"] += k
            out["per_layer"][name] = k
            if k and s["fill"] > 0:
                w1, w2 = m.w1.weight, m.w2.weight
                live = ~dead
                mn = w1[live].norm(dim=1).mean() if live.any() else w1.norm(dim=1).mean()
                sd = w2[:, live].std() if live.sum() > 1 else w2.std()
                idx = dead.nonzero().squeeze(-1)
                picks = s["buf"][torch.randint(0, s["fill"], (k,), device=w1.device)]
                w1[idx] = (F.normalize(picks, dim=1) * mn).to(w1.dtype)
                w2[:, idx] = (self.noise * sd * torch.randn(w2.shape[0], k, device=w2.device)).to(w2.dtype)
                if opt is not None:
                    for p, sl in ((w1, lambda t: t.__setitem__(idx, 0)),
                                  (w2, lambda t: t.__setitem__((slice(None), idx), 0))):
                        stt = opt.state.get(p)
                        if stt:
                            for key in ("exp_avg", "exp_avg_sq"):
                                if key in stt:
                                    sl(stt[key])
                out["revived"] += k
            s["active"].zero_()
            s["n"] = 0
        self.total_revived += out["revived"]
        return out if out["checked"] else None


def selftest():
    import torch.nn as nn
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = FFN(32, 64)

        def forward(self, x):
            return self.ffn(x)
    net = Net()
    with torch.no_grad():
        dead_idx = torch.tensor([3, 17, 40])
        net.ffn.w1.weight[dead_idx] = 0.0          # output exactly 0 on every input ...
        net.ffn.w2.weight[:, dead_idx] = 0.0       # ... and no gradient can reach them either
    opt =torch.optim.AdamW(net.parameters(), lr=1e-3)
    opts = dict(dense=opt)
    keeper = NeuronKeeper(net, every=10, min_tokens=1000, dead_frac=0.001)
    revived_at = None
    for step in range(1, 21):
        x = torch.randn(16, 32, 32) * 2
        loss = net(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        r = keeper.step(step, opts)
        if r and r["revived"] and revived_at is None:
            revived_at = step
            assert r["dead"] >= 3, r
            for j in dead_idx.tolist():
                assert net.ffn.w1.weight[j].abs().sum() > 0, "w1 row not revived"
                assert net.ffn.w2.weight[:, j].abs().max() < 0.05 * net.ffn.w2.weight.std() + 1e-3, "w2 col not small"
                for key in ("exp_avg", "exp_avg_sq"):
                    assert opt.state[net.ffn.w1.weight][key][j].abs().sum() == 0
                    assert opt.state[net.ffn.w2.weight][key][:, j].abs().sum() == 0
    assert revived_at == 10, revived_at
    print(f"[ok] dead units {dead_idx.tolist()} detected and revived at step {revived_at} (w1 re-aimed, w2 ~0, Adam moments zeroed)")
    # revived units must now fire and learn; healthy units must be untouched at revival time
    x = torch.randn(64, 32, 32) * 2
    a = F.gelu(net.ffn.w1(x)).abs().reshape(-1, 64)
    frac = (a >= 0.01).float().mean(0)
    print(f"[ok] revived units now active on {frac[dead_idx].mean() * 100:.0f}% of tokens (healthy mean {frac.mean() * 100:.0f}%)")
    assert frac[dead_idx].min() > 0.05
    # eval-mode forwards must not be tracked
    keeper2 = NeuronKeeper(net, every=10, min_tokens=1)
    net.eval()
    net(torch.randn(4, 8, 32))
    assert all(s["n"] == 0 for s in keeper2.st.values())
    net.train()
    with torch.no_grad():
        net(torch.randn(4, 8, 32))
    assert all(s["n"] == 0 for s in keeper2.st.values())
    print("[ok] eval-mode / no-grad forwards are not counted")
    print("neuron_keeper selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
