#!/usr/bin/env python3
"""zombie_v6.py -- GQA transformer whose FFN is replaced by (shared product-key memory | Branch B), routed by a gate.

Needs baseline.py and (for --selftest) counter.py in the same folder.

Per Zombie layer:   x -> [gate g]  out = g*B(x) + (1-g)*A(x)
    B(x) = 2-matrix GELU FFN, hidden = d            (compute path)
    A(x) = scale * PKM(BatchNorm(Wq x))             (memory path; values/sub-keys SHARED by a group of layers)
Phases (ZombieLM.set_step):
    shadow  step <  warm_steps              g = 1 (B only). Memory runs in shadow, trained by MSE(A, B.detach()).
    soft    warm .. warm+soft_steps         g = sigmoid(gate)   (soft mix)
    hard    afterwards                      train: straight-through hard gate; inference: true per-token routing
At step == warm_steps: optional k-means init of sub-keys + values (kmeans_init), trackers/optimizer rows reset.

Deliberate choices (see comments): 2 PKM heads share ONE value table per cache; sub-key rows are the unit
of death; values are only reset for slots that are dead for *every* head; the "score loss" from the v6 notes
is NOT implemented (undefined; the MSE already trains the scores through the softmax weights).

Minimal training loop:
    model = ZombieLM(ZCfg()); opts = build_optimizers(model)
    for step in range(6100):
        apply_lr_scale(opts, lr_at(step, 6100))
        info = train_step(model, opts, x, y, step, init_batches=lambda: token_batches())
"""
import argparse
import math
import time
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline import Cfg, FFN, Transformer, bench, count_params


@dataclass
class ZCfg(Cfg):
    n_kv: int = 2
    ffn_mult: float = 4.0                 # dense FFN of layers NOT in mem_layers (partial-replacement ablation)
    mem_layers: Optional[tuple] = None    # default: all layers
    n_groups: int = 2                     # layers 0-5 share cache A, 6-11 cache B (layer l -> l*n_groups//L)
    n_sub: int = 128                      # n_sub^2 = 16384 slots / cache
    pk_heads: int = 2
    half: int = 96                        # each query half; query dim per head = 2*half
    topk: int = 4
    gate_hidden: int = 32
    gate_mode: str = "learned"            # learned | fixed_half (0.5B+0.5A) | mem_only (no B, no gate)
    qnorm: bool = True                    # BatchNorm on queries (original PKM); False = stress config
    warm_steps: int = 1000
    soft_steps: int = 2000
    distill_w: float = 1.0                # weight of MSE(A, B.detach()) in the shadow phase
    gate_target: float = 0.5              # target mean gate (sweep {0.3, 0.5, 0.7})
    lam_mean: float = 0.01
    lam_bimodal: float = 0.01
    init: str = "kmeans"                  # kmeans | none
    gravekeeper: bool = True
    dead_after: int = 500                 # steps without use => dead sub-key row
    gk_every: int = 100                   # how often Gravekeeper runs
    buf_size: int = 1024                  # ring buffers (poorly-served queries, B outputs)
    revive_noise: float = 0.01
    route: str = "subset"                 # subset (real FLOP saving) | where (compute both, mask)
    dense_vgrad: bool = True              # memory-efficient value-table gradient (one dense buffer per cache, sparsified after backward)
    moe_experts: int = 0                  # >0: every memory layer ALSO gets a routed top-k expert FFN (L24xc)
    moe_top_k: int = 2
    moe_hidden: int = 24
    moe_aux_w: float = 0.01

    def __post_init__(self):
        if self.mem_layers is None:
            self.mem_layers = tuple(range(self.n_layers))
        self.mem_layers = tuple(self.mem_layers)


class ZState:
    """Mutable run state shared by all layers (plain object, deliberately not an nn.Module)."""
    def __init__(self, route):
        self.phase = "shadow"
        self.collect = None      # None | 'q' | 'v'   (k-means init passes)
        self.route = route
        self.force_p = None      # benchmarking only: route fraction p to B at inference


# --------------------------------------------------------------------------------------
class _PKGather(torch.autograd.Function):
    """out[n] = sum_j w[n,j] * V[idx[n,j]].  Same math as F.embedding_bag(mode='sum', per_sample_weights), but the gradient
    of V is accumulated into ONE dense (n_slots, d) buffer instead of a separate sparse tensor per layer.  With sparse=True
    every layer keeps N*heads*k*d floats of gradient alive until the optimizer step (24 layers x ~100 MB at 8192 tokens): the
    'VRAM leak' of deep memory models.  Here peak is one cache-sized buffer, independent of depth."""

    @staticmethod
    def forward(ctx, weight, idx, w):
        ctx.save_for_backward(weight, idx, w)
        return F.embedding_bag(idx, weight, per_sample_weights=w, mode="sum")

    @staticmethod
    def backward(ctx, g):
        weight, idx, w = ctx.saved_tensors
        N = idx.shape[0]
        gv = torch.zeros_like(weight)
        gw = torch.empty_like(w)
        for c in range(0, N, 2048):                                   # chunked: bounded temporaries
            ii, gc = idx[c:c + 2048], g[c:c + 2048].to(weight.dtype)
            gw[c:c + 2048] = (weight[ii] * gc[:, None, :]).sum(-1).to(w.dtype)
            gv.index_add_(0, ii.reshape(-1), (w[c:c + 2048, :, None].to(weight.dtype) * gc[:, None, :]).reshape(-1, weight.shape[1]))
        return gv, None, gw


class PKCache(nn.Module):
    """Product-key memory shared by several layers.
    keys   (heads, 2, n_sub, half)   dense param, trained by AdamW (every row gets gradient through the dense score matrix)
    values EmbeddingBag(n_slots, d)  sparse grads -> SparseAdam; ONE table shared by the heads
    slot id = i*n_sub + j, score(i,j) = k1[i].q1 + k2[j].q2 (per head)"""

    def __init__(self, cfg):
        super().__init__()
        self.n_sub, self.heads, self.half, self.topk, self.d = cfg.n_sub, cfg.pk_heads, cfg.half, cfg.topk, cfg.d
        self.n_slots = cfg.n_sub ** 2
        self.dead_after, self.buf, self.noise = cfg.dead_after, cfg.buf_size, cfg.revive_noise
        self.keys = nn.Parameter(torch.randn(self.heads, 2, self.n_sub, self.half) / math.sqrt(self.half))
        self.dense_vgrad = cfg.dense_vgrad
        self.values = nn.EmbeddingBag(self.n_slots, self.d, mode="sum", sparse=True)
        nn.init.normal_(self.values.weight, std=0.02)
        if self.dense_vgrad:
            self.register_buffer("_touched", torch.zeros(self.n_slots, dtype=torch.bool), persistent=False)
            self.values.weight.register_post_accumulate_grad_hook(self._sparsify_grad)
        self.register_buffer("last_used", torch.zeros(self.heads, 2, self.n_sub, dtype=torch.long))
        self.register_buffer("slot_last", torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer("slot_hits", torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer("q_buf", torch.zeros(self.heads, 2, self.buf, self.half))
        self.register_buffer("b_buf", torch.zeros(self.buf, self.d))
        self.cur_step, self.qp, self.q_fill, self.bp, self.b_fill = 0, 0, 0, 0, 0
        self._qs, self._vsum, self._vcnt = [], None, None

    # ---- lookup -------------------------------------------------------------------
    def lookup(self, q):
        """q (N, heads, 2, half) -> out (N,d), slot ids (N,heads,k), best combined score (N,heads,k)"""
        N = q.shape[0]
        sc = torch.einsum("nhcd,hcsd->nhcs", q, self.keys.to(q.dtype))        # (N,h,2,n_sub)
        v, i = sc.topk(self.topk, dim=-1)                                      # (N,h,2,k)
        comb = (v[:, :, 0, :, None] + v[:, :, 1, None, :]).flatten(2)          # (N,h,k*k)
        cidx = (i[:, :, 0, :, None] * self.n_sub + i[:, :, 1, None, :]).flatten(2)
        top, pos = comb.topk(self.topk, dim=-1)                                # (N,h,k)
        idx = cidx.gather(-1, pos)
        w = torch.softmax(top.float(), -1).to(self.values.weight.dtype)
        flat_idx, flat_w = idx.reshape(N, -1), w.reshape(N, -1)
        if self.dense_vgrad and torch.is_grad_enabled() and self.values.weight.requires_grad:
            with torch.no_grad():
                self._touched[flat_idx.reshape(-1)] = True
            out = _PKGather.apply(self.values.weight, flat_idx, flat_w)
        else:
            out = self.values(flat_idx, per_sample_weights=flat_w)
        return out, idx, top

    @torch.no_grad()
    def _sparsify_grad(self, p):
        """runs once per backward, after all layers' contributions are summed: dense grad -> the sparse grad SparseAdam
        expects (rows = slots looked up this step; identical update to the old per-layer sparse path)."""
        g = p.grad
        if g is None or g.is_sparse:
            return
        rows = self._touched.nonzero().squeeze(1)
        p.grad = torch.sparse_coo_tensor(rows[None], g[rows], g.shape, is_coalesced=True, check_invariants=False)
        self._touched.zero_()

    # ---- usage tracking (training only; no grad) -----------------------------------
    @torch.no_grad()
    def track(self, idx, top, q, mask=None):
        if mask is not None:
            idx, top, q = idx[mask], top[mask], q[mask]
        N = idx.shape[0]
        if N == 0:
            return
        i, j = idx // self.n_sub, idx % self.n_sub
        for h in range(self.heads):
            self.last_used[h, 0, i[:, h].reshape(-1)] = self.cur_step
            self.last_used[h, 1, j[:, h].reshape(-1)] = self.cur_step
        flat = idx.reshape(-1)
        self.slot_hits += torch.bincount(flat, minlength=self.n_slots)
        self.slot_last[flat] = self.cur_step
        # queries of poorly-served tokens: lowest best-score tokens of this call (per head)
        m = min(32, N)
        low = top[..., 0].topk(m, dim=0, largest=False).indices                # (m,h)
        ql = torch.stack([q[low[:, h], h] for h in range(self.heads)], 1)      # (m,h,2,half)
        pos = (torch.arange(m, device=q.device) + self.qp) % self.buf
        self.q_buf[:, :, pos] = ql.permute(1, 2, 0, 3).to(self.q_buf.dtype)
        self.qp = (self.qp + m) % self.buf
        self.q_fill = min(self.buf, self.q_fill + m)

    @torch.no_grad()
    def push_b(self, b_out, mask=None):
        """Remember Branch-B OUTPUTS (output space) of tokens that chose the compute path."""
        sel = b_out if mask is None else b_out[mask]
        if sel.shape[0] == 0:
            return
        if sel.shape[0] > 16:
            sel = sel[torch.randperm(sel.shape[0], device=sel.device)[:16]]
        m = sel.shape[0]
        pos = (torch.arange(m, device=sel.device) + self.bp) % self.buf
        self.b_buf[pos] = sel.to(self.b_buf.dtype)
        self.bp = (self.bp + m) % self.buf
        self.b_fill = min(self.buf, self.b_fill + m)

    @torch.no_grad()
    def usage_stats(self, step, reset=True):
        hits = self.slot_hits.float()
        tot = hits.sum()
        H = 0.0
        if tot > 0:
            p = hits[hits > 0] / tot
            H = float(-(p * p.log()).sum())
        st = dict(slot_entropy=H, slot_entropy_norm=H / math.log(self.n_slots),
                  slots_used_frac=float((hits > 0).float().mean()),
                  dead_rows=int(((step - self.last_used) >= self.dead_after).sum()),
                  dead_slots=int(((step - self.slot_last) >= self.dead_after).sum()))
        if reset:
            self.slot_hits.zero_()
        return st

    @torch.no_grad()
    def reset_trackers(self, step):
        self.last_used.fill_(step)
        self.slot_last.fill_(step)
        self.slot_hits.zero_()

    # ---- Gravekeeper ---------------------------------------------------------------
    @torch.no_grad()
    def revive(self, step, opts=None):
        """Sub-key row = unit of death. A dead row (never in a selected top-k for `dead_after` steps) gets
          * key  <- a recent query half-vector of a poorly-served token, rescaled to the mean norm of live rows
          * values of its n_sub slots <- B outputs / heads (+noise), but ONLY slots that are dead for every head
            (the value table is shared across heads, so a live slot must not be overwritten).
        Optimizer moments of touched rows are zeroed. CPU/GPU, no gradient."""
        dead = (step - self.last_used) >= self.dead_after
        log = dict(dead_rows=int(dead.sum()), revived=0, slots_reset=0)
        if log["dead_rows"] == 0 or self.q_fill == 0:
            return log
        dense_opt = None if opts is None else opts.get("dense")
        sparse_opt = None if opts is None else opts.get("sparse")
        ar = torch.arange(self.n_sub, device=self.keys.device)
        norms = self.keys.norm(dim=-1)                                          # (h,2,n_sub)
        for h, c, r in dead.nonzero().tolist():
            live = ~dead[h, c]
            mn = norms[h, c][live].mean() if live.any() else norms[h, c].mean()
            q = self.q_buf[h, c, torch.randint(0, self.q_fill, (1,)).item()]
            self.keys[h, c, r] = F.normalize(q, dim=-1) * mn
            self.last_used[h, c, r] = step
            _reset_rows(dense_opt, self.keys, (h, c, r))
            log["revived"] += 1
            if self.b_fill > 0:
                slots = r * self.n_sub + ar if c == 0 else ar * self.n_sub + r
                slots = slots[(step - self.slot_last[slots]) >= self.dead_after]
                if slots.numel():
                    picks = self.b_buf[torch.randint(0, self.b_fill, (slots.numel(),), device=slots.device)]
                    self.values.weight[slots] = picks / self.heads + self.noise * torch.randn_like(picks)
                    self.slot_last[slots] = step
                    _reset_rows(sparse_opt, self.values.weight, slots)
                    log["slots_reset"] += slots.numel()
        return log

    # ---- k-means init collection (used by ZombieLM.kmeans_init) ----------------------
    @torch.no_grad()
    def collect(self, mode, q, bo):
        if mode == "q":
            sel = torch.randperm(q.shape[0], device=q.device)[:4096]
            self._qs.append(q[sel].detach())
        else:  # 'v': accumulate B/heads into the top-1 cell of every head
            sc = torch.einsum("nhcd,hcsd->nhcs", q, self.keys.to(q.dtype)).argmax(-1)  # (N,h,2)
            cell = (sc[:, :, 0] * self.n_sub + sc[:, :, 1]).reshape(-1)
            self._vsum.index_add_(0, cell, (bo / self.heads).repeat_interleave(self.heads, 0).float())
            self._vcnt.index_add_(0, cell, torch.ones_like(cell, dtype=torch.float))

    @torch.no_grad()
    def finish_init(self):
        cnt, n = self._vcnt, self.n_sub
        mean = self._vsum / cnt.clamp(min=1)[:, None]
        empty = cnt == 0
        if empty.any() and (~empty).any():
            # nearest non-empty cell in key space: ||k1[i]-k1[i']||^2 + ||k2[j]-k2[j']||^2 (summed over heads)
            D = [sum(torch.cdist(self.keys[h, c], self.keys[h, c]) ** 2 for h in range(self.heads)) for c in (0, 1)]
            ne = (~empty).nonzero().squeeze(-1)
            ni, nj = ne // n, ne % n
            std = mean[~empty].std()
            for chunk in empty.nonzero().squeeze(-1).split(1024):
                dist = D[0][chunk // n][:, ni] + D[1][chunk % n][:, nj]
                mean[chunk] = mean[ne[dist.argmin(1)]] + 0.01 * std * torch.randn(len(chunk), self.d, device=mean.device)
        stats = dict(empty_cells=int(empty.sum()), n_cells=self.n_slots)
        self.values.weight.data.copy_(mean)
        self._vsum = self._vcnt = None
        self._qs = []
        return stats


def _reset_rows(opt, p, rows):
    """Zero Adam moments of the given rows (stale momentum would keep moving a freshly revived row)."""
    st = None if opt is None else opt.state.get(p)
    if not st:
        return
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st:
            st[k][rows] = 0


def kmeans(x, k, iters=15):
    """Plain Lloyd on (M,D). Empty clusters re-seeded from random points."""
    M = x.shape[0]
    if M < k:
        x = torch.cat([x, x[torch.randint(0, M, (k - M,), device=x.device)] + 1e-3 * torch.randn(k - M, x.shape[1], device=x.device)])
        M = k
    c = x[torch.randperm(M, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = (-2 * x @ c.T + (c * c).sum(1)[None]).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k).float()
        dead = n == 0
        c = torch.where(dead[:, None], x[torch.randint(0, M, (k,), device=x.device)], s / n.clamp(min=1)[:, None])
    return c


# --------------------------------------------------------------------------------------
class ZombieFFN(nn.Module):
    def __init__(self, cfg, cache, zs):
        super().__init__()
        d = cfg.d
        self.cfg = cfg
        self._c, self._z = (cache,), (zs,)          # tuples => NOT registered as children (cache is shared)
        self.qdim = cfg.pk_heads * 2 * cfg.half
        self.qproj = nn.Linear(d, self.qdim, bias=False)
        self.qnorm = nn.BatchNorm1d(self.qdim) if cfg.qnorm else nn.Identity()
        self.scale = nn.Parameter(torch.ones(d))     # per-layer rescale of shared values
        self.moe = None
        if cfg.moe_experts > 0:
            from moe import MoEFFN
            self.moe = MoEFFN(types.SimpleNamespace(d=d, n_experts=cfg.moe_experts, top_k=cfg.moe_top_k, exp_hidden=cfg.moe_hidden))
        self.has_b = cfg.gate_mode != "mem_only"
        if self.has_b:
            self.B = FFN(d, d)
        if cfg.gate_mode == "learned":
            self.gate = nn.Sequential(nn.Linear(d, cfg.gate_hidden), nn.GELU(), nn.Linear(cfg.gate_hidden, 1))
        self.aux_distill = self.aux_gate = self.gate_mean = None

    def queries(self, xf):
        c = self._c[0]
        return self.qnorm(self.qproj(xf)).view(-1, c.heads, 2, c.half)

    def memory(self, xf, track=False, mask_fn=None):
        c = self._c[0]
        q = self.queries(xf)
        A, idx, top = c.lookup(q)
        if track:
            c.track(idx, top, q, None if mask_fn is None else mask_fn())
        return A * self.scale

    def _route(self, xf, logit):
        zs = self._z[0]
        useB = logit > 0
        if zs.force_p is not None:
            useB = torch.rand(xf.shape[0], device=xf.device) < zs.force_p
        if zs.route == "where":                               # both computed, then masked: no FLOP saving
            return torch.where(useB[:, None], self.B(xf), self.memory(xf))
        iB = useB.nonzero(as_tuple=True)[0]
        iA = (~useB).nonzero(as_tuple=True)[0]
        oB = self.B(xf[iB]) if iB.numel() else None
        oA = self.memory(xf[iA]) if iA.numel() else None
        # under autocast the two branches can return different dtypes (fp16 vs fp32): use one common dtype
        dts = [o.dtype for o in (oB, oA) if o is not None]
        out = xf.new_empty(xf.shape, dtype=torch.promote_types(dts[0], dts[-1]) if dts else xf.dtype)
        if oB is not None:
            out[iB] = oB.to(out.dtype)
        if oA is not None:
            out[iA] = oA.to(out.dtype)
        return out

    def forward(self, x):
        zs, cfg, c = self._z[0], self.cfg, self._c[0]
        Bsz, T, d = x.shape
        xf = x.reshape(-1, d)
        self.aux_distill = self.aux_gate = self.gate_mean = None
        mode, tr = cfg.gate_mode, self.training
        if mode == "mem_only":
            out = self.memory(xf, track=tr)
            if self.moe is not None:                          # memory (stored knowledge) + routed experts (computation)
                out = out + self.moe(x).reshape(-1, d).to(out.dtype)
            return out.view(Bsz, T, d)
        if zs.collect is not None:                            # k-means init pass: gate=1, just observe
            Bo = self.B(xf)
            c.collect(zs.collect, self.queries(xf), Bo)
            return Bo.view(Bsz, T, d)
        if zs.phase == "shadow":
            Bo = self.B(xf)
            if tr:
                A = self.memory(xf, track=True)
                self.aux_distill = F.mse_loss(A, Bo.detach())
                c.push_b(Bo.detach())
            return Bo.view(Bsz, T, d)
        # soft / hard
        if mode == "learned":
            logit = self.gate(xf).squeeze(-1)
            gs = torch.sigmoid(logit)
            if zs.phase == "hard" and not tr:
                return self._route(xf, logit).view(Bsz, T, d)
        else:
            gs = xf.new_full((xf.shape[0],), 0.5)
        Bo = self.B(xf)
        mask_fn = None
        if mode == "learned" and zs.phase == "hard":
            g = gs + ((logit > 0).to(gs.dtype) - gs).detach()          # straight-through hard gate
            mask_fn = lambda: logit <= 0                                 # only tokens that actually USE memory count as usage
        else:
            g = gs
        A = self.memory(xf, track=tr, mask_fn=mask_fn)
        out = g[:, None] * Bo + (1 - g[:, None]) * A
        if tr:
            c.push_b(Bo.detach(), gs.detach() > 0.7)
            if mode == "learned":
                self.gate_mean = gs.mean()
                self.aux_gate = (cfg.lam_mean * (gs.mean() - cfg.gate_target).abs()
                                 - cfg.lam_bimodal * (gs - 0.5).abs().mean())
        return out.view(Bsz, T, d)


class ZombieLM(Transformer):
    def __init__(self, cfg):
        self.zs = ZState(cfg.route)                 # plain object (not a Module); shared by all ZombieFFN layers
        zs = self.zs
        groups = {}
        for l in cfg.mem_layers:
            g = l * cfg.n_groups // cfg.n_layers
            if g not in groups:
                groups[g] = PKCache(cfg)
        super().__init__(cfg, ffn_factory=lambda l: (ZombieFFN(cfg, groups[l * cfg.n_groups // cfg.n_layers], zs)
                                                     if l in cfg.mem_layers else FFN(cfg.d, int(cfg.ffn_mult * cfg.d))))
        self.caches = nn.ModuleList([groups[g] for g in sorted(groups)])
        self.step = 0
        self.mem_started = False

    # ---- state ---------------------------------------------------------------------
    def mem_blocks(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, ZombieFFN)]

    def set_step(self, step):
        cfg = self.cfg
        self.step = step
        if cfg.gate_mode == "mem_only" or not self.mem_blocks():
            ph = "hard"
        else:
            ph = "shadow" if step < cfg.warm_steps else ("soft" if step < cfg.warm_steps + cfg.soft_steps else "hard")
        self.zs.phase = ph
        for c in self.caches:
            c.cur_step = step

    def forward(self, idx, targets=None, past=None, use_cache=False):
        return super().forward(idx, targets, past, use_cache)

    # ---- losses / stats ----------------------------------------------------------------
    def aux_loss(self):
        mb = self.mem_blocks()
        tot = 0.0
        for f in mb:
            if f.aux_distill is not None:
                tot = tot + self.cfg.distill_w * f.aux_distill
            if f.aux_gate is not None:
                tot = tot + f.aux_gate
        out = tot / max(len(mb), 1) if isinstance(tot, torch.Tensor) else torch.zeros(())
        mo = [f.moe.aux for f in mb if f.moe is not None and f.moe.aux is not None]
        if mo:
            out = out + self.cfg.moe_aux_w * sum(mo) / len(mo)
        return out

    def gate_means(self):
        return [None if f.gate_mean is None else float(f.gate_mean.detach()) for f in self.mem_blocks()]

    def cache_stats(self, reset=True):
        return [c.usage_stats(self.step, reset) for c in self.caches]

    # ---- k-means init at the shadow -> memory switch ---------------------------------------
    @torch.no_grad()
    def kmeans_init(self, batches_fn, fit_batches=8, assign_tokens=1_000_000, iters=15):
        """batches_fn() -> iterator of LongTensor (B,T). Forward-only. Query halves are clustered per (head, half);
        keys <- unit centroids * current mean key norm; value[i,j] <- mean B-OUTPUT / heads of tokens whose
        top-1 cell is (i,j); empty cells copy the nearest non-empty cell (+noise)."""
        was_train, ph = self.training, self.zs.phase
        self.eval()
        self.zs.phase = "shadow"
        for c in self.caches:
            c._qs = []
        self.zs.collect = "q"
        for k, x in enumerate(batches_fn()):
            if k >= fit_batches:
                break
            self(x)
        self.zs.collect = None
        for c in self.caches:
            Q = torch.cat(c._qs)                                               # (M,h,2,half)
            for h in range(c.heads):
                for cc in range(2):
                    norm = c.keys[h, cc].norm(dim=-1).mean()
                    c.keys[h, cc] = F.normalize(kmeans(Q[:, h, cc].float(), c.n_sub, iters), dim=-1) * norm
            c._vsum = torch.zeros(c.n_slots, c.d, device=Q.device)
            c._vcnt = torch.zeros(c.n_slots, device=Q.device)
        self.zs.collect = "v"
        seen = 0
        for x in batches_fn():
            self(x)
            seen += x.numel()
            if seen >= assign_tokens:
                break
        self.zs.collect = None
        stats = [c.finish_init() for c in self.caches]
        self.zs.phase = ph
        self.train(was_train)
        return stats

    def begin_memory_phase(self, step, opts=None, init_batches=None, **kw):
        """Called once at step == warm_steps."""
        info = {}
        if self.cfg.init == "kmeans" and init_batches is not None and self.mem_blocks() and self.cfg.gate_mode != "mem_only":
            info["kmeans"] = self.kmeans_init(init_batches, **kw)
        for c in self.caches:
            c.reset_trackers(step)
            if opts:
                _reset_rows(opts.get("sparse"), c.values.weight, slice(None))
                _reset_rows(opts.get("dense"), c.keys, slice(None))
        self.mem_started = True
        return info

    # ---- checkpoint/resume: python-side state that is not in state_dict() ---------------------
    def extra_state(self):
        return dict(step=self.step, mem_started=self.mem_started,
                    caches=[dict(qp=c.qp, q_fill=c.q_fill, bp=c.bp, b_fill=c.b_fill) for c in self.caches])

    def load_extra_state(self, d):
        self.step, self.mem_started = d["step"], d["mem_started"]
        for c, s in zip(self.caches, d["caches"]):
            c.qp, c.q_fill, c.bp, c.b_fill = s["qp"], s["q_fill"], s["bp"], s["b_fill"]
        self.set_step(self.step)

    def maintenance(self, step, opts=None):
        cfg = self.cfg
        if cfg.gravekeeper and self.zs.phase != "shadow" and step > 0 and step % cfg.gk_every == 0:
            return [c.revive(step, opts) for c in self.caches]
        return None


# --------------------------------------------------------------------------------------
def build_optimizers(model, lr=3e-4, cache_lr=1e-3, key_lr=None, wd=0.1, betas=(0.9, 0.95)):
    """dense AdamW (everything except cache values; sub-keys with wd=0) + SparseAdam for cache values."""
    vals = [c.values.weight for c in model.caches]
    keys = [c.keys for c in model.caches]
    special = {id(p) for p in vals + keys}
    decay = [p for n, p in model.named_parameters() if id(p) not in special and p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if id(p) not in special and p.ndim < 2]
    groups = [dict(params=decay, weight_decay=wd), dict(params=nodecay, weight_decay=0.0)]
    if keys:
        groups.append(dict(params=keys, weight_decay=0.0, lr=key_lr or lr))
    for g in groups:
        g["base_lr"] = g.get("lr", lr)
    dense = torch.optim.AdamW(groups, lr=lr, betas=betas)
    opts = dict(dense=dense)
    if vals:
        sp = torch.optim.SparseAdam(vals, lr=cache_lr, betas=betas)
        sp.param_groups[0]["base_lr"] = cache_lr
        opts["sparse"] = sp
    opts["clip_params"] = decay + nodecay + keys
    return opts


def lr_at(step, total, warmup=200, min_frac=0.1):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))


def apply_lr_scale(opts, scale):
    for k in ("dense", "sparse"):
        if k in opts:
            for g in opts[k].param_groups:
                g["lr"] = g["base_lr"] * scale


def train_step(model, opts, x, y, step, grad_clip=1.0, init_batches=None, init_kw=None):
    model.set_step(step)
    info = {}
    if step == model.cfg.warm_steps and not model.mem_started:
        info["init"] = model.begin_memory_phase(step, opts, init_batches, **(init_kw or {}))
    model.train()
    for k in ("dense", "sparse"):
        if k in opts:
            opts[k].zero_grad(set_to_none=True)
    _, ce, _ = model(x, y)
    aux = model.aux_loss()
    (ce + aux).backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(opts["clip_params"], grad_clip)   # dense params only (sparse grads can't be clipped this way)
    for k in ("dense", "sparse"):
        if k in opts:
            opts[k].step()
    gk = model.maintenance(step, opts)
    info.update(ce=float(ce.detach()), aux=float(torch.as_tensor(aux).detach()), phase=model.zs.phase)
    if gk is not None:
        info["gravekeeper"] = gk
    return info


# --------------------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    from counter import Spec, model_costs

    def mk(**kw):
        base = dict(vocab=50, d=64, n_layers=4, n_heads=4, n_kv=2, ctx=64, n_sub=8, pk_heads=2, half=16,
                    topk=4, gate_hidden=8, n_groups=2, warm_steps=60, soft_steps=60, dead_after=30, gk_every=10,
                    buf_size=256)
        base.update(kw)
        return ZombieLM(ZCfg(**base))

    # 1. param count == closed form (real default config + variants)
    for label, kw, spec in [
        ("default 12 layers", {}, dict(mem_layers=tuple(range(12)))),
        ("mem layers 6-11", dict(mem_layers=tuple(range(6, 12))), dict(mem_layers=tuple(range(6, 12)))),
        ("mem_only", dict(gate_mode="mem_only"), dict(mem_layers=tuple(range(12)), gate_mode="mem_only")),
        ("fixed_half", dict(gate_mode="fixed_half"), dict(mem_layers=tuple(range(12)), gate_mode="fixed_half")),
        ("no qnorm", dict(qnorm=False), dict(mem_layers=tuple(range(12)), qnorm=False)),
    ]:
        m = ZombieLM(ZCfg(**kw))
        s = Spec(n_kv=2, ffn_mult=4.0, **spec)
        a, b = count_params(m), model_costs(s)["total_params"]
        assert a == b, (label, a, b)
        print(f"[ok] params {label:18s} {a / 1e6:.3f}M == counter.py")

    # tiny data: markov chain
    g = torch.Generator().manual_seed(1)
    Tm = torch.softmax(torch.randn(50, 50, generator=g) * 3, -1)
    seq = [0]
    for _ in range(60000):
        seq.append(torch.multinomial(Tm[seq[-1]], 1, generator=g).item())
    data = torch.tensor(seq)

    def batch(n=16, L=32):
        ix = torch.randint(0, len(data) - L - 1, (n,))
        return (torch.stack([data[i:i + L] for i in ix]), torch.stack([data[i + 1:i + L + 1] for i in ix]))

    def batches():
        for _ in range(50):
            yield batch(32, 32)[0]

    # 2. KV-cache decode == full forward, in hard-routing inference
    m = mk()
    m.set_step(10_000)
    m.eval()
    x = torch.randint(0, 50, (2, 20))
    full = m(x)[0]
    lg, _, pst = m(x[:, :11], use_cache=True)
    outs = [lg]
    for t in range(11, 20):
        lg, _, pst = m(x[:, t:t + 1], past=pst, use_cache=True)
        outs.append(lg)
    err = (full - torch.cat(outs, 1)).abs().max().item()
    assert err < 1e-4, err
    print(f"[ok] zombie KV-cache decode == full forward (hard routing, max err {err:.1e})")

    # 3. 'subset' routing == 'where' routing, and both really mix paths (force different gates per token)
    m.zs.route = "subset"
    a = m(x)[0]
    m.zs.route = "where"
    b = m(x)[0]
    assert (a - b).abs().max() < 1e-5
    fr = []
    for f in m.mem_blocks():
        h = torch.randn(200, 64)
        fr.append(float((f.gate(h).squeeze(-1) > 0).float().mean()))
    print(f"[ok] route subset == where  (gate->B fraction per layer at init: {[round(v, 2) for v in fr]})")
    m.zs.route = "subset"

    # 4. causality
    y = x.clone()
    y[0, 12] = (y[0, 12] + 1) % 50
    dd = (m(x)[0] - m(y)[0]).abs().amax(-1)[0]
    assert dd[:12].max() < 1e-5 and dd[12:].max() > 0
    print("[ok] causal")

    # 5. full run through all phases: shadow -> kmeans init -> soft -> hard, with gravekeeper
    m = mk()
    opts = build_optimizers(m, lr=3e-3, cache_lr=1e-2)
    log = {}
    first = None
    t0 = time.time()
    for step in range(260):
        xb, yb = batch()
        info = train_step(m, opts, xb, yb, step, init_batches=batches, init_kw=dict(fit_batches=8, assign_tokens=20000))
        first = info["ce"] if first is None else first
        if "init" in info:
            ks = info["init"]["kmeans"]
            log["init@%d" % step] = [f"{k['empty_cells']}/{k['n_cells']} empty cells" for k in ks]
        log.setdefault(info["phase"], step)
        if step in (59, 60, 119, 120, 259):
            print(f"    step {step:3d} phase={info['phase']:6s} ce={info['ce']:.3f} aux={info['aux']:+.4f}")
    print(f"[ok] phases ran {log}  ce {first:.2f} -> {info['ce']:.2f}  ({time.time() - t0:.0f}s)")
    assert info["ce"] < first - 0.5
    gm = m.gate_means()
    assert all(v is not None and 0 <= v <= 1 for v in gm)
    print(f"[ok] mean gate per layer after training: {[round(v, 2) for v in gm]}")
    st = m.cache_stats()
    print(f"[ok] cache stats: {[{k: round(v, 3) for k, v in s.items()} for s in st]}")

    # 6. k-means init vs (a) untrained random values and (b) shadow-trained values.
    #    helper: capture the real input of layer 2's FFN with a hook, compare memory path A to Branch B
    def mem_vs_b(m):
        m.eval()
        m.zs.phase = "shadow"
        f = m.blocks[2].ffn
        cap = []
        h = f.register_forward_pre_hook(lambda mod, inp: cap.append(inp[0].reshape(-1, inp[0].shape[-1]).detach()))
        errs = []
        with torch.no_grad():
            for k, xb in enumerate(batches()):
                m(xb)
                zf = cap.pop()
                A, Bo = f.memory(zf), f.B(zf)
                errs.append(((A - Bo).pow(2).mean() / Bo.pow(2).mean()).item())
                if k >= 4:
                    break
        h.remove()
        return sum(errs) / len(errs)

    # (a) fresh model, no shadow training: k-means must be a much better start than random values
    m = mk(init="kmeans")
    before = mem_vs_b(m)
    m.kmeans_init(batches, fit_batches=8, assign_tokens=40000)
    after = mem_vs_b(m)
    print(f"[ok] fresh model: rel. error ||A-B||^2/||B||^2 random values {before:.3f} -> k-means {after:.3f}")
    assert after < before * 0.8, (before, after)
    # (b) informational: after shadow training, does k-means help or hurt?
    m = mk(init="kmeans")
    opts = build_optimizers(m, lr=3e-3, cache_lr=1e-2)
    for step in range(60):
        xb, yb = batch()
        train_step(m, opts, xb, yb, step)
    sh = mem_vs_b(m)
    m.kmeans_init(batches, fit_batches=8, assign_tokens=40000)
    km = mem_vs_b(m)
    print(f"[info] after 60 shadow steps: shadow-trained {sh:.3f} vs after k-means overwrite {km:.3f}"
          f"  ({'k-means helps' if km < sh else 'k-means HURTS at this scale -> ablate init=none vs kmeans'})")

    # 7. Gravekeeper: a dead sub-key row is detected, its key re-aimed, shared live slots kept, optimizer moments zeroed
    m = mk(dead_after=20, gk_every=5, warm_steps=10, soft_steps=10, init="none")
    opts = build_optimizers(m, lr=3e-3, cache_lr=1e-2)
    for step in range(40):
        xb, yb = batch()
        train_step(m, opts, xb, yb, step)
    c0 = m.caches[0]
    key_before = c0.keys[0, 0, 3].clone()
    nrm_live = c0.keys[0, 0].norm(dim=-1).mean().item()
    c0.last_used[0, 0, 3] = -10_000                       # force row (head0, half0, row3) dead
    slots = 3 * c0.n_sub + torch.arange(c0.n_sub)
    c0.slot_last[slots[:4]] = -10_000                     # 4 of its slots dead for all heads
    c0.slot_last[slots[4:]] = 40                          # rest explicitly 'recently used' (shared with live rows)
    c0.slot_last[slots[4:]] = 40                          # remaining slots explicitly 'recently used' (shared with live rows)
    live_vals = c0.values.weight[slots[4:]].detach().clone()
    dead_vals = c0.values.weight[slots[:4]].detach().clone()
    st = opts["dense"].state[c0.keys]
    assert st["exp_avg"][0, 0, 3].abs().sum() > 0         # moments exist before revive
    log = m.maintenance(40, opts)[0]                      # step 40 % gk_every == 0
    assert log["revived"] >= 1 and log["dead_rows"] >= 1, log
    assert (c0.keys[0, 0, 3] - key_before).abs().max() > 1e-3, "key not re-aimed"
    assert abs(c0.keys[0, 0, 3].norm().item() - nrm_live) / nrm_live < 0.5
    assert c0.last_used[0, 0, 3] == 40
    assert st["exp_avg"][0, 0, 3].abs().sum() == 0, "adam moments not reset"
    assert torch.equal(c0.values.weight[slots[4:]].detach(), live_vals), "live shared slots must not be overwritten"
    assert (c0.values.weight[slots[:4]].detach() - dead_vals).abs().max() > 1e-4, "dead slots not reset"
    print(f"[ok] Gravekeeper: {log} -- key re-aimed, 4 dead slots reset, live slots untouched, Adam moments zeroed")

    # 7b. stress config (no QueryNorm, high key lr) also runs end-to-end
    m = mk(qnorm=False, dead_after=20, gk_every=5, warm_steps=10, soft_steps=10, init="none")
    opts = build_optimizers(m, lr=3e-3, cache_lr=1e-2, key_lr=1e-2)
    for step in range(60):
        xb, yb = batch()
        info = train_step(m, opts, xb, yb, step)
    print(f"[ok] stress config (qnorm off, key lr 1e-2) ran; cache stats {[{k: round(v, 2) for k, v in s.items()} for s in m.cache_stats()]}")

    # 8. sparse optimizer: untouched value rows must not move
    m = mk(warm_steps=0, soft_steps=1000, init="none", gravekeeper=False, n_sub=32)   # soft phase: memory must get gradient
    opts = build_optimizers(m, lr=3e-3, cache_lr=1e-2)
    w0 = m.caches[0].values.weight.detach().clone()
    xb, yb = batch(2, 4)                                    # 8 tokens -> few slots touched
    train_step(m, opts, xb, yb, 0)
    moved = ((m.caches[0].values.weight - w0).abs().sum(1) > 0).float().mean().item()
    assert 0.0 < moved < 0.9, moved
    print(f"[ok] SparseAdam touched only {moved * 100:.1f}% of value rows in a tiny batch")

    # 9. ablation configs all build and run fwd/bwd
    for kw in (dict(gate_mode="mem_only"), dict(gate_mode="fixed_half"), dict(mem_layers=(2, 3)),
               dict(mem_layers=()), dict(init="none")):
        mm = mk(warm_steps=5, soft_steps=5, **kw)
        oo = build_optimizers(mm)
        for step in range(14):
            xb, yb = batch(8, 16)
            train_step(mm, oo, xb, yb, step)
        mm.eval()
        mm(xb)
    print("[ok] mem_only / fixed_half / partial (2,3) / no-mem layers / init=none all train")
    print("zombie selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--route", default="subset", choices=["subset", "where"])
    ap.add_argument("--p-b", type=float, default=None, help="force fraction of tokens routed to B (bench only)")
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.count or a.bench:
        m = ZombieLM(ZCfg(route=a.route))
        print(f"zombie_v6: {count_params(m):,} params")
        if a.bench:
            m.set_step(10 ** 9)
            m.zs.force_p = a.p_b
            print(bench(m, a.prompt, a.gen, threads=a.threads))
