"""configs.py -- every run in the paper as a named config, plus model (de)serialisation.

HEADLINE (3 seeds):  mha, gqa, gqa1x, zombie
ABLATION (1 seed):   attn_only, zombie_mem6_11, zombie_memonly, zombie_fixed_half, zombie_nogk, zombie_noinit,
                     zombie_gt03, zombie_gt07, zombie_stress, zombie_stress_nogk

gqa1x (GQA + 1x FFN, Branch-B-sized) is the control that matters: Zombie only "wins" if its memory buys lower
ppl than a plain 1x FFN at ~the same MACs/bytes.
"""
import dataclasses
import json
import os

import torch

import counter
from baseline import Cfg, PRESETS, Transformer
from moe import MCfg, MoELM
from zombie_v6 import ZCfg, ZombieLM

BASELINES = ["mha", "gqa", "gqa1x", "attn_only"]

XC_NSUB = 79

# name -> (cfg overrides, optimizer overrides)
ZOMBIES = {
    "zombie":            ({}, {}),
    "zombie_mem6_11":    (dict(mem_layers="second_half"), {}),          # layers 0-5 keep dense 4x GELU FFN
    "zombie_memonly":    (dict(gate_mode="mem_only"), {}),
    # memory-only sized to MoE's TOTAL params (28.9M), extra budget either in slots or in attention depth
    "zombie_memonly_b29":  (dict(gate_mode="mem_only", n_sub=146), {}),                     # 12 layers, 21.3k slots/cache
    "zombie_memonly_L16":  (dict(gate_mode="mem_only", n_layers=16, n_sub=136), {}),        # 16 layers, attention 4.4M -> 5.9M
    # attention-heavy, SAME total params as moe (22.8M at vocab 256): FFN budget moved from memory slots into attention depth
    "zombie_memonly_L20":  (dict(gate_mode="mem_only", n_layers=20, n_sub=126), {}),        # attention 7.4M (32%), 15.9k slots/cache
    "zombie_memonly_L24":  (dict(gate_mode="mem_only", n_layers=24, n_sub=115), {}),        # attention 8.9M (39%), 13.2k slots/cache
    # L24xc = L24 + routed MoE experts in every layer (next to the memory) + 4 KV heads instead of 2 (attention buff);
    # n_sub shrunk so total params stay ~ moe (22.8M).  ReDo-for-attention is ON by default in reason_train for *xc models.
    "zombie_memonly_L24xc": (dict(gate_mode="mem_only", n_layers=24, n_kv=4, n_sub=XC_NSUB, moe_experts=8, moe_top_k=2, moe_hidden=24), {}),
    "zombie_fixed_half": (dict(gate_mode="fixed_half"), {}),
    "zombie_nogk":       (dict(gravekeeper=False), {}),
    "zombie_noinit":     (dict(init="none"), {}),
    "zombie_gt03":       (dict(gate_target=0.3), {}),
    "zombie_gt07":       (dict(gate_target=0.7), {}),
    "zombie_stress":     (dict(qnorm=False), dict(key_lr=1e-3)),         # forces collapse so Gravekeeper must fire
    "zombie_stress_nogk": (dict(qnorm=False, gravekeeper=False), dict(key_lr=1e-3)),
}
# MoE variants (same total params as moe at other depths): name -> MCfg overrides
MOES = {"moe": {}, "moe_L16": dict(n_layers=16, exp_hidden=113)}
EXTRA = ["zombie_memonly_b29", "zombie_memonly_L16", "zombie_memonly_L20", "zombie_memonly_L24", "zombie_memonly_L24xc", "moe_L16"]     # matched-budget runs (not part of `sweep`)
# "moe": plain top-2 MoE (12 experts, hidden 164) matched to Zombie on total params and active MACs (moe.py)
HEADLINE = ["mha", "gqa", "gqa1x", "moe", "zombie"]
# "<base>_nrev" = same config + dead-FFN-neuron revival (neuron_keeper.py). Baseline gets it too, so it stays fair.
NREV = ["gqa1x_nrev", "zombie_nrev"]
ABLATION = ["attn_only", "zombie_mem6_11", "zombie_memonly", "zombie_fixed_half", "zombie_nogk",
            "zombie_noinit", "zombie_gt03", "zombie_gt07", "zombie_stress", "zombie_stress_nogk"] + NREV
ALL = HEADLINE + ABLATION


def split_name(name):
    """'zombie_nrev' -> ('zombie', True)"""
    return (name[:-5], True) if name.endswith("_nrev") else (name, False)

# small config for smoke tests / CPU debugging only
TINY = dict(d=64, n_layers=4, n_heads=4)
TINY_Z = dict(n_sub=16, half=16, pk_heads=2, topk=4, gate_hidden=8, dead_after=20, gk_every=10, buf_size=256)


def is_zombie(name):
    return split_name(name)[0] in ZOMBIES


def build(name, vocab, ctx, overrides=None, tiny=False):
    """-> (model, cfg, opt_overrides)"""
    name = split_name(name)[0]
    ov ={k: v for k, v in (overrides or {}).items() if v is not None}
    if name in PRESETS:
        kw = dict(PRESETS[name])
        if tiny:
            kw.update(TINY)
        fields = {f.name for f in dataclasses.fields(Cfg)}
        kw.update({k: v for k, v in ov.items() if k in fields})
        cfg = Cfg(vocab=vocab, ctx=ctx, **kw)
        return Transformer(cfg), cfg, {}
    if name in MOES:
        kw = dict(MOES[name])
        if tiny:
            kw.update(TINY)
            kw.update(n_experts=4, top_k=2, exp_hidden=32)
        fields = {f.name for f in dataclasses.fields(MCfg)}
        kw.update({k: v for k, v in ov.items() if k in fields})
        cfg = MCfg(vocab=vocab, ctx=ctx, **kw)
        return MoELM(cfg), cfg, {}
    if name not in ZOMBIES:
        raise SystemExit(f"unknown config '{name}'. choices: {ALL}")
    zc, oo = ZOMBIES[name]
    kw = dict(zc)
    if tiny:
        kw.update(TINY)
        kw.update(TINY_Z)
    fields = {f.name for f in dataclasses.fields(ZCfg)}
    kw.update({k: v for k, v in ov.items() if k in fields})
    L = kw.get("n_layers", ZCfg.n_layers)
    if kw.get("mem_layers") == "second_half":
        kw["mem_layers"] = tuple(range(L // 2, L))
    cfg = ZCfg(vocab=vocab, ctx=ctx, **kw)
    return ZombieLM(cfg), cfg, dict(oo)


def spec_from(cfg, p_b=0.5, name=""):
    """counter.Spec equivalent to a model config (so closed-form costs match the real model)."""
    common = dict(name=name, d=cfg.d, L=cfg.n_layers, H=cfg.n_heads, vocab=cfg.vocab, n_kv=cfg.n_kv,
                  ffn_mult=cfg.ffn_mult)
    if isinstance(cfg, ZCfg):
        return counter.Spec(**common, mem_layers=tuple(cfg.mem_layers), n_groups=cfg.n_groups, n_sub=cfg.n_sub,
                            pk_heads=cfg.pk_heads, half=cfg.half, topk=cfg.topk, gate_hidden=cfg.gate_hidden,
                            gate_mode=cfg.gate_mode, qnorm=cfg.qnorm, p_b=p_b)
    if isinstance(cfg, MCfg):
        return counter.Spec(**common, moe_layers=tuple(range(cfg.n_layers)), n_experts=cfg.n_experts,
                            top_k=cfg.top_k, exp_hidden=cfg.exp_hidden)
    return counter.Spec(**common)


def save_config(run_dir, name, cfg):
    d = dataclasses.asdict(cfg)
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        kind = "zombie" if isinstance(cfg, ZCfg) else ("moe" if isinstance(cfg, MCfg) else "baseline")
        json.dump(dict(name=name, kind=kind, cfg=d), f, indent=1)


def load_run(run_dir, map_location="cpu"):
    """Rebuild a trained model from <run_dir>/config.json + model.pt."""
    with open(os.path.join(run_dir, "config.json")) as f:
        j = json.load(f)
    if j["kind"] == "zombie":
        d = dict(j["cfg"])
        d["mem_layers"] = tuple(d["mem_layers"])
        cfg = ZCfg(**d)
        model = ZombieLM(cfg)
    elif j["kind"] == "moe":
        cfg = MCfg(**j["cfg"])
        model = MoELM(cfg)
    else:
        cfg = Cfg(**j["cfg"])
        model = Transformer(cfg)
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location=map_location)
    model.load_state_dict(sd)
    model.eval()
    if isinstance(model, ZombieLM):
        model.set_step(10 ** 9)           # hard routing, memory on
    return model, cfg, j["name"]
