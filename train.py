#!/usr/bin/env python3
"""train.py -- train ONE config (baseline or Zombie) on data/{train,val}.bin and write a full result.json.

    python train.py --name zombie --data data --out runs --seed 0
    python train.py --name gqa1x  --data data --out runs --seed 0

Fair-comparison guarantees: identical data order per seed (own numpy RNG, independent of the model), identical
optimizer/schedule/clipping for all models, identical fixed eval windows, ppl reported under hard routing.

Features: fp16 (T4) / bf16 (A100) autocast with GradScaler, gradient accumulation, cosine LR, atomic checkpoints with
exact resume (--resume), JSONL logging, final eval (val/train ppl, dead neurons, slot entropy, routing, PTQ, KV recall).
Colab disconnect?  Just re-run the same command with --resume (run_ablations.py does this for you).
"""
import argparse
import contextlib
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch

import configs
import counter
import evaluate
import kv_recall
from baseline import count_params
from moe import MoELM
from zombie_v6 import ZombieLM, apply_lr_scale, build_optimizers, lr_at


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=configs.ALL + configs.EXTRA)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (effective batch = batch*accum)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cache-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500, help="0 = only final eval")
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--final-eval-batches", type=int, default=200)
    ap.add_argument("--dead-batches", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if result.json exists")
    ap.add_argument("--keep-ckpt", action="store_true")
    ap.add_argument("--no-ptq", action="store_true")
    ap.add_argument("--revive-neurons", action="store_true", help="dead-FFN-neuron revival (also on for *_nrev configs)")
    ap.add_argument("--neuron-every", type=int, default=500)
    ap.add_argument("--neuron-dead-frac", type=float, default=0.001, help="dead = active on < this fraction of tokens")
    ap.add_argument("--neuron-min-tokens", type=int, default=500_000, help="min tokens in a window before judging")
    ap.add_argument("--tiny", action="store_true", help="d=64, 4 layers, small cache: smoke tests only")
    # model overrides (None = config default)
    for k, t in [("d", int), ("n_layers", int), ("n_heads", int), ("n_sub", int), ("half", int), ("pk_heads", int),
                 ("topk", int), ("gate_target", float), ("warm_steps", int), ("soft_steps", int),
                 ("dead_after", int), ("gk_every", int)]:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=t, default=None)
    return ap.parse_args()


def get_batch(tokens, rng, batch, seq, device):
    ix = rng.integers(0, len(tokens) - seq - 1, size=batch)
    x = np.stack([tokens[i:i + seq] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + 1 + seq] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def build_dense_opts(model, lr, wd, betas=(0.9, 0.95)):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    groups = [dict(params=decay, weight_decay=wd, base_lr=lr), dict(params=nodecay, weight_decay=0.0, base_lr=lr)]
    return dict(dense=torch.optim.AdamW(groups, lr=lr, betas=betas), clip_params=decay + nodecay)


def rng_state():
    st = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    # torch.load(map_location=<cuda>) moves these tensors to the GPU, but the RNG setters need CPU uint8 tensors
    as_cpu_u8 = lambda t: torch.as_tensor(t).detach().cpu().to(torch.uint8)
    torch.set_rng_state(as_cpu_u8(st["torch"]))
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([as_cpu_u8(t) for t in st["cuda"]])


def main():
    a = parse()
    run_dir = os.path.join(a.out, f"{a.name}_s{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    res_path = os.path.join(run_dir, "result.json")
    if os.path.exists(res_path) and not a.force:
        print(f"[skip] {res_path} exists")
        return

    # ---- device / precision
    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    amp = a.amp
    if amp == "auto":
        if dev.startswith("cuda"):
            amp = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
        else:
            amp = "none"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    dtype_dev = dev.split(":")[0]

    def amp_ctx():
        return torch.autocast(device_type=dtype_dev, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(dtype_dev, enabled=(amp == "fp16" and dtype_dev == "cuda"))

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if dtype_dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- data
    with open(os.path.join(a.data, "meta.json")) as f:
        meta = json.load(f)
    train = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    data_rng = np.random.default_rng(a.seed)

    # ---- model
    ov = {k: getattr(a, k) for k in ["d", "n_layers", "n_heads", "n_sub", "half", "pk_heads", "topk", "gate_target",
                                     "warm_steps", "soft_steps", "dead_after", "gk_every"]}
    if configs.is_zombie(a.name):
        if ov["warm_steps"] is None:
            ov["warm_steps"] = int(a.steps * 1000 / 6100)          # 1000/6100 of the run in shadow phase
        if ov["soft_steps"] is None:
            ov["soft_steps"] = int(a.steps * 2000 / 6100)          # next 2000/6100 soft gate, then hard STE
    model, cfg, opt_over = configs.build(a.name, meta["vocab"], a.seq, ov, a.tiny)
    model.to(dev)
    is_z = isinstance(model, ZombieLM)
    is_moe = isinstance(model, MoELM)
    has_aux = hasattr(model, "aux_loss")                 # Zombie (gate/distill loss) and MoE (load balance)
    configs.save_config(run_dir, a.name, cfg)
    n_params = count_params(model)
    print(f"[{a.name} seed {a.seed}] {n_params / 1e6:.2f}M params | device {dev} | amp {amp} | "
          f"{a.batch}x{a.accum}x{a.seq} tok/step | {a.steps} steps"
          + (f" | warm {cfg.warm_steps} soft {cfg.soft_steps}" if is_z else ""))
    opts = (build_optimizers(model, a.lr, a.cache_lr, opt_over.get("key_lr"), a.wd) if is_z
            else build_dense_opts(model, a.lr, a.wd))

    keeper = None
    if a.revive_neurons or a.name.endswith("_nrev"):
        from neuron_keeper import NeuronKeeper
        keeper = NeuronKeeper(model, every=a.neuron_every, dead_frac=a.neuron_dead_frac, min_tokens=a.neuron_min_tokens)
        print(f"  neuron revival ON: {len(keeper.layers)} FFN layers, check every {a.neuron_every} steps")

    def init_batches():                                           # TRAIN data only, own RNG, infinite
        r = np.random.default_rng(a.seed + 12345)
        while True:
            yield get_batch(train, r, a.batch, a.seq, dev)[0]

    # ---- resume
    start = 0
    ckpt = os.path.join(run_dir, "ckpt.pt")
    if a.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        for k in ("dense", "sparse"):
            if k in opts and k in st["opts"]:
                opts[k].load_state_dict(st["opts"][k])
        if scaler.is_enabled() and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        if is_z:
            model.load_extra_state(st["extra"])
        data_rng.bit_generator.state = st["data_rng"]
        set_rng_state(st["rng"])
        start = st["step"]
        print(f"[resume] from step {start}")

    def save_ckpt(next_step):
        st = dict(model=model.state_dict(), opts={k: opts[k].state_dict() for k in ("dense", "sparse") if k in opts},
                  scaler=scaler.state_dict() if scaler.is_enabled() else None,
                  extra=model.extra_state() if is_z else None, data_rng=data_rng.bit_generator.state,
                  rng=rng_state(), step=next_step)
        torch.save(st, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    # ---- one optimizer step
    def train_step(step):
        info = {}
        if is_z:
            model.set_step(step)
            if step == cfg.warm_steps and not model.mem_started:
                info["init"] = model.begin_memory_phase(step, opts, init_batches)
        model.train()
        for k in ("dense", "sparse"):
            if k in opts:
                opts[k].zero_grad(set_to_none=True)
        ce_sum = 0.0
        for _ in range(a.accum):
            x, y = get_batch(train, data_rng, a.batch, a.seq, dev)
            with amp_ctx():
                _, ce, _ = model(x, y)
                aux = model.aux_loss() if has_aux else 0.0
            scaler.scale((ce + aux) / a.accum).backward()
            ce_sum += float(ce.detach())
        for k in ("dense", "sparse"):
            if k in opts:
                scaler.unscale_(opts[k])
        if a.clip:
            torch.nn.utils.clip_grad_norm_(opts["clip_params"], a.clip)
        for k in ("dense", "sparse"):
            if k in opts:
                scaler.step(opts[k])
        scaler.update()
        if is_z:
            gk = model.maintenance(step, opts)
            if gk is not None:
                info["gravekeeper"] = dict(dead_rows=sum(g["dead_rows"] for g in gk), revived=sum(g["revived"] for g in gk),
                                           slots_reset=sum(g["slots_reset"] for g in gk))
        if keeper is not None:
            nk = keeper.step(step, opts)
            if nk:
                info["neurons"] = dict(dead=nk["dead"], revived=nk["revived"])
        info["ce"] = ce_sum / a.accum
        return info

    # ---- loop
    logf = open(os.path.join(run_dir, "log.jsonl"), "a")
    t_train, t_win, run_ce, run_n, gk_acc = 0.0, 0.0, 0.0, 0, dict(dead_rows=0, revived=0, slots_reset=0)
    nk_acc, nk_total = dict(dead=0, revived=0), dict(dead=0, revived=0)
    tok_per_step = a.batch * a.accum * a.seq
    for step in range(start, a.steps):
        apply_lr_scale(opts, lr_at(step, a.steps, a.warmup))
        t0 = time.time()
        info = train_step(step)
        if dtype_dev == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        t_win += time.time() - t0
        if not math.isfinite(info["ce"]) and amp != "fp16":
            raise SystemExit(f"non-finite loss at step {step}; aborting")
        run_ce += info["ce"]
        run_n += 1
        for k in gk_acc:
            gk_acc[k] += info.get("gravekeeper", {}).get(k, 0)
        for k in nk_acc:
            nk_acc[k] += info.get("neurons", {}).get(k, 0)
        if "init" in info:
            km = info["init"].get("kmeans")
            msg = dict(step=step, event="memory_phase_start", kmeans=km)
            logf.write(json.dumps(msg) + "\n")
            print(f"  step {step}: memory enabled; k-means init: {km}")
        if (step + 1) % a.log_every == 0 or step == a.steps - 1:
            rec = dict(step=step + 1, ce=run_ce / run_n, lr_scale=lr_at(step, a.steps, a.warmup),
                       tok_s=tok_per_step * run_n / max(t_win, 1e-9),
                       phase=model.zs.phase if is_z else None)
            if is_z:
                rec["gate_mean"] = model.gate_means()
                if model.zs.phase != "shadow":
                    rec["cache"] = [dict(H=round(s["slot_entropy_norm"], 4), used=round(s["slots_used_frac"], 4),
                                         dead_rows=s["dead_rows"]) for s in model.cache_stats(reset=True)]
                rec["gravekeeper"] = dict(gk_acc)
            if is_moe:
                ms_ = model.moe_stats(reset=True)
                rec["moe"] = dict(entropy=round(ms_["entropy_norm_mean"], 4), dead_experts=ms_["dead_experts"],
                                  max_over_mean=round(ms_["max_over_mean"], 3))
            if keeper is not None:
                rec["neurons"] = dict(nk_acc)
                for k in nk_total:
                    nk_total[k] += nk_acc[k]
                nk_acc = dict(dead=0, revived=0)
            if dtype_dev == "cuda":
                rec["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"  step {step + 1:5d}/{a.steps} ce {rec['ce']:.4f}"
                  + (f" phase {rec['phase']}" if is_z else "")
                  + (f" gate {np.mean([g for g in rec['gate_mean'] if g is not None]):.2f}"
                     if is_z and any(g is not None for g in rec['gate_mean']) else "")
                  + (f" | {rec['tok_s']:.0f} tok/s" if rec["tok_s"] else ""))
            run_ce, run_n, t_win = 0.0, 0, 0.0
            gk_acc = dict(dead_rows=0, revived=0, slots_reset=0)
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            ev = evaluate.eval_loss(model, evaluate.windows(val, a.batch, a.seq, a.eval_batches, 999, dev), amp_ctx)
            logf.write(json.dumps(dict(step=step + 1, val_loss=ev["loss"], val_ppl=ev["ppl"],
                                       phase=model.zs.phase if is_z else None)) + "\n")
            logf.flush()
            print(f"  [eval] step {step + 1} val ppl {ev['ppl']:.3f}")
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            save_ckpt(step + 1)

    # ================= final evaluation (hard routing, memory on) =================
    print("[final eval]")
    model.eval()
    if is_z:
        model.set_step(10 ** 9)
    res = dict(name=a.name, seed=a.seed, steps=a.steps, amp=amp, device=dev, batch=a.batch, accum=a.accum, seq=a.seq,
               train_time_s=t_train, train_tok_s=tok_per_step * (a.steps - start) / max(t_train, 1e-9),
               params_total=n_params, vocab=meta["vocab"], train_tokens_seen=tok_per_step * a.steps,
               data=dict(train_tokens=meta["train_tokens"], val_tokens=meta["val_tokens"]))
    W = lambda n, s: evaluate.windows(val, a.batch, a.seq, n, s, dev)
    res["val"] = evaluate.eval_loss(model, W(a.final_eval_batches, 999), amp_ctx)
    res["train_eval"] = evaluate.eval_loss(model, evaluate.windows(train, a.batch, a.seq, min(a.final_eval_batches, 50), 998, dev), amp_ctx)
    test_path = os.path.join(a.data, "test.bin")                  # only exists if prep_data got --test
    if os.path.exists(test_path):
        test = np.memmap(test_path, dtype=np.uint16, mode="r")
        if len(test) > a.seq + 2:
            res["test"] = evaluate.eval_loss(model, evaluate.windows(test, a.batch, a.seq, a.final_eval_batches, 995, dev), amp_ctx)
    res["dead_neurons"] = evaluate.dead_neurons(model, W(a.dead_batches, 997))
    res["neuron_revival"] = dict(enabled=keeper is not None, **nk_total)
    if keeper is not None:
        keeper.remove()
    rs = evaluate.routing_stats(model, W(50, 996)) if is_z else None
    res["routing"] = rs
    if is_moe:
        res["moe_routing"] = evaluate.moe_routing(model, W(50, 996))
    p_b =(rs or {}).get("p_b_mean")
    cs = counter.model_costs(configs.spec_from(cfg, p_b if p_b is not None else 0.5, a.name), wb=2, vb=2)
    res["costs_fp16"] = dict(macs=cs["macs"], mb_read=cs["bytes"] / 1e6, active_core=cs["active_core"],
                             total_params_counter=cs["total_params"], p_b_used=p_b if p_b is not None else 0.5)
    if not a.no_ptq:
        ptq = {}
        plan = [("w8", 8, None)] + ([("w8_c8", 8, 8), ("w8_c4", 8, 4)] if is_z else [])
        for label, wb, cb in plan:
            m2 = evaluate.fakequant_(copy.deepcopy(model), wb, cb)
            ptq[label] = evaluate.eval_loss(m2, W(a.final_eval_batches, 999), amp_ctx)["ppl"]
            del m2
        res["ptq_ppl"] = ptq
    kvp = os.path.join(a.data, "kv_test.json")
    tokp = os.path.join(a.data, "tokenizer.json")
    if os.path.exists(kvp) and os.path.exists(tokp):
        from tokenizers import Tokenizer
        res["kv_recall"] = kv_recall.kv_eval(model, Tokenizer.from_file(tokp), kv_recall.load_pairs(kvp), dev, amp_ctx)
    if dtype_dev == "cuda":
        res["gpu_mem_mb_peak"] = torch.cuda.max_memory_allocated() / 2 ** 20

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    with open(res_path, "w") as f:
        json.dump(res, f, indent=1)
    if not a.keep_ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"[done] val ppl {res['val']['ppl']:.3f} | train ppl {res['train_eval']['ppl']:.3f}"
          + (f" | dead {res['dead_neurons']['pct_dead']:.1f}%" if res["dead_neurons"] else "")
          + (f" | p_B {p_b:.2f} slotH {np.mean([c['slot_entropy_norm'] for c in rs['caches']]):.3f}" if rs and p_b is not None else "")
          + (f" | kv acc {res['kv_recall']['acc']:.3f}" if "kv_recall" in res else "")
          + f"\n       -> {res_path}")


if __name__ == "__main__":
    main()
