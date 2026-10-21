#!/usr/bin/env python3
"""reason_train.py -- train ONE model from scratch on the reasoning mixture (reason_tasks.py) and evaluate it.

    python reason_train.py --name zombie_memonly_b29 --seed 0 --out runs_reason/final
    python reason_train.py --name moe --seed 0 --lr 1e-3 --steps 3000

Same architectures, optimizer, schedule and AMP logic as train.py (imported, not copied), so the models are the exact
ones from the language-modelling runs.  Differences (all deliberate and identical for every model):
  * own small token vocabulary (reason_tasks.py), padded to --vocab 256 ids;  seq 256 x batch 32 = 8192 tok/step
  * loss on ANSWER tokens only
  * zombie_memonly: memory is used from step 0 (mem_only never has a shadow phase); memory-phase bookkeeping starts at
    step 0 (warm_steps=0).  In the LM runs it started at step 1000.
  * data = pure function of (seed, step) -> exact resume with no data-RNG state; a DataLoader worker builds batches
    ahead of time so generation never stalls the GPU.
Writes <run-dir>/result.json (val + test accuracy for every task x difficulty, Wilson CIs, params, timing).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation -> lower reserved VRAM
import argparse
import contextlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import configs
import attn_redo
import reason_tasks as RT
from baseline import count_params
from train import build_dense_opts, rng_state, set_rng_state
from zombie_v6 import ZombieLM, apply_lr_scale, build_optimizers, lr_at

CHOICES = ["mha", "gqa", "gqa1x", "moe", "moe_L16", "zombie_memonly", "zombie_memonly_b29", "zombie_memonly_L16", "zombie_memonly_L20", "zombie_memonly_L24", "zombie_memonly_L24xc"]
CACHE_LR_RATIO = 1e-3 / 3e-4            # same cache-lr / lr ratio as the LM runs


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=CHOICES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None, help="default: <out>/<name>_s<seed>")
    ap.add_argument("--out", default="runs_reason/final")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache-lr", type=float, default=None, help="default: lr * 1e-3/3e-4 (as in the LM runs)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--facts", type=int, default=2048, help="number of stored facts (0 = no facts task)")
    ap.add_argument("--mix", default=None, help="e.g. mqar=.2,hop=.2,add=.2,state=.2,facts=.2")
    ap.add_argument("--diffs", default=None, help="DIAGNOSTIC: restrict TRAIN difficulties, e.g. mqar=2:3:4,hop=1:2,state=2:3 "
                    "(larger test difficulties then count as OOD).  Use with --skip-test; not for the final protocol.")
    ap.add_argument("--curriculum", type=int, default=0, help="ramp steps: training difficulty range grows from the easiest 25%% "
                    "to all train difficulties over this many steps (0 = off).  Test/OOD difficulties are unchanged.")
    ap.add_argument("--redo-every", type=int, default=0, help="ReDo for attention: every N steps recycle dormant heads (0 = off)")
    ap.add_argument("--redo-tau", type=float, default=0.1, help="a head is dormant when its score < tau (1.0 = average head)")
    ap.add_argument("--no-redo", action="store_true", help="*xc models run ReDo by default (every 250 steps); this turns it off")
    ap.add_argument("--dormancy-every", type=int, default=0, help="log the fraction of dormant attention heads every N steps (0 = off)")
    ap.add_argument("--eval-n", type=int, default=500, help="instances per (task,difficulty) in the final eval")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n-mid", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--tiny", action="store_true", help="d=64, 4 layers: tests only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-ckpt", action="store_true")
    ap.add_argument("--skip-test", action="store_true", help="tuning runs: evaluate on val only")
    return ap.parse_args(argv)


class StepBatches(torch.utils.data.Dataset):
    """index i -> the batch for training step start+i (pure function of seed and step)."""
    def __init__(self, seed, start, steps, batch, seq, facts, mix, ramp=0):
        self.seed, self.start, self.n, self.batch, self.seq, self.facts, self.mix = seed, start, steps - start, batch, seq, facts, mix
        self.ramp = ramp

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        step = self.start + i
        frac = 1.0 if self.ramp <= 0 else min(1.0, 0.25 + 0.75 * step / self.ramp)   # pure function of step -> exact resume
        x, y, m = RT.make_batch(self.seed, step, self.batch, self.seq, self.facts, self.mix, frac)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def parse_mix(s):
    if not s:
        return dict(RT.DEFAULT_MIX)
    d = {k: 0.0 for k in RT.DEFAULT_MIX}
    for kv in s.split(","):
        k, v = kv.split("=")
        assert k in d, k
        d[k] = float(v)
    return d


def main(argv=None):
    a = parse(argv)
    if a.name.endswith("xc") and not a.no_redo and a.redo_every == 0:
        a.redo_every = 250
    run_dir = a.run_dir or os.path.join(a.out, f"{a.name}_s{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    res_path = os.path.join(run_dir, "result.json")
    if os.path.exists(res_path) and not a.force:
        print(f"[skip] {res_path} exists")
        return
    assert a.vocab >= RT.VOCAB_MIN and a.seq >= RT.max_len() - 1, (a.vocab, RT.VOCAB_MIN, a.seq, RT.max_len())

    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    amp = a.amp
    if amp == "auto":
        amp = ("bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16") if dev.startswith("cuda") else "none"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    dtype_dev = dev.split(":")[0]

    def amp_ctx():
        return torch.autocast(device_type=dtype_dev, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    scaler = torch.amp.GradScaler(dtype_dev, enabled=(amp == "fp16" and dtype_dev == "cuda"))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if dtype_dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if a.diffs:
        for kv in a.diffs.split(","):
            k, v = kv.split("=")
            assert k in RT.REASONING, k
            RT.TASKS[k]["train"] = [int(x) for x in v.split(":")]
        print("[diffs] train difficulties:", {k: RT.TASKS[k]["train"] for k in RT.REASONING}, flush=True)
    mix = parse_mix(a.mix)
    facts = RT.make_fact_table(a.facts) if a.facts > 0 else None
    if not facts:
        mix["facts"] = 0.0

    ov = dict(warm_steps=0, soft_steps=0) if configs.is_zombie(a.name) else {}
    model, cfg, opt_over = configs.build(a.name, a.vocab, a.seq, ov, a.tiny)
    model.to(dev)
    is_z = isinstance(model, ZombieLM)
    if is_z:
        assert cfg.gate_mode == "mem_only", "reason_train supports memory-only zombies only"
    has_aux = hasattr(model, "aux_loss")
    n_params = count_params(model)
    cache_lr = a.cache_lr if a.cache_lr is not None else a.lr * CACHE_LR_RATIO
    opts = build_optimizers(model, a.lr, cache_lr, opt_over.get("key_lr"), a.wd) if is_z else build_dense_opts(model, a.lr, a.wd)
    print(f"[{a.name} seed {a.seed}] {n_params / 1e6:.2f}M params | device {dev} | amp {amp} | lr {a.lr:g} | "
          f"{a.batch}x{a.seq} tok/step | {a.steps} steps | facts {a.facts}")

    # ---- resume
    start = 0
    ckpt = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        for k in ("dense", "sparse"):
            if k in opts and k in st["opts"]:
                opts[k].load_state_dict(st["opts"][k])
        if scaler.is_enabled() and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        if is_z:
            model.load_extra_state(st["extra"])
        set_rng_state(st["rng"])
        start = st["step"]
        print(f"[resume] from step {start}")

    def save_ckpt(next_step):
        st = dict(model=model.state_dict(), opts={k: opts[k].state_dict() for k in ("dense", "sparse") if k in opts},
                  scaler=scaler.state_dict() if scaler.is_enabled() else None,
                  extra=model.extra_state() if is_z else None, rng=rng_state(), step=next_step)
        torch.save(st, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    val_mid = RT.build_eval("val", a.eval_n_mid, facts)
    ds = StepBatches(a.seed, start, a.steps, a.batch, a.seq, facts, mix, a.curriculum)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                         prefetch_factor=4 if a.workers > 0 else None,
                                         persistent_workers=a.workers > 0)
    logf = open(os.path.join(run_dir, "log.jsonl"), "a")
    t_train = t_win = 0.0
    run_ce = run_acc = 0.0
    run_n = 0
    tok_per_step = a.batch * a.seq
    model.train()
    for i, (x, y, m) in enumerate(loader):
        step = start + i
        apply_lr_scale(opts, lr_at(step, a.steps, a.warmup))
        t0 = time.time()
        if is_z:
            model.set_step(step)
            if step == cfg.warm_steps and not model.mem_started:
                model.begin_memory_phase(step, opts, None)
        model.train()
        for k in ("dense", "sparse"):
            if k in opts:
                opts[k].zero_grad(set_to_none=True)
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        if a.redo_every and step > 0 and step % a.redo_every == 0:
            hs, _ = attn_redo.head_scores(model, x, amp_ctx)
            nr = attn_redo.redo_reset(model, opts, hs, a.redo_tau, a.seed, step)
            logf.write(json.dumps(dict(step=step, redo_heads_reset=nr)) + "\n")
            logf.flush()
            print(f"  [redo] step {step}: recycled {nr} dormant attention heads", flush=True)
        with amp_ctx():
            logits = model(x)[0]
            aux = model.aux_loss() if has_aux else 0.0
        tokl = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view_as(y)
        denom = m.sum().clamp(min=1.0)
        ce = (tokl * m).sum() / denom
        scaler.scale(ce + aux).backward()
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
            model.maintenance(step, opts)
        if dtype_dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        t_train += dt
        t_win += dt
        cev = float(ce.detach())
        if not math.isfinite(cev) and amp != "fp16":
            raise SystemExit(f"non-finite loss at step {step}")
        run_ce += cev
        run_acc += float((((logits.argmax(-1) == y).float() * m).sum() / denom).detach())
        run_n += 1
        if (step + 1) % a.log_every == 0 or step == a.steps - 1:
            rec = dict(step=step + 1, ce=run_ce / run_n, ans_tok_acc=run_acc / run_n,
                       tok_s=tok_per_step * run_n / max(t_win, 1e-9))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"  step {step + 1:5d}/{a.steps} ans-ce {rec['ce']:.4f} ans-tok-acc {rec['ans_tok_acc']:.3f} | {rec['tok_s']:.0f} tok/s")
            run_ce = run_acc = 0.0
            run_n, t_win = 0, 0.0
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            if is_z:
                model.set_step(step)
            ev = RT.eval_sets(model, val_mid, dev, amp_ctx)
            sm = RT.summarize(ev)
            logf.write(json.dumps(dict(step=step + 1, val={k: round(v["acc"], 4) for k, v in ev.items()})) + "\n")
            logf.flush()
            print("  [val] " + " | ".join(f"{t} id {v['id']:.2f}" for t, v in sm.items() if v["id"] is not None))
        if a.dormancy_every and (step + 1) % a.dormancy_every == 0:
            hs, cs = attn_redo.head_scores(model, x, amp_ctx)
            ds_ = attn_redo.summarize(hs, cs, a.redo_tau)
            logf.write(json.dumps(dict(step=step + 1, dormancy=ds_)) + "\n")
            logf.flush()
            print(f"  [dormancy] step {step + 1}: heads<{a.redo_tau:g}: {ds_['heads_dormant']}/{ds_['heads_total']} "
                  f"({100 * ds_['heads_frac']:.1f}%) | channels: {100 * ds_['chan_frac']:.1f}% | min head score {ds_['min_head_score']:.3f}", flush=True)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            save_ckpt(step + 1)

    # ---------------- final eval
    print("[final eval]")
    model.eval()
    if is_z:
        model.set_step(10 ** 9)
    res = dict(name=a.name, seed=a.seed, lr=a.lr, cache_lr=cache_lr, steps=a.steps, batch=a.batch, seq=a.seq,
               vocab=a.vocab, amp=amp, device=dev, params_total=n_params, facts=a.facts, mix=mix,
               train_time_s=t_train, train_tok_s=tok_per_step * max(a.steps - start, 1) / max(t_train, 1e-9),
               train_tokens_seen=tok_per_step * a.steps, fact_seed=RT.FACT_SEED, eval_n=a.eval_n)
    splits = ["val"] if a.skip_test else ["val", "test"]
    for sp in splits:
        sets = RT.build_eval(sp, a.eval_n, facts)
        r = RT.eval_sets(model, sets, dev, amp_ctx)
        res[sp] = r
        res[sp + "_summary"] = RT.summarize(r)
    if dtype_dev == "cuda":
        res["gpu_mem_mb_peak"] = torch.cuda.max_memory_allocated() / 2 ** 20
        res["gpu_mem_mb_reserved_peak"] = torch.cuda.max_memory_reserved() / 2 ** 20
    with open(res_path, "w") as f:
        json.dump(res, f, indent=1)
    if not a.keep_ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    sm = res["test_summary"] if "test_summary" in res else res["val_summary"]
    print(f"[done] ({'test' if 'test_summary' in res else 'val'}) "
          + " | ".join(f"{t}: id {v['id']:.3f}" + (f" ood {v['ood']:.3f}" if v["ood"] is not None else "")
                       for t, v in sm.items() if v["id"] is not None) + f"\n       -> {res_path}")


if __name__ == "__main__":
    main()
