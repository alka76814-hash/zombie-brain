#!/usr/bin/env python3
"""bench_cpu.py -- measured CPU speed of TRAINED models (run this on your i3-1215U, not on the Colab GPU box).

    python bench_cpu.py runs/gqa1x_s0 runs/gqa_s0 runs/zombie_s0 --threads 6
    python bench_cpu.py runs/zombie_s0 --route where        # compute-both-and-mask instead of true per-token routing
    python bench_cpu.py runs/zombie_s0 runs/gqa1x_s0 --int8 # dynamic int8 on all nn.Linear (same for both models)

Reports batch-1 decode tok/s (KV cache), prefill tok/s and peak RSS, best of --reps, in fp32 (fp16 is slow/unsupported
on many CPUs). Eager PyTorch only: on a model this small, eager decode is op-overhead-bound, so a bandwidth advantage
may NOT show up here -- that is a legitimate (negative) result to report, alongside the theoretical bytes from
counter.py. The cache gather (random-access rows from tables bigger than L3) is latency-bound; this measures it.
Close other programs; fix the thread count; run each model in a fresh process for clean RSS (default: it does).
"""
import argparse
import json
import subprocess
import sys

import torch
import torch.nn as nn


def one(run, a):
    import baseline
    import configs
    from zombie_v6 import ZombieLM
    torch.set_num_threads(a.threads)
    model, cfg, name = configs.load_run(run)
    model = model.float().eval()
    if isinstance(model, ZombieLM):
        model.zs.route = a.route
    if a.int8:
        from torch.ao.quantization import quantize_dynamic
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    r = baseline.bench(model, a.prompt, a.gen, batch=1, threads=a.threads, reps=a.reps)
    r.update(name=name, run=run, params_m=baseline.count_params(model) / 1e6 if not a.int8 else None,
             route=a.route if isinstance(model, ZombieLM) else None, int8=a.int8, threads=a.threads)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--route", default="subset", choices=["subset", "where"])
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--_child", action="store_true")
    a = ap.parse_args()
    if a._child:
        print("@@" + json.dumps(one(a.runs[0], a)))
        return
    rows = []
    for run in a.runs:
        cmd = [sys.executable, __file__, run, "--_child", "--threads", str(a.threads), "--prompt", str(a.prompt),
               "--gen", str(a.gen), "--reps", str(a.reps), "--route", a.route] + (["--int8"] if a.int8 else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("@@")]
        if not line:
            print(f"!! {run} failed:\n{out.stderr[-800:]}")
            continue
        rows.append(json.loads(line[0][2:]))
    print(f"\n{'model':22s} {'decode tok/s':>13s} {'prefill tok/s':>14s} {'peak RSS MB':>12s}   (threads {a.threads}, "
          f"{'int8' if a.int8 else 'fp32'}, prompt {a.prompt}, gen {a.gen}, batch 1)")
    for r in rows:
        print(f"{r['name']:22s} {r['decode_tok_s']:13.1f} {r['prefill_tok_s']:14.0f} {r['peak_rss_mb']:12.0f}")


if __name__ == "__main__":
    main()
