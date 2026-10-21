#!/usr/bin/env python3
"""reason_run.py -- the full, fair protocol for the reasoning suite.  Resumable: re-run the same command after a Colab cut.

  1. TUNE   every architecture gets the SAME learning-rate grid on a separate tuning seed (100) and is scored on the
            VALIDATION split only (test is never touched).  lr_choice.json stores the winner per architecture.
  2. FINAL  each architecture trains at its chosen lr on seeds 0..S-1 and is scored on the held-out TEST split.
  3. python reason_aggregate.py --runs <out>/final

    python reason_run.py tune  --archs zombie_memonly_b29,moe --out /content/drive/MyDrive/zombie_reason
    python reason_run.py final --archs zombie_memonly_b29,moe --seeds 3 --out /content/drive/MyDrive/zombie_reason
    python reason_run.py status --out /content/drive/MyDrive/zombie_reason
    # extra reason_train.py args go after --,  e.g.   ... -- --steps 2000 --facts 4096
(the same extra args must be used for tune and final)

Cost on a T4 (about 29k tok/s, 8192 tok/step): 3000 steps ~ 14 min per run.  Default grid 3 lrs x 2 archs = 6 tune runs,
then 3 seeds x 2 archs = 6 final runs.  --tune-steps lets you tune on shorter runs (cheaper, slightly less faithful).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

TUNE_SEED = 100


def tune_dir(out, name, lr):
    return os.path.join(out, "tune", f"{name}_lr{lr:g}")


def final_dir(out, name, seed):
    return os.path.join(out, "final", f"{name}_s{seed}")


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def val_score(res):
    """tuning criterion: mean in-distribution VAL accuracy over all tasks (macro)"""
    ids = [v["id"] for v in res["val_summary"].values() if v["id"] is not None]
    return float(np.mean(ids))


def choose(out, archs, lrs):
    path = os.path.join(out, "lr_choice.json")
    choice = json.load(open(path)) if os.path.exists(path) else {}
    for n in archs:
        scores = {}
        for lr in lrs:
            p = os.path.join(tune_dir(out, n, lr), "result.json")
            if os.path.exists(p):
                scores[f"{lr:g}"] = val_score(json.load(open(p)))
        if len(scores) == len(lrs):
            best = max(scores, key=scores.get)
            edge = best in (f"{min(lrs):g}", f"{max(lrs):g}")
            choice[n] = dict(lr=float(best), val_scores=scores, at_grid_edge=edge)
            print(f"  {n}: " + "  ".join(f"lr {k}: {v:.3f}" for k, v in scores.items()) + f"  -> chose {best}"
                  + ("   (!! at grid edge: a better lr may lie outside the grid; extend --lrs)" if edge else ""))
    json.dump(choice, open(path, "w"), indent=1)
    return choice


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "final", "status"])
    ap.add_argument("--archs", default="zombie_memonly_b29,moe")
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tune-steps", type=int, default=None, help="default: same as the final --steps")
    ap.add_argument("--out", default="runs_reason")
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args(argv)
    archs = a.archs.split(",")
    lrs = [float(x) for x in a.lrs.split(",")]

    if a.mode == "status":
        for n in archs:
            t = [lr for lr in lrs if os.path.exists(os.path.join(tune_dir(a.out, n, lr), "result.json"))]
            f = [s for s in range(a.seeds) if os.path.exists(os.path.join(final_dir(a.out, n, s), "result.json"))]
            print(f"{n}: tune {len(t)}/{len(lrs)} done | final {len(f)}/{a.seeds} done")
        if os.path.exists(os.path.join(a.out, "lr_choice.json")):
            print(open(os.path.join(a.out, "lr_choice.json")).read())
        return

    if a.mode == "tune":
        jobs = [(n, lr) for lr in lrs for n in archs
                if not os.path.exists(os.path.join(tune_dir(a.out, n, lr), "result.json"))]
        print(f"tune: {len(jobs)} runs to do ({len(archs) * len(lrs)} total), seed {TUNE_SEED}, VAL split only\n")
        for n, lr in jobs:
            cmd = [sys.executable, "reason_train.py", "--name", n, "--seed", str(TUNE_SEED), "--lr", f"{lr:g}",
                   "--run-dir", tune_dir(a.out, n, lr), "--skip-test", "--eval-every", "0"]
            if a.tune_steps:
                cmd += ["--steps", str(a.tune_steps)]
            rc = run(cmd + extra)
            if rc != 0:
                print(f"!! {n} lr {lr:g} failed (exit {rc})")
                if not a.keep_going:
                    sys.exit(1)
        print("\nlr choice (by mean in-distribution VAL accuracy):")
        choose(a.out, archs, lrs)
        return

    # final
    cp = os.path.join(a.out, "lr_choice.json")
    choice = json.load(open(cp)) if os.path.exists(cp) else {}
    missing = [n for n in archs if n not in choice]
    if missing:
        raise SystemExit(f"no lr chosen for {missing}: run `reason_run.py tune` first (so every arch is tuned equally)")
    jobs = [(n, s) for s in range(a.seeds) for n in archs
            if not os.path.exists(os.path.join(final_dir(a.out, n, s), "result.json"))]
    print(f"final: {len(jobs)} runs to do ({len(archs) * a.seeds} total)\n")
    for n, s in jobs:
        lr = choice[n]["lr"]
        rc = run([sys.executable, "reason_train.py", "--name", n, "--seed", str(s), "--lr", f"{lr:g}",
                  "--run-dir", final_dir(a.out, n, s)] + extra)
        if rc != 0:
            print(f"!! {n} seed {s} failed (exit {rc})")
            if not a.keep_going:
                sys.exit(1)
    print("\nall done.  next:  python reason_aggregate.py --runs", os.path.join(a.out, "final"))


if __name__ == "__main__":
    main()
