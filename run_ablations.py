#!/usr/bin/env python3
"""run_ablations.py -- run many configs x seeds, sequentially, resumable.

    python run_ablations.py --suite list
    python run_ablations.py --suite sweep    --out /content/drive/MyDrive/zombie_runs      # all 14 configs, seed 0
    python run_ablations.py --suite headline --seeds 3 --out ...                            # mha,gqa,gqa1x,zombie x seeds 0..2
    python run_ablations.py --suite full     --seeds 3 --out ...                            # sweep first, then extra headline seeds
    python run_ablations.py --suite zombie,gqa1x --seeds 1 --out ... -- --steps 3000      # custom list, extra train.py args after --

Plan from the notes: sweep everything with ONE seed first, then spend 3 seeds only on the headline four.
Finished runs (result.json) are skipped; interrupted runs continue from ckpt.pt (--resume is always passed).
If a Colab session dies, re-run the same command.
"""
import argparse
import os
import subprocess
import sys

import configs


def plan(suite, seeds):
    if suite == "list":
        return []
    if suite == "sweep":
        return [(n, 0) for n in configs.ALL]
    if suite == "headline":
        return [(n, s) for s in range(seeds) for n in configs.HEADLINE]
    if suite == "full":
        return [(n, 0) for n in configs.ALL] + [(n, s) for s in range(1, seeds) for n in configs.HEADLINE]
    names = suite.split(",")
    bad = [n for n in names if n not in configs.ALL + configs.EXTRA]
    if bad:
        raise SystemExit(f"unknown configs {bad}; choices {configs.ALL + configs.EXTRA}")
    return [(n, s) for s in range(seeds) for n in names]


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="sweep")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    a = ap.parse_args(argv)
    if a.suite == "list":
        print("headline:", configs.HEADLINE, "\nablation:", configs.ABLATION)
        return
    jobs = plan(a.suite, a.seeds)
    todo = [(n, s) for n, s in jobs if not os.path.exists(os.path.join(a.out, f"{n}_s{s}", "result.json"))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run\n")
    failed = []
    for i, (n, s) in enumerate(todo):
        print(f"===== [{i + 1}/{len(todo)}] {n} seed {s} =====", flush=True)
        cmd = [sys.executable, "train.py", "--name", n, "--seed", str(s), "--data", a.data, "--out", a.out,
               "--resume"] + extra
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append((n, s))
            print(f"!! {n} seed {s} failed (exit {rc})", flush=True)
            if not a.keep_going:
                break
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("all done. next:  python aggregate.py --runs", a.out)


if __name__ == "__main__":
    main()
