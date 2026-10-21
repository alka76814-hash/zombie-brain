#!/usr/bin/env python3
"""redo_compare.py -- ReDo-for-attention experiment: arm OFF (dormancy logged only) vs arm ON (dormant heads recycled).

    python redo_compare.py --off /kaggle/working/redo_off --on /kaggle/working/redo_on [--name zombie_memonly_b29]

Per seed: test accuracy (ID) of every reasoning task, the dormant-head fraction at the end of training, and the largest
dormant-head fraction seen during training.  Then the strict verdict (>=3 seeds, |gap| > 2x pooled std, same sign in every seed)
on the REASON score (mean of mqar, hop, add, state) and on each task that at least one run learned.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

TASKS = ["mqar", "hop", "add", "state"]


def load(d, name):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"{name}_s*", "result.json"))):
        s = int(re.search(r"_s(\d+)$", os.path.dirname(p)).group(1))
        r = json.load(open(p))
        dor = []
        lp = os.path.join(os.path.dirname(p), "log.jsonl")
        if os.path.exists(lp):
            for ln in open(lp):
                rec = json.loads(ln)
                if "dormancy" in rec:
                    dor.append((rec["step"], rec["dormancy"]["heads_frac"], rec["dormancy"]["chan_frac"]))
        out[s] = dict(acc={t: r["test_summary"][t]["id"] for t in TASKS if r["test_summary"].get(t)}, dor=dor)
    return out


def reason(r):
    xs = [r["acc"].get(t) for t in TASKS]
    return None if any(x is None for x in xs) else float(np.mean(xs))


def verdict(a, b):
    d = np.array(a) - np.array(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = float(np.sqrt((sa ** 2 + sb ** 2) / 2))
    if len(a) < 3:
        return "descriptive only (<3 seeds)"
    if min(a + b) >= 0.98:
        return "both at ceiling"
    if abs(d.mean()) > 2 * pooled and ((d > 0).all() or (d < 0).all()):
        return ("ON" if d.mean() > 0 else "OFF") + " better"
    return "indistinguishable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--name", default="zombie_memonly_b29")
    a = ap.parse_args()
    off, on = load(a.off, a.name), load(a.on, a.name)
    seeds = sorted(set(off) & set(on))
    if not seeds:
        raise SystemExit("no seed finished in both arms yet")
    print(f"model {a.name}   seeds {seeds}   (test split, in-distribution accuracy)")
    print(f"{'seed':>4s} {'arm':>4s} " + " ".join(f"{t:>7s}" for t in TASKS) + f" {'REASON':>7s} {'dormant heads end':>18s} {'max during':>11s}")
    for s in seeds:
        for arm, d in (("OFF", off), ("ON", on)):
            r = d[s]
            dor = r["dor"]
            end = f"{100 * dor[-1][1]:.1f}%" if dor else "n/a"
            mx = f"{100 * max(x[1] for x in dor):.1f}%" if dor else "n/a"
            rs = reason(r)
            print(f"{s:>4d} {arm:>4s} " + " ".join(f"{r['acc'].get(t, float('nan')):7.3f}" for t in TASKS)
                  + f" {rs if rs is not None else float('nan'):7.3f} {end:>18s} {mx:>11s}")
    print()
    print("ON minus OFF (positive = ReDo helps):")
    for t in TASKS + ["REASON"]:
        f = (lambda r: reason(r)) if t == "REASON" else (lambda r, t=t: r["acc"].get(t))
        xa = [f(on[s]) for s in seeds]
        xb = [f(off[s]) for s in seeds]
        if any(x is None for x in xa + xb):
            continue
        d = np.array(xa) - np.array(xb)
        best = max(xa + xb)
        v = "not learned by either arm" if (t != "REASON" and best < {"mqar": 0.5, "hop": 0.3, "add": 0.5, "state": 0.3}[t]) else verdict(xa, xb)
        print(f"   {t:7s} ON {np.mean(xa):.3f}  OFF {np.mean(xb):.3f}  gap {d.mean():+.3f} (per seed {', '.join(f'{x:+.3f}' for x in d)})  -> {v}")
    # does dormancy track failure?  (OFF arm: correlate end-of-training dormant fraction with REASON)
    pts = [(off[s]["dor"][-1][1], reason(off[s])) for s in seeds if off[s]["dor"] and reason(off[s]) is not None]
    if len(pts) >= 3:
        x, y = zip(*pts)
        if np.std(x) > 0 and np.std(y) > 0:
            print(f"\nOFF arm: correlation between dormant-head fraction and REASON score across seeds = {np.corrcoef(x, y)[0, 1]:+.2f} (n={len(pts)}; tiny sample, descriptive only)")


if __name__ == "__main__":
    main()
