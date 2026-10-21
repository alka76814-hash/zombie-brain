#!/usr/bin/env python3
"""reason_aggregate.py -- tables, difficulty curves, paired comparisons and a plot for the reasoning suite.

    python reason_aggregate.py --runs /content/drive/MyDrive/zombie_reason/final [--plot curves.png]

Reading rules (printed again at the end):
  * Metric = instance-exact accuracy on the held-out TEST split. ID = difficulties seen in training, OOD = harder
    difficulties never seen in training. `facts` = recall of facts stored in the weights (train=test by design).
  * 'trivial' = best of {most common answer, task shortcuts, analytic chance}: a model must clear this to count as having
    learned the task. A task where NO model beats trivial+0.05 is reported as NOT LEARNED and never used to rank models.
  * Comparisons use seeds as the unit: gap must exceed 2x the pooled seed std AND have the same sign in every paired seed.
    With < 3 seeds nothing is claimed.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import reason_tasks as RT

PAIRS = [("zombie_memonly_b29", "moe"), ("zombie_memonly_L16", "moe_L16"), ("zombie_memonly", "moe"),
         ("zombie_memonly_b29", "gqa1x")]
ORDER = ["mha", "gqa", "gqa1x", "moe", "zombie_memonly", "zombie_memonly_b29", "moe_L16", "zombie_memonly_L16"]


def load(runs):
    by = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(runs, "*", "result.json"))):
        r = json.load(open(p))
        if "test" in r:
            by[r["name"]][r["seed"]] = r
    return by


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)


def fmt(m, s, w=5):
    return " " * w + "-" if m is None else (f"{m:.3f}±{s:.3f}" if s is not None else f"{m:.3f}")


def metric(r, task, kind):
    v = r["test_summary"].get(task)
    return None if v is None else v[kind]


def macro(r, kind):
    xs = [metric(r, t, kind) for t in RT.REASONING]
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if len(xs) == len(RT.REASONING) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--pair", action="append", default=[], help="A,B  (extra comparison)")
    a = ap.parse_args()
    by = load(a.runs)
    if not by:
        raise SystemExit("no finished runs with a test split found")
    names = [n for n in ORDER if n in by] + [n for n in by if n not in ORDER]
    any_run = next(iter(next(iter(by.values())).values()))
    facts = RT.make_fact_table(any_run["facts"]) if any_run["facts"] else None
    base = RT.baseline_table("test", any_run["eval_n"], facts)
    act = [t for t in RT.TASKS if t != "facts" or facts]          # runs trained without the facts task simply have no facts row

    print("=" * 100)
    print("MODELS")
    for n in names:
        rs = by[n]
        r0 = next(iter(rs.values()))
        print(f"  {n:20s} params {r0['params_total'] / 1e6:6.2f}M  lr {r0['lr']:g}  seeds {sorted(rs)}  "
              f"steps {r0['steps']}  {r0['train_tok_s'] / 1e3:.1f}k tok/s")

    # ---------------- which tasks were learned at all: some model must beat the trivial baseline by >0.05 on average
    # over the in-distribution cells (cells with n < MIN_N ignored; cells a shortcut already solves count 0 at best)
    learned = {}
    for t in act:
        cells = [d for d in RT.TASKS[t]["train"] if d in RT.TASKS[t]["test"] and base[(t, d)]["n"] >= RT.MIN_N]
        best = -1.0
        for n in names:
            for r in by[n].values():
                ex = [r["test"][f"{t}|{d}"]["acc"] - base[(t, d)]["trivial"] for d in cells]
                if ex:
                    best = max(best, float(np.mean(ex)))
        learned[t] = best > 0.05
        learned[t + "_info"] = best

    print("=" * 100)
    print("PER-TASK ACCURACY (test, instance-exact; mean±std over seeds).  ID = trained difficulties, OOD = harder, unseen")
    hdr = f"{'task':8s} {'':4s}" + "".join(f"{n[:19]:>21s}" for n in names)
    print(hdr)
    for t in act:
        for kind in ("id", "ood"):
            vals = [ms([metric(r, t, kind) for r in by[n].values()]) for n in names]
            if all(v[0] is None for v in vals):
                continue
            print(f"{t:8s} {kind.upper():4s}" + "".join(f"{fmt(*v):>21s}" for v in vals)
                  + ("" if (learned[t] or kind == 'ood') else "   <- NOT LEARNED by any model"))
    vals = [ms([macro(r, "id") for r in by[n].values()]) for n in names]
    print(f"{'REASON':8s} {'ID':4s}" + "".join(f"{fmt(*v):>21s}" for v in vals) + "   (mean of mqar,hop,add,state)")
    vals = [ms([macro(r, "ood") for r in by[n].values()]) for n in names]
    print(f"{'REASON':8s} {'OOD':4s}" + "".join(f"{fmt(*v):>21s}" for v in vals))

    print("=" * 100)
    print("FRONTIER: hardest difficulty solved with acc>=0.9 / >=0.5 walking up from the easiest (median over seeds)")
    for t in RT.REASONING:
        line = f"{t:8s}"
        for n in names:
            f9 = [r["test_summary"][t]["frontier"]["0.9"] for r in by[n].values()]
            f5 = [r["test_summary"][t]["frontier"]["0.5"] for r in by[n].values()]
            md = lambda xs: "none" if all(x is None for x in xs) else str(int(np.median([x if x is not None else 0 for x in xs])))
            line += f"  {n[:14]}: {md(f9):>4s}/{md(f5):<4s}"
        print(line)

    print("=" * 100)
    print("DIFFICULTY CURVES (test).  trivial = best cheat baseline;  * = OOD difficulty;  n = test instances")
    for t in act:
        print(f"-- {t}")
        print(f"   {'diff':>5s} {'n':>5s} {'trivial':>8s}" + "".join(f"{n[:16]:>18s}" for n in names))
        for d in RT.TASKS[t]["test"]:
            b = base[(t, d)]
            row = f"   {d:>4d}{'*' if RT.is_ood(t, d) else ' '} {b['n']:>5d} {b['trivial']:>8.3f}"
            for n in names:
                xs = [r["test"][f"{t}|{d}"]["acc"] for r in by[n].values() if f"{t}|{d}" in r["test"]]
                m, s = ms(xs)
                row += f"{fmt(m, s):>18s}"
            print(row)


    # ---------------- learning SPEED: first mid-run val checkpoint (logged every 500 steps) with accuracy >= 0.9.
    # Final accuracy saturates (facts) or depends on when a sudden jump happens (add), so "how early" is the fair measure.
    def speed(run_dir, cell):
        p = os.path.join(run_dir, "log.jsonl")
        if not os.path.exists(p):
            return "?"
        last = None
        for ln in open(p):
            rec = json.loads(ln)
            if "val" in rec:
                last = rec["step"]
                if rec["val"].get(cell, 0) >= 0.9:
                    return str(rec["step"])
        return f">{last}" if last else "?"
    print("=" * 100)
    print("LEARNING SPEED: first checkpoint (every 500 steps, 100 val instances) with accuracy >= 0.9;  '>N' = not by step N")
    for label, cell in ((("facts", "facts|0"),) if facts else ()) + (("add d=3", "add|3"),):
        print(f"  {label}")
        for n in names:
            sp = [f"s{s}:{speed(os.path.join(a.runs, f'{n}_s{s}'), cell)}" for s in sorted(by[n])]
            print(f"     {n:20s} " + "  ".join(sp))

    # ---------------- paired comparisons
    print("=" * 100)
    print("PAIRED COMPARISONS  (A - B, positive = A better;  verdict needs >=3 seeds, |gap| > 2x pooled std, same sign in every seed)")
    pairs = PAIRS + [tuple(p.split(",")) for p in a.pair]
    kinds = [(t, "id") for t in act] + [(t, "ood") for t in RT.REASONING] + [("REASON", "id"), ("REASON", "ood")]
    for A, B in pairs:
        if A not in by or B not in by:
            continue
        seeds = sorted(set(by[A]) & set(by[B]))
        pa, pb = by[A][seeds[0]]["params_total"], by[B][seeds[0]]["params_total"]
        print(f"\n  {A} ({pa / 1e6:.2f}M)  vs  {B} ({pb / 1e6:.2f}M)   paired seeds {seeds}")
        for t, kind in kinds:
            get = (lambda r: macro(r, kind)) if t == "REASON" else (lambda r, t=t: metric(r, t, kind))
            xa = [get(by[A][s]) for s in seeds]
            xb = [get(by[B][s]) for s in seeds]
            if any(x is None for x in xa + xb):
                continue
            d = np.array(xa) - np.array(xb)
            ma, sa = ms(xa)
            mb, sb = ms(xb)
            pooled = float(np.sqrt((sa ** 2 + sb ** 2) / 2))
            gap = float(d.mean())
            if (t != "REASON" and not learned[t]) or (t == "REASON" and not any(learned[x] for x in RT.REASONING)):
                verdict = "n/a (task not learned by any model)"
            elif len(seeds) < 3:
                verdict = "descriptive only (<3 seeds)"
            elif max(xa + xb) >= 0.98 and min(xa + xb) >= 0.98:
                verdict = "both at ceiling"
            elif abs(gap) > 2 * pooled and (d > 0).all() or abs(gap) > 2 * pooled and (d < 0).all():
                verdict = f"{A if gap > 0 else B} better"
            else:
                verdict = "indistinguishable"
            print(f"    PAIR {t:7s} {kind.upper():3s} {ma:.3f}±{sa:.3f} vs {mb:.3f}±{sb:.3f}  gap {gap:+.3f} "
                  f"(per-seed {', '.join(f'{x:+.3f}' for x in d)})  -> {verdict}")

    print("=" * 100)
    print("TASKS LEARNED AT ALL (best single run: mean ID accuracy minus trivial baseline, must exceed 0.05):")
    for t in act:
        print(f"   {t:6s} excess over trivial {learned[t + '_info']:+.3f}  -> "
              f"{'learned' if learned[t] else 'NOT LEARNED (no ranking possible)'}")
    print(f"(cells with n < {RT.MIN_N} held-out instances are shown in the curves but excluded from every summary number)")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print("(no matplotlib:", e, ")")
            return
        fig, axs = plt.subplots(1, len(RT.TASKS) - 1, figsize=(4.2 * (len(RT.TASKS) - 1), 3.6), squeeze=False)
        for ax, t in zip(axs[0], RT.REASONING):
            ds = [d for d in RT.TASKS[t]["test"] if base[(t, d)]["n"] >= RT.MIN_N]       # tiny cells are not drawn
            ax.plot(ds, [base[(t, d)]["trivial"] for d in ds], "k:", label="trivial")
            first_ood = min((i for i, d in enumerate(ds) if RT.is_ood(t, d)), default=None)
            if first_ood is not None:
                ax.axvspan(ds[first_ood] - 0.5 * (ds[first_ood] - ds[first_ood - 1]), ds[-1] * 1.02, color="0.9", zorder=0)
            for n in names:
                M = np.array([[r["test"][f"{t}|{d}"]["acc"] for d in ds] for r in by[n].values()])
                m, s = M.mean(0), (M.std(0, ddof=1) if len(M) > 1 else np.zeros(len(ds)))
                ax.plot(ds, m, marker="o", ms=3, label=n)
                ax.fill_between(ds, m - s, m + s, alpha=0.15)
            ax.set_title(t + " (grey = unseen difficulty)")
            ax.set_xlabel("difficulty")
            ax.set_ylim(-0.02, 1.02)
            if t in ("mqar", "state"):
                ax.set_xscale("log")
                ax.set_xticks(ds)
                ax.set_xticklabels([str(d) for d in ds], fontsize=7)
                ax.minorticks_off()
        axs[0][0].set_ylabel("instance-exact acc")
        axs[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print("plot ->", a.plot)


if __name__ == "__main__":
    main()
