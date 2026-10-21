#!/usr/bin/env python3
"""aggregate.py -- runs/*/result.json -> results.md, results.csv, pareto.png  (mean +- std over seeds)

    python aggregate.py --runs /content/drive/MyDrive/zombie_runs

Reading the table:
  * val/train ppl     eval-mode, hard routing for Zombie, FIXED windows (identical for all models)
  * MACs / MB read    closed-form (counter.py) at the MEASURED routing fraction p_B, fp16 weights, batch-1 decode.
                      Includes LM head; attention scores excluded. These are theoretical; measure tok/s with bench_cpu.py.
  * dead% (GELU)      different metric from slot entropy -- never compare the two directly
  * slotH             normalised slot-usage entropy of the memory (1.0 = perfectly uniform use of all 16384 slots)
  * KV acc            exact recall of synthetic facts (chance ~1e-4)
  * PTQ w8 / w8c4     val ppl after per-row int8 weights (+ int4 cache values for Zombie), fake-quantised
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

ORDER = ["mha", "gqa", "gqa1x", "moe", "attn_only", "zombie", "zombie_mem6_11", "zombie_memonly", "zombie_memonly_b29", "zombie_memonly_L16", "moe_L16", "zombie_fixed_half",
         "zombie_nogk", "zombie_noinit", "zombie_gt03", "zombie_gt07", "zombie_stress", "zombie_stress_nogk",
         "gqa1x_nrev", "zombie_nrev"]


def get(r, *path):
    for p in path:
        if r is None or not isinstance(r, dict) or p not in r:
            return None
        r = r[p]
    return r


def ms(vals, fmt="{:.2f}", pm=True):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = st.mean(vals)
    if len(vals) > 1 and pm:
        return (fmt + " ± " + fmt).format(m, st.stdev(vals))
    return fmt.format(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None, help="where to write results.* (default: --runs)")
    a = ap.parse_args()
    out = a.out or a.runs
    groups = defaultdict(list)
    for p in glob.glob(os.path.join(a.runs, "*", "result.json")):
        with open(p) as f:
            r = json.load(f)
        groups[r["name"]].append(r)
    if not groups:
        raise SystemExit(f"no result.json under {a.runs}")
    names = [n for n in ORDER if n in groups] + sorted(n for n in groups if n not in ORDER)

    cols = ["config", "seeds", "total params", "active core", "MACs/tok", "MB read/tok", "val ppl", "test ppl", "train ppl",
            "dead % (GELU)", "FFN revived", "p_B", "slotH", "dead rows", "expert H", "dead experts", "KV acc", "PTQ w8", "PTQ w8c4", "train tok/s"]
    rows, csv_rows = [], []
    for n in names:
        R = groups[n]
        slotH = [sum(c["slot_entropy_norm"] for c in r["routing"]["caches"]) / len(r["routing"]["caches"])
                 if r.get("routing") else None for r in R]
        deadrows = [sum(c["dead_rows"] for c in r["routing"]["caches"]) if r.get("routing") else None for r in R]
        row = [n, str(len(R)),
               ms([r["params_total"] / 1e6 for r in R], "{:.2f}M", False),
               ms([get(r, "costs_fp16", "active_core") / 1e6 for r in R], "{:.2f}M", False),
               ms([get(r, "costs_fp16", "macs") / 1e6 for r in R], "{:.2f}M", False),
               ms([get(r, "costs_fp16", "mb_read") for r in R], "{:.1f}", False),
               ms([get(r, "val", "ppl") for r in R], "{:.3f}"),
               ms([get(r, "test", "ppl") for r in R], "{:.3f}"),
               ms([get(r, "train_eval", "ppl") for r in R], "{:.3f}"),
               ms([get(r, "dead_neurons", "pct_dead") for r in R], "{:.1f}"),
               ms([get(r, "neuron_revival", "revived") if get(r, "neuron_revival", "enabled") else None for r in R], "{:.0f}", False),
               ms([get(r, "routing", "p_b_mean") for r in R], "{:.2f}"),
               ms(slotH, "{:.3f}"),
               ms(deadrows, "{:.0f}", False),
               ms([get(r, "moe_routing", "entropy_norm_mean") for r in R], "{:.3f}"),
               ms([get(r, "moe_routing", "dead_experts") for r in R], "{:.0f}", False),
               ms([get(r, "kv_recall", "acc") for r in R], "{:.3f}"),
               ms([get(r, "ptq_ppl", "w8") for r in R], "{:.3f}"),
               ms([get(r, "ptq_ppl", "w8_c4") for r in R], "{:.3f}"),
               ms([r.get("train_tok_s") for r in R], "{:.0f}", False)]
        rows.append(row)
        csv_rows.append(dict(config=n, seeds=len(R), val_ppl_mean=st.mean(get(r, "val", "ppl") for r in R),
                             val_ppl_std=st.stdev(get(r, "val", "ppl") for r in R) if len(R) > 1 else 0.0,
                             macs=get(R[0], "costs_fp16", "macs"), mb_read=get(R[0], "costs_fp16", "mb_read"),
                             params_total=R[0]["params_total"]))
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    print(md)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    # headline reading aid
    def mean_ppl(n):
        return st.mean(get(r, "val", "ppl") for r in groups[n]) if n in groups else None
    if "zombie" in groups and "moe" in groups:
        zs_ = [get(r, "val", "ppl") for r in groups["zombie"]]
        ms_ = [get(r, "val", "ppl") for r in groups["moe"]]
        z, mo = st.mean(zs_), st.mean(ms_)
        line = f"\nMATCHED COMPARISON: zombie val ppl {z:.3f} vs moe {mo:.3f}  ({100 * (z - mo) / mo:+.1f}%)"
        if min(len(zs_), len(ms_)) < 2:
            line += "  -> single seed: NOT conclusive"
        else:
            noise = max(st.stdev(zs_), st.stdev(ms_))
            line += (f"  -> within 2x seed std ({noise:.3f}): INDISTINGUISHABLE" if abs(z - mo) <= 2 * noise
                     else f"  -> {'zombie' if z < mo else 'moe'} better (gap > 2x seed std {noise:.3f})")
        print(line)
    if "zombie" in groups and "gqa1x" in groups:
        z, c = mean_ppl("zombie"), mean_ppl("gqa1x")
        zs = [get(r, "val", "ppl") for r in groups["zombie"]]
        cs = [get(r, "val", "ppl") for r in groups["gqa1x"]]
        msg = f"\nHEADLINE CONTROL: zombie val ppl {z:.3f} vs gqa1x {c:.3f}  ({100 * (z - c) / c:+.1f}%)"
        if min(len(zs), len(cs)) < 2:
            msg += "  -> single seed: NOT conclusive, run --suite headline --seeds 3"
        else:
            noise = max(st.stdev(zs), st.stdev(cs))
            if abs(z - c) <= 2 * noise:
                msg += f"  -> within 2x seed std ({noise:.3f}): memory and plain 1x FFN are INDISTINGUISHABLE"
            else:
                msg += f"  -> memory {'HELPS' if z < c else 'HURTS'} at ~equal cost (gap > 2x seed std {noise:.3f})"
        print(msg)
    # budget-matched pairs (same total params): memory-only vs MoE, with a bigger slot budget or more attention depth
    for a_, b_ in [("zombie_memonly", "moe"), ("zombie_memonly_b29", "moe"), ("zombie_memonly_L16", "moe_L16")]:
        if a_ in groups and b_ in groups:
            xs = [get(r, "val", "ppl") for r in groups[a_]]
            ys = [get(r, "val", "ppl") for r in groups[b_]]
            x, y = st.mean(xs), st.mean(ys)
            line = f"\nPAIR {a_} vs {b_}: val ppl {x:.3f} vs {y:.3f}  ({100 * (x - y) / y:+.1f}%)"
            if min(len(xs), len(ys)) < 2:
                line += "  -> single seed on at least one side: NOT conclusive"
            else:
                noise = max(st.stdev(xs), st.stdev(ys))
                line += (f"  -> within 2x seed std ({noise:.3f}): INDISTINGUISHABLE" if abs(x - y) <= 2 * noise
                         else f"  -> {a_ if x < y else b_} better (gap > 2x seed std {noise:.3f})")
            print(line)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for j, (key, lab) in enumerate([("mb_read", "MB read per decoded token (fp16, theoretical)"),
                                        ("macs", "MACs per token")]):
            for r in csv_rows:
                x = r[key] if key == "mb_read" else r[key] / 1e6
                z = r["config"].startswith("zombie")
                ax[j].errorbar(x, r["val_ppl_mean"], yerr=r["val_ppl_std"], fmt="o" if z else "s", capsize=3,
                               color="#c0392b" if z else "#2c3e50")
                ax[j].annotate(r["config"], (x, r["val_ppl_mean"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
            ax[j].set_xlabel(lab + ("" if key == "mb_read" else " (M)"))
            ax[j].set_ylabel("val perplexity (hard routing)")
            ax[j].grid(alpha=.3)
        fig.suptitle("Pareto: quality vs cost  (red = Zombie variants, dark = dense baselines)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "pareto.png"), dpi=140)
        print(f"wrote {out}/results.md, results.csv, pareto.png")
    except Exception as e:
        print(f"(no plot: {e})")


if __name__ == "__main__":
    main()
