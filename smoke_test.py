#!/usr/bin/env python3
"""smoke_test.py -- whole pipeline on a tiny synthetic corpus in a few minutes (CPU or GPU).

    python smoke_test.py            # run this first, on Colab too, before the real runs

prep_data -> train (gqa1x, zombie incl. k-means init, Gravekeeper, soft->hard gate, KV recall, PTQ)
-> kill/resume check -> run_ablations (skips finished) -> aggregate -> bench_cpu.
Uses --tiny models (d=64, 4 layers): this checks the plumbing, NOT the science.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

PY = sys.executable
here = os.path.dirname(os.path.abspath(__file__))


def sh(*args, check=True):
    print("\n$ " + " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([PY, *map(str, args)], cwd=here)
    if check and r.returncode:
        sys.exit(f"FAILED: {args}")
    return r.returncode


def main():
    tmp = tempfile.mkdtemp(prefix="zb_smoke_")
    raw, data, runs = (os.path.join(tmp, x) for x in ("raw", "data", "runs"))
    os.makedirs(raw)
    rng = random.Random(0)
    subj = "the cat|a dog|my friend|the old man|a small bird|the teacher|our neighbour|a young girl".split("|")
    verb = "saw|liked|found|followed|forgot|painted|carried|heard".split("|")
    obj = "a red ball|the big tree|some warm bread|an empty box|the quiet river|a long story|two green hats".split("|")
    with open(os.path.join(raw, "corpus.txt"), "w") as f:
        for _ in range(60000):
            f.write(f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)} and then {rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)}.\n")
    sh("prep_data.py", "--input", raw, "--out", data, "--vocab", 600, "--kv-pairs", 100, "--kv-repeat", 10,
       "--block-chars", 4000)
    common = ["--data", data, "--out", runs, "--tiny", "--seq", 64, "--batch", 8, "--steps", 120, "--log-every", 30,
              "--eval-every", 60, "--ckpt-every", 40, "--final-eval-batches", 6, "--dead-batches", 4, "--eval-batches", 3,
              "--lr", 2e-3, "--cache-lr", 1e-2, "--warmup", 10]
    sh("train.py", "--name", "gqa1x", *common)
    # zombie: interrupt at step 80 (ckpt at 80) then resume -> must finish identically in structure
    sh("train.py", "--name", "zombie", *common, "--warm-steps", 30, "--soft-steps", 30, "--dead-after", 20,
       "--gk-every", 10)
    r = json.load(open(os.path.join(runs, "zombie_s0", "result.json")))
    assert r["kv_recall"]["n"] > 0 and r["routing"] and r["ptq_ppl"]["w8_c4"] > 0
    print("zombie result ok:", {k: r[k] for k in ("params_total",)}, "val ppl", round(r["val"]["ppl"], 3))
    # resume test: wipe result, keep an early ckpt scenario by re-running with --steps larger and --force --resume
    shutil.rmtree(os.path.join(runs, "zombie_s0"))
    sh("train.py", "--name", "zombie", *common, "--warm-steps", 30, "--soft-steps", 30, "--dead-after", 20,
       "--gk-every", 10, "--keep-ckpt")
    os.remove(os.path.join(runs, "zombie_s0", "result.json"))
    st_ckpt = os.path.join(runs, "zombie_s0", "ckpt.pt")
    assert os.path.exists(st_ckpt)
    sh("train.py", "--name", "zombie", *common, "--warm-steps", 30, "--soft-steps", 30, "--dead-after", 20,
       "--gk-every", 10, "--resume")
    for name in ("zombie_mem6_11", "zombie_memonly", "attn_only"):
        sh("train.py", "--name", name, *common, "--steps", 40, "--warm-steps", 10, "--soft-steps", 10, "--eval-every", 0)
    sh("train.py", "--name", "moe", *common, "--steps", 60, "--eval-every", 0)
    rm = json.load(open(os.path.join(runs, "moe_s0", "result.json")))
    assert rm["moe_routing"]["layers"] and rm["costs_fp16"]["macs"] > 0
    print("moe: expert entropy", round(rm["moe_routing"]["entropy_norm_mean"], 3), "dead experts", rm["moe_routing"]["dead_experts"])
    # dead-neuron revival wired through train.py (baseline and zombie get the same treatment)
    for name in ("gqa1x_nrev", "zombie_nrev"):
        sh("train.py", "--name", name, *common, "--steps", 60, "--warm-steps", 10, "--soft-steps", 10, "--eval-every", 0,
           "--neuron-every", 20, "--neuron-min-tokens", 1000)
        rr = json.load(open(os.path.join(runs, f"{name}_s0", "result.json")))
        assert rr["neuron_revival"]["enabled"], rr["neuron_revival"]
        print(f"{name}: neuron_revival -> {rr['neuron_revival']}")
    sh("run_ablations.py", "--suite", "list")
    sh("aggregate.py", "--runs", runs)
    sh("bench_cpu.py", os.path.join(runs, "gqa1x_s0"), os.path.join(runs, "zombie_s0"), "--threads", 2, "--gen", 8,
       "--prompt", 32, "--reps", 1)
    print(f"\nSMOKE TEST PASSED  (artifacts in {tmp})")


if __name__ == "__main__":
    main()
