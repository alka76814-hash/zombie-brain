#!/usr/bin/env python3
"""reason_parallel.py -- run many reason_train.py jobs at once, ONE JOB PER GPU.

Counts the GPUs (torch.cuda.device_count(); 0 or 1 -> jobs run one after another) and keeps every GPU busy: each worker
pins itself to one GPU with CUDA_VISIBLE_DEVICES, so the training code itself is unchanged (single-GPU, same numerics,
same exact-resume behaviour).  Jobs whose result.json exists are skipped, so re-running the same command resumes.

    python reason_parallel.py --archs zombie_memonly_b29,moe --seeds 3 --out /kaggle/working/diag3 \
        -- --lr 1e-3 --steps 3000 --facts 0 --mix mqar=.4,hop=.1,state=.1,induct=.4,add=0 \
           --diffs mqar=2:3,hop=1:2,state=2:3 --skip-test --eval-every 250
(everything after `--` is passed to reason_train.py unchanged; --name/--seed/--run-dir are set per job)
"""
import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="zombie_memonly_b29,moe")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=None, help="default: all visible GPUs")
    a = ap.parse_args(argv)

    g = a.gpus if a.gpus is not None else n_gpus()
    workers = max(1, g)
    jobs = [(n, s) for s in range(a.seed_start, a.seed_start + a.seeds) for n in a.archs.split(",")
            if not os.path.exists(os.path.join(a.out, f"{n}_s{s}", "result.json"))]
    print(f"{g} GPU(s) -> {workers} parallel worker(s); {len(jobs)} job(s) to run", flush=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    failed = []

    def work(w):
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if g > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(w)
        while True:
            try:
                n, s = q.get_nowait()
            except queue.Empty:
                return
            rd = os.path.join(a.out, f"{n}_s{s}")
            os.makedirs(rd, exist_ok=True)
            log = os.path.join(rd, "stdout.log")
            cmd = [sys.executable, "reason_train.py", "--name", n, "--seed", str(s), "--run-dir", rd] + extra
            print(f"[gpu {w}] start {n} seed {s}", flush=True)
            with open(log, "a") as f:
                rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            tail = ""
            try:
                tail = [l for l in open(log).read().splitlines() if l.startswith("[done]")][-1]
            except Exception:
                pass
            print(f"[gpu {w}] {'done' if rc == 0 else 'FAILED'} {n} seed {s}  {tail}", flush=True)
            if rc != 0:
                failed.append((n, s))

    ts = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("\nfinished.  failed:", failed or "none")
    print("logs:  <out>/<arch>_s<seed>/stdout.log   (see [val] lines with:  !grep -h '\\[val\\]' <out>/*/stdout.log)")


if __name__ == "__main__":
    main()
