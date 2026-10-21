# Zombie-Brain v6 — attention-only core + shared product-key memory + gate + self-healing

Everything needed to train and compare Zombie-Brain against standard baselines on your own text dataset (built for a
~200 MB corpus on a Colab T4), then reproduce the paper tables. By XTErion.

## Architecture (what `zombie_v6.py` implements)

Decoder-only, d=384, 12 layers, 8 Q heads / 2 KV heads (GQA, head dim 48), RoPE, RMSNorm, untied 8k embedding/head.
In every Zombie layer the FFN is replaced by:

```
out = g · B(x) + (1−g) · A(x)
B(x) = GELU FFN, 384→384→384                     (compute path, "Branch B")
A(x) = scale ⊙ PKM(BatchNorm(Wq x))              (memory path)
g    = sigmoid(384→32→1) per token               (gate)
```
* **PKM** = product-key memory: 2 heads, each splits its query into two 96-d halves, scores 128 sub-keys per half,
  takes the top-4 of the 128×128 = 16 384 slots, softmax-weights 4 value rows. One value table (16384×384) per cache,
  **two caches** (layers 0-5 share A, 6-11 share B). A learned 384-vector rescales the shared values per layer.
* **Gate phases:** `shadow` (first ~16% of steps: g=1, memory trained by `MSE(A, B.detach())`) → k-means init of
  sub-keys/values from Branch-B *outputs* → `soft` (next ~33%: g=sigmoid) → `hard` (straight-through hard gate; at
  inference true per-token routing, only the chosen path is computed). Loss aux = λ|mean g − target| − λ_b·mean|g−½|.
* **Optimizers:** AdamW for everything dense, `SparseAdam` for cache values (sparse grads only touch selected rows).
* **Gravekeeper** (no gradient, every 100 steps): the unit of death is a **sub-key row** that was in no top-k for 500
  steps. Revive = key ← a recent query-half of a poorly-served token (rescaled to live-key norm); its slots' values ←
  buffered Branch-B outputs + noise (only slots dead for *every* head); Adam moments of those rows zeroed.
* int4 is **post-training** fake-quant only (per-row absmax, +1 fp16 scale/row in practice) — no int4 kernels, so no
  speed claim comes from int4.

Configs (`configs.py`): headline `mha`, `gqa`, `gqa1x`, `moe`, `zombie`; ablations `attn_only`, `zombie_mem6_11` (dense FFN in
layers 0-5), `zombie_memonly` (no gate/B), `zombie_fixed_half`, `zombie_nogk`, `zombie_noinit`, `zombie_gt03/gt07`
(gate-target sweep), `zombie_stress` / `zombie_stress_nogk` (no QueryNorm + key lr 1e-3, to force collapse so
Gravekeeper must fire), `gqa1x_nrev` / `zombie_nrev` (same models + dead-FFN-neuron revival, below). **`gqa1x` (GQA + 1× FFN) is the control that matters**: Zombie only wins if memory buys lower ppl
than a plain 1× FFN at ~equal MACs/bytes.

## The MoE baseline (`moe`, `moe.py`)
A plain top-2 Mixture-of-Experts transformer: same GQA attention/RoPE/embeddings, every FFN replaced by a router + 12 small
GELU experts (hidden 164), Mixtral-style softmax over the selected logits, Switch-style load-balance loss (weight 0.01),
no token dropping. Sized so that **on paper it matches Zombie v6** (`python counter.py`, batch-1 decode, gate at 50%):

| | total params | active core | MACs/token | MB read/token (fp16) |
|---|---|---|---|---|
| Zombie v6 | 28.88M | 7.55M | 10.68M | 21.40 |
| MoE top-2 (matched) | 28.92M | 7.50M | 10.65M | 21.32 |

So the question is: same capacity, same active compute — does learned memory + gate + self-healing beat routed experts?
Caveats: MoE reads only 2 experts at batch-1 but most experts as batch/prefill grows; the router can collapse (expert load
entropy and dead experts are logged per 50 steps and reported as `expert H` / `dead experts`); `aggregate.py` prints a
MATCHED COMPARISON line (zombie vs moe) and refuses to call a winner inside 2x seed std.

## Quickstart (Colab T4)

```bash
pip install tokenizers matplotlib            # torch is preinstalled on Colab
python smoke_test.py                         # ~minutes, tiny models: checks the plumbing end to end. Run it first.
python prep_data.py --input /content/mydata --out data --vocab 8192 --kv-pairs 5000      # one pool, random 10% val
# OR, if you already have train.txt / validation.txt / test.txt (use these, don't use --input on that folder):
python prep_data.py --train /content/mydata/train.txt --val /content/mydata/validation.txt \
                    --test /content/mydata/test.txt --out data --vocab 8192 --kv-pairs 5000
python train.py --name zombie --steps 50 --out /tmp/t --eval-every 0 --no-ptq     # time it: read the tok/s lines
python run_ablations.py --suite sweep    --out /content/drive/MyDrive/zombie_runs  # 17 configs, 1 seed each
python run_ablations.py --suite headline --seeds 3 --out /content/drive/MyDrive/zombie_runs   # then 3 seeds on the 5 headline
python aggregate.py --runs /content/drive/MyDrive/zombie_runs                     # results.md / .csv / pareto.png
```
`colab_quickstart.ipynb` has the same as cells. **Write runs to Google Drive**; a disconnect then costs at most
`--ckpt-every` (500) steps — just re-run the same command, finished runs are skipped and partial ones resume exactly
(verified in the smoke test: a run killed and resumed from step 80 gave the same val/train ppl as the uninterrupted
run, to 3 decimals).

* **Your own splits:** `--train/--val/--test` are used exactly as given (never re-split; the tokenizer sees train only;
  test is scored once at the end and reported as `test ppl`). It prints a leakage check (% of val/test lines that also
  occur verbatim in train): above ~5% remove duplicates, otherwise ppl looks too good. Pick models by **val**, report **test**.
* **Data:** `prep_data.py` takes .txt/.md (or .jsonl with `--jsonl-field`), cuts 64k-char blocks at newlines, assigns a
  random 10% of blocks to val (seeded), trains the 8k byte-level BPE on the *train blocks only*, writes uint16 bins.
  `--kv-pairs N` adds N synthetic facts (`qzxkf => 4821`) ×20 to **train only** and writes `kv_test.json`; recall is
  scored after training (chance ≈ 1e-4). If Zombie ties `gqa1x` there, the cache isn't storing facts.
* **Steps:** default 6100 = 8192 tok/step ⇒ ≈ 50M tokens ≈ one epoch of 200 MB. Warm/soft phase lengths scale with
  `--steps`. T4 has no bf16: `--amp auto` picks fp16 + GradScaler there, bf16 on A100+. OOM? `--batch 8 --accum 2`
  keeps the effective batch identical.
* **Resumable & fair:** same data order per seed (own RNG), same schedule/clip/optimizer for all models, identical fixed
  eval windows, ppl reported **under hard routing**.


## Dead weights / dead neurons — what is and isn't revived
A single weight can't be "dead"; a **unit** is (a hidden neuron whose output is ~0 on every input, or a memory slot nobody
selects). Two systems, both with no gradient:
* **Memory (always on): Gravekeeper** — dead sub-key rows and their slots (see above).
* **FFN neurons (opt-in, `--revive-neurons` or the `*_nrev` configs): `neuron_keeper.py`** — tracks |GELU| per hidden unit
  during training; a unit active on < 0.1% of the tokens in a 500-step window is dead. Revive: `w1` row ← unit-norm
  copy of a recent input (scaled to live-row norm), `w2` column ← tiny noise, Adam moments of those entries zeroed. It wraps
  *any* `FFN`, so the dense baselines and Zombie's Branch B get identical treatment (run `gqa1x_nrev` with `zombie_nrev`).
* **Not covered:** attention weights, embeddings, the gate (watch `gate_mean` in `log.jsonl`). The end-of-run
  "dead % (GELU)" is a separate *measurement* (|act| < 0.01 on > 95% of tokens) and is reported for every run.
* Caveat: a unit that is legitimately rare (< 0.1% of tokens) also gets reset. Revival is a hypothesis to test
  (`gqa1x` vs `gqa1x_nrev`), not a guaranteed improvement.

## What each run writes (`runs/<name>_s<seed>/`)
`result.json` (val/train ppl, dead-neuron %, neurons revived, routing p_B per layer, slot-usage entropy, dead sub-key rows, MACs and
MB-read at the *measured* p_B, PTQ ppl, KV recall, train tok/s, peak GPU MB), `log.jsonl` (loss, per-layer mean gate,
cache usage, Gravekeeper counts every 50 steps — plot gate collapse from this), `config.json`, `model.pt`.

## CPU speed (run on your i3, not Colab)
```bash
python bench_cpu.py runs/gqa1x_s0 runs/gqa_s0 runs/zombie_s0 --threads 6          # fp32, batch-1 decode, KV cache
python bench_cpu.py runs/zombie_s0 runs/gqa1x_s0 --int8                           # dynamic int8 on both
python bench_cpu.py runs/zombie_s0 --route where                                  # compute-both-and-mask comparison
python counter.py --wb 4 --ctx 512                                                # theoretical params/MACs/bytes tables
```

## Read this before writing the paper
1. **Honest framing:** Zombie has the *most total params* (28.9M vs 24.9M GQA — the cache is 12.7M). The pitch is fewer
   *active* params/bytes per token, not a smaller model. Do not claim "28 MB on disk" or "same param budget".
2. **Whole-model vs FFN-only:** FFN slot alone is ~4.5× fewer MACs/bytes than a 4× FFN, but end to end it is ~2×
   (LM head + attention remain), and **only ~1.04× vs `gqa1x`**. Report both tables (`counter.py`).
3. **Measured speed may not follow theory.** Eager batch-1 decode is op-overhead-bound and the cache gather is
   latency-bound (tables > L3). In a sandbox CPU test Zombie decoded *slower* than `gqa1x`. Report what `bench_cpu.py`
   measures, including if that is negative.
4. **Expect a ppl cost and a Pareto story** (ppl vs MB-read), not a win on both axes; partial replacement
   (`zombie_mem6_11`) is the likeliest sweet spot. A clean mixed result is publishable; an overclaim isn't.
5. **k-means init is an ablation, not a given:** on tiny tests it beat random values from scratch but *hurt* after
   shadow training. `zombie_noinit` vs `zombie` settles it.
6. **Gate collapse is possible** (everything drifts to Branch B, memory dies). Check `gate_mean` and `dead_rows` in
   `log.jsonl`; use the gate-target sweep and `zombie_fixed_half`.
7. "Dead neurons" (GELU) and "slot entropy" (memory) are different metrics — reported side by side, never compared.
8. The "score loss" from the v6 notes is **not implemented** (undefined); `MSE(A, B.detach())` already trains the
   scores via the softmax weights.

## Reasoning suite (does the memory FFN *compute*, not just predict text?)
Perplexity cannot separate storing from computing, so `reason_tasks.py` generates 5 tasks with ground truth:
`mqar` (in-context recall), `hop` (k-hop pointer chasing, needs depth), `add` (little-endian digit addition), `state`
(running sum mod 5), `facts` (facts stored in the weights). Metric = instance-exact accuracy; every task has unseen
harder difficulties (OOD), content-hash-disjoint train/val/test splits, and a trivial-baseline table (cheap shortcuts).
Protocol (`reason_run.py`): tune lr on a separate seed + VAL split with the same grid for every architecture, then run
seeds 0..2 at the chosen lr and score on TEST. `reason_aggregate.py` reports curves, frontier, paired gaps (>2x pooled seed
std and same sign in every seed, else "indistinguishable") and refuses to rank a task no model learned.
Checks: `python reason_tasks.py --selftest` (generator vs independent verifier, split disjointness, no OOD leakage,
shortcut baselines) -- CPU-verified, as are exact resume and the full tune->final->aggregate flow on tiny models.
**Not run on a GPU yet; no real result exists.**

## Verification status
CPU-verified: param counts == closed form for every config; KV-cache decode == full forward (hard routing); causality;
all phases incl. k-means init, Gravekeeper (revive, live-slot protection, optimizer reset), sparse optimizer, route
`subset` == `where`, exact checkpoint resume, fp16 GradScaler + SparseAdam path, KV-recall metric (oracle = 1.0), full
pipeline smoke test. **Not verified on a real GPU** (none available while writing) — run `smoke_test.py` on Colab first.
No full-scale training has been run, so no result in the paper is pre-supposed here.

## Files
`zombie_v6.py` model · `baseline.py` baselines + shared attention/bench · `moe.py` MoE baseline · `configs.py` all runs · `train.py` ·
`evaluate.py` metrics · `prep_data.py` / `kv_recall.py` data · `run_ablations.py` · `aggregate.py` · `bench_cpu.py` ·
`reason_tasks.py` / `reason_train.py` / `reason_run.py` / `reason_aggregate.py` reasoning suite · `neuron_keeper.py` dead-FFN-neuron revival · `counter.py` theoretical costs · `smoke_test.py` · `colab_quickstart.ipynb`
