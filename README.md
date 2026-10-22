# Zombie-Brain: a shared product-key memory in place of the FFN, tested against a MoE on a rigorous reasoning suite

**Naitik (XTErion)** ([ORCID](https://orcid.org/0009-0004-0820-7660)) · independent research · October 2026 · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23552538.svg)](https://doi.org/10.5281/zenodo.23552538) · code, benchmark and results in this repository

> **Authorship and AI disclosure.** The core design (replacing the FFN with a shared product-key memory, reinvesting the freed budget in attention, and the experimental question) was made by the author, an independent student researcher. **The code, the benchmark implementation, the experiment scripts and the first draft of this text were written by an AI model (Claude, by Anthropic)** working from the author's design and direction. The author ran all experiments and reviewed the results. Please read the code and results with that in mind and open an issue if you find an error.

> **TL;DR.** We replace every transformer FFN with a *shared product-key memory* (PKM) and ask whether the freed budget, reinvested in attention depth, beats a standard top-2 MoE at the same total parameters (~22.8M). **It does not, and it does not lose either:** on a procedurally generated reasoning suite (3 seeds, held-out splits, unseen difficulties) every comparison is *indistinguishable* under a pre-registered rule. The deeper variants cost 1.5–2.4× more compute per token. Attention-head dormancy (ReDo) never occurred, so recycling heads had nothing to fix. No model generalises to longer numbers. We release the benchmark and the negative result.

---

## 1. Idea

A standard transformer block is attention + FFN. In **Zombie-Brain** the FFN is replaced by a product-key memory: each token forms a query, looks up the top-4 of ~10⁴ value slots through two small key sets (product keys), and the output is the weighted sum of those values. The key/value tables are **shared by many layers** (two caches serve all layers), so parameters that an FFN would spend per layer become one large, sparsely used store of knowledge.

The thesis we test: *the parameters freed from the FFNs can be spent on more attention (deeper stack, more KV heads), and that lets the model reason better than a MoE of the same total size.* The benchmark here is built so that this can fail.

## 2. Models (all ≈22.8–22.9M parameters, vocabulary 256)

| name | layers | FFN replacement | attention params | notes |
|---|---|---|---|---|
| `moe` | 12 | 12 experts, top-2, hidden 164 (Mixtral-style, load-balance loss) | 4.42M (19%) | baseline |
| `zombie_memonly_b29` | 12 | PKM only (no gate), 21.3k slots/cache | 4.42M (19%) | memory gets the whole budget |
| `zombie_memonly_L20` | 20 | PKM only, 15.9k slots/cache | 7.37M (32%) | FFN budget moved into depth |
| `zombie_memonly_L24` | 24 | PKM only, 13.2k slots/cache | 8.85M (39%) | |
| `zombie_memonly_L24xc` | 24 | PKM **+ routed MoE** (8 experts, top-2, hidden 24) every layer, 4 KV heads, ReDo on | 10.62M (46%) | the "everything" variant |

Common backbone: d=384, GQA (8 query / 2 KV heads, except L24xc: 4 KV), RoPE, RMSNorm, tied embeddings, fp16 autocast. PKM: 2 caches shared across the two halves of the stack, 2 query heads, top-4, per-layer query projection + BatchNorm + rescale of the shared values; cache values trained with SparseAdam. A "Gravekeeper" revives dead sub-key rows from poorly served queries.

## 3. The reasoning suite

All tasks are **procedurally generated** (no downloaded data), with:

* tasks: `mqar` (multi-query associative recall), `hop` (multi-hop pointer chasing), `add` (multi-digit addition), `state` (long running sums mod 5), `facts` (stored-knowledge recall, earlier runs only);
* **instance-exact** accuracy, loss on answer tokens only;
* **held-out splits** by content hash (train / val / test never share an instance), deterministic eval sets, an independent verifier of every generated answer;
* **OOD difficulties**: harder than anything trained on (longer recall lists, more hops, 8- and 10-digit sums, longer state chains);
* a **trivial-baseline table** (best cheating strategy per task/difficulty), so "learned" means *better than a cheat*;
* a **verdict rule**, fixed before the final run: ≥3 seeds, |gap| > 2× pooled std **and** the same sign in every seed; otherwise "indistinguishable"; "n/a" if no model learned the task; "both at ceiling" if all ≥ 0.98.

Training aids, identical for every model: an auxiliary `induct` copy task (dense signal for the induction circuit) and a difficulty curriculum (easiest 25% of train difficulties ramping to all over 1500 steps). Without them, recall plateaus for *every* model on mixed difficulties; with them, recall is learnable by all.

## 4. Main results (final protocol)

6000 steps, batch 32×256 tokens, lr 1e-3 for every model, mix mqar .4 / hop .2 / add .2 / state .1 / induct .1, 3 seeds (0, 1, 2), test split. Mean ± std over seeds.

| | MoE | L24 | L24xc |
|---|---|---|---|
| params | 22.82M | 22.87M | 22.87M |
| speed (k tok/s, T4) | 27.3 | 18.2 | 11.6 |
| mqar ID | 0.745 ± 0.442 | 0.734 ± 0.454 | 0.877 ± 0.213 |
| mqar OOD | 0.435 ± 0.341 | 0.317 ± 0.254 | 0.398 ± 0.264 |
| hop ID | 0.169 ± 0.077 | 0.184 ± 0.080 | 0.199 ± 0.084 |
| add ID | 1.000 | 1.000 | 0.999 |
| add OOD (8, 10 digits) | 0.000 | 0.000 | 0.000 |
| state ID | 0.201 ± 0.003 | 0.261 ± 0.105 | 0.201 ± 0.003 |
| **REASON ID** (mean of 4 tasks) | 0.529 ± 0.128 | 0.545 ± 0.106 | 0.569 ± 0.075 |
| **REASON OOD** | 0.182 ± 0.085 | 0.152 ± 0.065 | 0.172 ± 0.067 |

Paired verdicts vs MoE (REASON ID): **L24 +0.016 (indistinguishable), L24xc +0.040 (indistinguishable; per-seed −0.113 / +0.005 / +0.229)**. Every individual task is indistinguishable, at ceiling (add), or not learned (hop). Full tables, difficulty curves, frontier and paired tests are in [`results/zombie_results.xlsx`](results/zombie_results.xlsx), built from the unedited notebook output saved in [`results/aggregate_output.txt`](results/aggregate_output.txt). **The raw per-run `result.json` files were lost when the Kaggle session was wiped**, so the aggregate output (which contains every number reported here) is the primary record; every seed can be regenerated with the commands in section 8 (data and training stream are a pure function of seed and step).

### What the numbers do and do not say

* **Recall (mqar) is bimodal.** In each seed a model either breaks through to ≈1.0 or stays near the trivial baseline, and *which* seed breaks through is random for *every* model. This swing, not architecture, dominates the means and is why no gap passes the rule. More seeds would be needed to resolve gaps of this size.
* **Addition** is solved in-distribution by all models (MoE and L24 learn it fastest, ≈1000–2500 steps to ≥0.9 at d=3) and **no model extrapolates** to longer numbers (0.000 at 8 and 10 digits).
* **Hop is not learned beyond difficulty 1**, and at difficulty 1 the trivial baseline scores 1.000, so there is no evidence of multi-hop reasoning in any model. **State** is learned in one L24 seed only.
* **Cost.** L24 runs at 0.67× and L24xc at 0.42× MoE's training throughput on the same T4, so quality-per-compute currently favours the MoE.
* **ReDo for attention** (recycling heads whose output is < 10% of their layer's average, Sokar et al. 2023 adapted to GQA) was on for L24xc. The dormant fraction was **0.0% at every checkpoint** (lowest head score 0.13–0.38 against a threshold of 0.1), so ReDo never fired. Head dormancy is not what limits these models.

### Earlier results (different protocol, kept for transparency)

| | MoE | b29 (12 L) | L20 |
|---|---|---|---|
| REASON ID | 0.383 ± 0.017 | 0.264 ± 0.295 | 0.406 ± 0.033 |

* **Language modelling** (12 layers): MoE reaches ~2–2.5% lower validation perplexity (36.29 vs 37.06; 35.57 vs 36.44 at 16 layers).
* **Stored facts**: both reach ≈99%; the memory model reaches ≥0.9 sooner (~step 1500 vs 2500+). This is its one consistent advantage.
* An early "memory beats MoE on OOD recall" result (shared lr 1e-3) **did not replicate** once each model got its own tuned lr; we report it as unconfirmed. The memory-only model prefers a higher lr (3e-3) but is unstable there (one seed learned everything, two learned nothing).

## 5. Engineering note: depth-dependent VRAM

A PKM layer returns a *separate sparse gradient* for the shared value table; with 20–24 layers these (≈100 MB each at 8192 tokens) all stay alive until the optimizer step, so memory grew with depth (10–13 GB on a T4 for L20/L24). We replaced the per-layer sparse gradients by a custom autograd function that accumulates into **one dense buffer per cache** and converts it to the sparse gradient SparseAdam expects after backward. The update is identical (loss equal, parameters within 3·10⁻⁶ after 3 steps) and peak memory no longer grows with depth (CPU test at 4096 tokens: 24-layer model 5.2 GB; the old path exceeded 7 GB).

## 6. Limitations

* **One small scale** (≈23M params, 6000 steps) on a single free GPU class. Nothing here says how the comparison behaves at 100M+ parameters; scaling is a hypothesis, not a result.
* **Three seeds**, with a bimodal metric: gaps of the size seen here cannot be resolved. "Indistinguishable" is not proof of "equal".
* **Synthetic tasks.** The suite tests recall, in-context lookup, arithmetic and state tracking; it is not a language-understanding benchmark.
* **Multi-hop and long-chain reasoning were not learned by any model**; OOD length generalisation is 0 for addition.
* The memory models were tuned (lr) on one validation seed per model; the earlier-protocol runs used different lrs and mixes and are not comparable with the final protocol.
* The MoE baseline has very small experts (hidden 164, and 24 in L24xc) because the 23M budget forces it.

## 7. Future work

* More seeds (≥10) and 12k-step runs, to resolve the recall gaps.
* The same comparison at ~10M / 23M / 50M parameters, with a *compute-matched* MoE as a second baseline, to test whether the gap grows, shrinks or stays flat.
* A fix for the memory model's instability at high lr.
* Tasks that need real multi-step composition, where the depth advantage should show if it exists.

## 8. Reproduce

```bash
pip install -r requirements.txt
python reason_tasks.py --selftest                      # generators, splits, verifier
python reason_train.py --name zombie_memonly_L24 --tiny --steps 30 --facts 0 --eval-every 0   # CPU smoke test

# the final protocol (2 GPUs: one job per GPU, resumable; re-run the same command after a restart)
C="--steps 6000 --facts 0 --mix mqar=.4,hop=.2,add=.2,state=.1,induct=.1 --curriculum 1500 --dormancy-every 500"
python reason_parallel.py --archs moe,zombie_memonly_L24,zombie_memonly_L24xc --seeds 3 --out runs/paper -- --lr 1e-3 $C
python reason_aggregate.py --runs runs/paper --pair zombie_memonly_L24xc,moe
python reason_aggregate.py --runs runs/paper --pair zombie_memonly_L24,moe
```

Runtime on 2×T4: ≈4–5 h. `/kaggle/working` can be wiped between sessions (it was, for our raw files): download `runs/paper` as soon as it finishes.

## 9. Files

`zombie_v6.py` memory models · `moe.py` MoE baseline · `baseline.py` shared transformer · `configs.py` all named models · `reason_tasks.py` benchmark · `reason_train.py` / `reason_parallel.py` / `reason_aggregate.py` training and statistics · `attn_redo.py` dormant-head ReDo · `results/` the workbook and the saved aggregate output of the runs.

## 10. Citation

```
Naitik (XTErion). Zombie-Brain: a shared product-key memory in place of the FFN, tested against a MoE
on a rigorous reasoning suite. v0.1.0, 2026. https://doi.org/10.5281/zenodo.23552538
```

DOI: [10.5281/zenodo.23552538](https://doi.org/10.5281/zenodo.23552538) · ORCID: [0009-0004-0820-7660](https://orcid.org/0009-0004-0820-7660) · GitHub's "Cite this repository" button reads `CITATION.cff`.

## 11. License

Released under the **MIT License** (see `LICENSE`): free to use, copy, modify and distribute, including commercially, as long as the copyright notice is kept.
