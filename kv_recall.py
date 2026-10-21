#!/usr/bin/env python3
"""kv_recall.py -- synthetic key->value memorisation task.

TinyStories-style text has little *factual* content, so it barely tests "memory as facts".
This adds N random facts ("qzxkf => 4821") to the TRAIN split only (prep_data.py --kv-pairs N), repeated
many times, and measures after training whether the model can recall them. 4 random digits = 1e-4 chance.
The metric is exact match of ALL value tokens, teacher-forced (identical to greedy generation, since a greedy
decode reproduces the target iff the argmax at every value position matches), so it is one forward pass/batch.

If the memory path works anywhere it should work here; if Zombie ties GQA+1xFFN on this, the cache is not
storing facts.
"""
import json
import random
import string


def make_pairs(n, seed=0):
    rng = random.Random(seed)
    keys, pairs = set(), []
    while len(pairs) < n:
        k = "".join(rng.choices(string.ascii_lowercase, k=5))
        if k in keys:
            continue
        keys.add(k)
        pairs.append({"key": k, "value": f"{rng.randrange(10000):04d}"})
    return pairs


def line(p):
    return f"{p['key']} => {p['value']}\n"


def make_train_blocks(pairs, repeat=20, lines_per_block=100, seed=0):
    """Every pair appears `repeat` times, shuffled, grouped into blocks of text."""
    rng = random.Random(seed)
    pool = [p for p in pairs for _ in range(repeat)]
    rng.shuffle(pool)
    return ["".join(line(p) for p in pool[i:i + lines_per_block]) for i in range(0, len(pool), lines_per_block)]


def load_pairs(path):
    with open(path) as f:
        return json.load(f)


def kv_eval(model, tok, pairs, device, amp=None, max_pairs=2000, bs=64):
    """-> dict(acc=exact match of all value tokens, tok_acc=per-token accuracy, n=evaluated, skipped=...)"""
    import torch
    model.eval()
    pairs = pairs[:max_pairs]
    ex, skipped = [], 0
    for p in pairs:
        a = tok.encode(f"{p['key']} =>").ids
        b = tok.encode(f"{p['key']} => {p['value']}").ids
        if b[:len(a)] != a or len(b) <= len(a):
            skipped += 1
            continue
        ex.append((b, len(a)))
    ok = tok_ok = tok_n = 0
    with torch.no_grad():
        for i in range(0, len(ex), bs):
            chunk = ex[i:i + bs]
            L = max(len(b) for b, _ in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, (b, _) in enumerate(chunk):
                x[r, :len(b)] = torch.tensor(b)
            x = x.to(device)
            ctx = amp() if amp is not None else torch.autocast("cpu", enabled=False)
            with ctx:
                logits = model(x)[0]
            pred = logits.argmax(-1).cpu()
            for r, (b, pl) in enumerate(chunk):
                tgt = torch.tensor(b[pl:])
                hit = pred[r, pl - 1:len(b) - 1] == tgt
                ok += int(hit.all())
                tok_ok += int(hit.sum())
                tok_n += len(tgt)
    n = len(ex)
    return dict(acc=ok / max(n, 1), tok_acc=tok_ok / max(tok_n, 1), n=n, skipped=skipped)
