#!/usr/bin/env python3
"""prep_data.py -- raw text  ->  8k byte-level BPE  ->  train.bin / val.bin (uint16) + tokenizer.json + meta.json

    python prep_data.py --input /content/mydata --out data --vocab 8192 --kv-pairs 5000

    python prep_data.py --train train.txt --val validation.txt --test test.txt --out data --vocab 8192 --kv-pairs 5000

* --train/--val/--test  YOUR OWN splits, used exactly as given (nothing is re-split, test is never trained on).
            Prints a leakage check (% of val/test lines that also occur in train). Tokenizer sees train only.
* --input   one pool of files/globs/folders (.txt .md .text; .jsonl/.json with --jsonl-field text); a random 10% of
            blocks becomes val. Do NOT point --input at a folder that holds train/val/test files: it would merge them.
* Split     BLOCK-level (default 64k chars, cut at newlines), randomly 90/10 with a fixed seed, so val has the same
            distribution as train and no val text is ever seen in training (the tokenizer is also trained on the
            train blocks only). Report both train and val ppl later.
* --kv-pairs N   adds N synthetic facts to the TRAIN split only and writes kv_test.json (see kv_recall.py).
* Output    data/train.bin, data/val.bin, data/tokenizer.json, data/meta.json, [data/kv_test.json]
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np

import kv_recall

EXTS = (".txt", ".text", ".md", ".jsonl", ".json")


def list_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                files += [os.path.join(r, f) for f in sorted(fs) if f.lower().endswith(EXTS)]
        else:
            files += sorted(glob.glob(p)) or [p]
    if not files:
        raise SystemExit("no input files found")
    return files


def read_text(path, jsonl_field):
    with open(path, encoding="utf-8", errors="ignore") as f:
        if path.lower().endswith((".jsonl", ".json")):
            if not jsonl_field:
                raise SystemExit(f"{path}: pass --jsonl-field <name> for json/jsonl input")
            out = []
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(str(json.loads(ln)[jsonl_field]))
                    except Exception:
                        pass
            return "\n\n".join(out)
        return f.read()


def blocks_of(text, size):
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + size)
        if j < n:
            k = text.find("\n", j, j + size)           # extend to the next newline (bounded)
            j = k + 1 if k != -1 else j
        if text[i:j].strip():
            yield text[i:j]
        i = j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=None,
                    help="everything in one pool; a random block-level val split is made for you")
    ap.add_argument("--train", nargs="+", default=None, help="explicit train files (use with --val [--test])")
    ap.add_argument("--val", nargs="+", default=None, help="explicit validation files")
    ap.add_argument("--test", nargs="+", default=None, help="explicit test files (never trained on; scored once at the end)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--block-chars", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jsonl-field", default=None)
    ap.add_argument("--tok-sample-chars", type=int, default=150_000_000, help="cap on text used to train the BPE")
    ap.add_argument("--kv-pairs", type=int, default=0)
    ap.add_argument("--kv-repeat", type=int, default=20)
    ap.add_argument("--eos", default="<|endoftext|>")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    t0 = time.time()
    rng = random.Random(a.seed)

    def load_blocks(paths, label):
        out = []
        for f in list_files(paths):
            txt = read_text(f, a.jsonl_field)
            nb = len(out)
            out += list(blocks_of(txt, a.block_chars))
            print(f"  [{label}] {f}: {len(txt) / 1e6:.1f}M chars -> {len(out) - nb} blocks")
        return out

    test_blocks = []
    if a.train or a.val or a.test:
        # ---- explicit splits: used exactly as given (NO random re-split, so test text is never trained on)
        if a.input:
            raise SystemExit("use either --input (one pool, random val split) or --train/--val[/--test], not both")
        if not (a.train and a.val):
            raise SystemExit("explicit splits need both --train and --val (--test is optional)")
        train_blocks = load_blocks(a.train, "train")
        val_blocks = load_blocks(a.val, "val")
        test_blocks = load_blocks(a.test, "test") if a.test else []
        # leakage check: how many substantial lines of val/test also occur verbatim in train?
        seen = {hash(ln) for b in train_blocks for ln in b.splitlines() if len(ln) >= 20}
        for label, bl in (("val", val_blocks), ("test", test_blocks)):
            lines = [ln for b in bl for ln in b.splitlines() if len(ln) >= 20]
            if lines:
                dup = sum(hash(ln) in seen for ln in lines) / len(lines)
                print(f"  leakage check: {100 * dup:.1f}% of {label} lines (>=20 chars) also appear verbatim in train"
                      + ("   <-- high: remove duplicates or val/test ppl will look too good" if dup > 0.05 else ""))
        del seen
        print(f"explicit split: {len(train_blocks)} train / {len(val_blocks)} val / {len(test_blocks)} test blocks")
    else:
        if not a.input:
            raise SystemExit("give --input (one pool) or --train/--val[/--test] (your own splits)")
        blocks = load_blocks(a.input, "all")
        val_idx = {i for i in range(len(blocks)) if rng.random() < a.val_frac}
        if not val_idx:
            val_idx = {len(blocks) - 1}
        if len(val_idx) == len(blocks):
            raise SystemExit("not enough text for a train/val split")
        train_blocks = [b for i, b in enumerate(blocks) if i not in val_idx]
        val_blocks = [blocks[i] for i in sorted(val_idx)]
        del blocks
        print(f"random split: {len(train_blocks)} train blocks, {len(val_blocks)} val blocks")

    kv_pairs = []
    if a.kv_pairs > 0:
        kv_pairs = kv_recall.make_pairs(a.kv_pairs, a.seed)
        kvb = kv_recall.make_train_blocks(kv_pairs, a.kv_repeat, seed=a.seed)
        train_blocks += kvb
        rng.shuffle(train_blocks)
        with open(os.path.join(a.out, "kv_test.json"), "w") as f:
            json.dump(kv_pairs, f)
        print(f"kv task: {a.kv_pairs} facts x{a.kv_repeat} repeats added to TRAIN only ({len(kvb)} blocks)")

    # ---- tokenizer: train split only
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[a.eos], show_progress=True,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def sample():
        used = 0
        for b in train_blocks:
            if used >= a.tok_sample_chars:
                return
            used += len(b)
            yield b
    tok.train_from_iterator(sample(), trainer)
    eos_id = tok.token_to_id(a.eos)
    tok.save(os.path.join(a.out, "tokenizer.json"))
    print(f"tokenizer: vocab {tok.get_vocab_size()}, eos id {eos_id}  ({time.time() - t0:.0f}s)")

    def encode(bl, path):
        n = 0
        with open(path, "wb") as f:
            for i in range(0, len(bl), 128):
                for e in tok.encode_batch(bl[i:i + 128]):
                    arr = np.asarray(e.ids + [eos_id], dtype=np.uint16)
                    f.write(arr.tobytes())
                    n += len(arr)
        return n

    ntr = encode(train_blocks, os.path.join(a.out, "train.bin"))
    nva = encode(val_blocks, os.path.join(a.out, "val.bin"))
    nte = encode(test_blocks, os.path.join(a.out, "test.bin")) if test_blocks else 0
    if not test_blocks and os.path.exists(os.path.join(a.out, "test.bin")):
        os.remove(os.path.join(a.out, "test.bin"))          # stale file from an earlier prep into the same folder
    meta = dict(vocab=tok.get_vocab_size(), eos=eos_id, train_tokens=ntr, val_tokens=nva, test_tokens=nte, seed=a.seed,
                kv_pairs=a.kv_pairs, kv_repeat=a.kv_repeat if a.kv_pairs else 0,
                split="explicit" if (a.train or a.val or a.test) else "random")
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"train {ntr / 1e6:.2f}M tokens | val {nva / 1e6:.2f}M tokens"
          + (f" | test {nte / 1e6:.2f}M tokens" if nte else "") + f" | {time.time() - t0:.0f}s total")
    print(f"batch 16 x seq 512 = 8192 tok/step  ->  one epoch = {ntr // 8192} steps")


if __name__ == "__main__":
    main()
