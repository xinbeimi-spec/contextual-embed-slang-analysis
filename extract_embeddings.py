# -*- coding: utf-8 -*-
"""Stage 3b: contextualised word-vector extraction.

Design notes that matter for the write-up:

1. The vector for a token occurrence is the MEAN of the hidden states over the
   tokens that cover the target character span, taken from EVERY layer. We keep
   all 12 layers so the layer choice can be made by a probe instead of a guess
   (that probe is also the direct evidence for RQ3).

2. Target location uses the tokenizer's offset mapping, not string matching,
   because subword segmentation differs from characters. The record stores
   character offsets, so any tokenizer can be used later.

3. Causal/decoder models are deliberately NOT supported here; see config.py.

Outputs
  data/processed/emb2/target/shard_XXX.npz   emb (n, 13, 768) float16, plus ids
  data/processed/emb2/target/index.jsonl     clean_id -> shard,row,target_token_span
  data/processed/emb2/target/meta.json       model fingerprint + params

Resumable: existing shards are skipped.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModel  # noqa: E402

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
EMB_DIR = C.EMB_DIR
SHARD_DIR = os.path.join(EMB_DIR, "shards")
INDEX = os.path.join(EMB_DIR, "index.jsonl")
META = os.path.join(EMB_DIR, "meta.json")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=C.MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--text-field", default="text_norm")
    a = ap.parse_args()

    os.makedirs(SHARD_DIR, exist_ok=True)
    rows = read_jsonl(CLEAN)
    if a.max_records:
        rows = rows[:a.max_records]
    print(f"records to embed: {len(rows)}")

    print("loading tokenizer/model:", C.EMBED_MODEL)
    tok = AutoTokenizer.from_pretrained(C.EMBED_MODEL)
    model = AutoModel.from_pretrained(C.EMBED_MODEL,
                                      output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"layers={n_layers} hidden={hidden}")

    n_shards = (len(rows) + a.shard_size - 1) // a.shard_size
    print(f"shards: {n_shards} (size {a.shard_size})")

    t_all = time.time()
    for si in range(n_shards):
        path = os.path.join(SHARD_DIR, f"shard_{si:03d}.npz")
        if os.path.exists(path):
            print(f"  shard {si:03d} exists, skip")
            continue
        chunk = rows[si * a.shard_size:(si + 1) * a.shard_size]
        embs = np.zeros((len(chunk), n_layers + 1, hidden), dtype=np.float16)
        meta_rows = []
        buf, buf_idx = [], []

        def flush():
            nonlocal buf, buf_idx
            if not buf:
                return
            texts = [b["text"] for b in buf]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping").tolist()
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states          # tuple: (emb, layer1..layerN)
            if hs is None:
                raise RuntimeError("model returned no hidden_states")
            stack = torch.stack(hs, dim=1)  # (B, L+1, T, H)
            for bi, b in enumerate(buf):
                offs = offsets[bi]
                start, end = b["start"], b["end"]
                span = [t for t, (s, e) in enumerate(offs)
                        if s != e and s < end and e > start]
                if not span:
                    meta_rows.append({"row": b["row"], "clean_id": b["clean_id"],
                                      "status": "no_token_for_target",
                                      "n_tokens": int(enc["attention_mask"][bi].sum())})
                    continue
                vec = stack[bi][:, span, :].mean(dim=1)   # (L+1, H)
                embs[b["row"]] = vec.to(torch.float16).numpy()
                meta_rows.append({"row": b["row"], "clean_id": b["clean_id"],
                                  "status": "ok",
                                  "target_token_span": [int(span[0]), int(span[-1])],
                                  "n_tokens": int(enc["attention_mask"][bi].sum())})
            buf, buf_idx = [], []

        for ri, r in enumerate(chunk):
            text = r.get(a.text_field) or ""
            start = r.get("target_start_norm", -1)
            end = r.get("target_end_norm", -1)
            if not text or start < 0 or end <= start or end > len(text):
                meta_rows.append({"row": ri, "clean_id": r.get("clean_id"),
                                  "status": "bad_record"})
                continue
            buf.append({"text": text, "start": start, "end": end, "row": ri,
                        "clean_id": r.get("clean_id")})
            if len(buf) >= a.batch_size:
                flush()
        flush()

        np.savez_compressed(path, emb=embs)
        with open(INDEX, "a", encoding="utf-8") as f:
            for m in meta_rows:
                m["shard"] = si
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        ok = sum(1 for m in meta_rows if m["status"] == "ok")
        print(f"  shard {si:03d}: {len(chunk)} records, ok={ok}, "
              f"elapsed={(time.time()-t_all)/60:.1f}min")

    fp = {}
    if os.path.exists(os.path.join(C.ROOT, "models", "model_fingerprint.json")):
        fp = json.load(open(os.path.join(C.ROOT, "models",
                                         "model_fingerprint.json"), encoding="utf-8"))
    json.dump({
        "model": C.EMBED_MODEL,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "layers_saved": "0(embedding)..N(layer N)",
        "pooling": "mean over target-span tokens",
        "text_field": a.text_field,
        "max_len": a.max_len,
        "n_records": len(rows),
        "weights_sha256": fp.get("weights_sha256"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n->", META)


if __name__ == "__main__":
    main()
