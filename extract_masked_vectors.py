# -*- coding: utf-8 -*-
"""Stage 3b-alt: TARGET-MASKED contextual vectors.

Motivation (this is the whole point of the file)
------------------------------------------------
The plain target-span vector encodes the surface form: 绷带/绷紧/绷不住 differ
mostly in the characters immediately after 绷, so clustering those vectors
recovers orthographic templates rather than meaning. Clustering then reports
"clusters = discourse templates", which is not a sense inventory.

Standard fix from the word-sense-induction literature (Amrami & Goldberg 2019;
Yaeini et al. use the same masked-probe idea): replace the target span with the
mask token and read the hidden state AT the mask position. The model must then
infer the sense from context alone, so what varies across items is meaning, not
spelling.

What is produced (separate tree; the unmasked vectors are never overwritten)
  data/processed/emb2/masked/shards/shard_XXX.npz   (n, 13, 768) float16
  data/processed/emb2/masked/index.jsonl
  data/processed/emb2/masked/meta.json

Run with the SAME cleaned table used for the unmasked extraction so the two
representations are directly comparable item-for-item.
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
OUT_DIR = C.EMB_MASK_DIR
SHARD_DIR = os.path.join(OUT_DIR, "shards")
INDEX = os.path.join(OUT_DIR, "index.jsonl")
META = os.path.join(OUT_DIR, "meta.json")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=C.MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--text-field", default="text_norm")
    a = ap.parse_args()

    os.makedirs(SHARD_DIR, exist_ok=True)
    rows = read_jsonl(CLEAN)
    print(f"records: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(C.EMBED_MODEL)
    model = AutoModel.from_pretrained(C.EMBED_MODEL)
    model.eval()
    L = model.config.num_hidden_layers
    H = model.config.hidden_size
    mask_tok = tok.mask_token
    print(f"model={C.EMBED_MODEL} layers={L} hidden={H} mask_token={mask_tok!r}")

    n_shards = (len(rows) + a.shard_size - 1) // a.shard_size
    t0 = time.time()
    for si in range(n_shards):
        path = os.path.join(SHARD_DIR, f"shard_{si:03d}.npz")
        if os.path.exists(path):
            print(f"  shard {si:03d} exists, skip")
            continue
        chunk = rows[si * a.shard_size:(si + 1) * a.shard_size]
        embs = np.zeros((len(chunk), L + 1, H), dtype=np.float16)
        meta_rows = []
        buf, rows_in_buf = [], []

        def flush():
            nonlocal buf, rows_in_buf
            if not buf:
                return
            texts = [b["masked"] for b in buf]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len,
                      return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping").tolist()
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            if out.hidden_states is None:
                raise RuntimeError("no hidden states")
            stack = torch.stack(out.hidden_states, dim=1)
            for bi, b in enumerate(buf):
                offs = offsets[bi]
                lo, hi = b["mask_char"], b["mask_char"] + b["mask_len"]
                span = [t for t, (s_, e_) in enumerate(offs)
                        if s_ != e_ and s_ < hi and e_ > lo]
                if not span:
                    meta_rows.append({"row": b["row"], "clean_id": b["clean_id"],
                                      "status": "no_mask_token",
                                      "n_tokens": int(enc["attention_mask"][bi].sum())})
                    continue
                vec = stack[bi][:, span, :].mean(dim=1)
                embs[b["row"]] = vec.to(torch.float16).numpy()
                meta_rows.append({"row": b["row"], "clean_id": b["clean_id"],
                                  "status": "ok",
                                  "mask_token_span": [int(span[0]), int(span[-1])],
                                  "n_tokens": int(enc["attention_mask"][bi].sum())})
            buf, rows_in_buf = [], []

        for ri, r in enumerate(chunk):
            text = r.get(a.text_field) or ""
            s = r.get("target_start_norm", -1)
            e = r.get("target_end_norm", -1)
            if not text or s < 0 or e <= s or e > len(text):
                meta_rows.append({"row": ri, "clean_id": r.get("clean_id"),
                                  "status": "bad_record"})
                continue
            masked = text[:s] + mask_tok + text[e:]
            mpos = masked.find(mask_tok)
            buf.append({"masked": masked, "mask_char": mpos,
                        "mask_len": len(mask_tok), "row": ri,
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
        print(f"  shard {si:03d}: {len(chunk)} rows, ok={ok}, "
              f"elapsed={(time.time()-t0)/60:.1f}min")

    fp = {}
    fp_path = os.path.join(C.ROOT, "models", "model_fingerprint.json")
    if os.path.exists(fp_path):
        fp = json.load(open(fp_path, encoding="utf-8"))
    json.dump({
        "model": C.EMBED_MODEL,
        "representation": "target-masked, pooled at [MASK] position",
        "mask_token": mask_tok,
        "n_layers": L, "hidden_size": H,
        "text_field": a.text_field,
        "n_records": len(rows),
        "weights_sha256": fp.get("weights_sha256"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "why": "removes surface-form signal so clusters reflect context/meaning "
               "rather than orthographic templates",
    }, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n->", META)


if __name__ == "__main__":
    main()
