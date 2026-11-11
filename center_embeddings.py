# -*- coding: utf-8 -*-
"""Stage 3d: anisotropy diagnosis + centering.

Why this exists
---------------
BERT-family contextual vectors are strongly anisotropic: all vectors sit in a
narrow cone, so plain cosine similarity between any two items is high (0.7-0.9)
and any clustering degenerates into "one giant blob plus fragments". Cosine-based
semantic work therefore needs the shared direction removed.

What is produced
  data/processed/emb2/target/centered.npz        Xc (N, D) float32 + ids
  data/processed/emb2/target/center_params.npz   per-layer mean vector, PCA basis, etc.
  results/tables/anisotropy_by_layer.csv
  results/tables/centering_effect.csv

Layer choice here is NOT made with labels (that probe would be circular: the
classes are defined by surface strings that appear in the input). Instead a
label-free diagnostic is used: mean cosine between random pairs, and
||mean|| / mean(||x||). Lower is less anisotropic.

Centering variants
  mean      : x - mu            (the standard, keeps all directions)
  abtt(k)   : mean + remove top-k principal directions
              (All-but-the-Top; k=1 or 2 is usually enough)
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

EMB_DIR = C.EMB_DIR
SHARD_DIR = os.path.join(EMB_DIR, "shards")
CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_all():
    index = read_jsonl(os.path.join(EMB_DIR, "index.jsonl"))
    by_shard = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by_shard[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by_shard):
        emb = np.load(os.path.join(SHARD_DIR, f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by_shard[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    return np.stack(xs).astype(np.float32), ids


def pairwise_cos_sample(X, n_pairs=20000, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    m = a != b
    a, b = a[m], b[m]
    A = X[a]
    B = X[b]
    na = np.linalg.norm(A, axis=1) + 1e-9
    nb = np.linalg.norm(B, axis=1) + 1e-9
    return float(np.mean((A * B).sum(1) / (na * nb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="9,10,11,12",
                    help="comma list of layer indices to average for the final "
                         "representation (0=embedding, 12=last)")
    ap.add_argument("--abtt-k", type=int, default=1,
                    help="principal directions to remove in the abtt variant")
    a = ap.parse_args()

    X, ids = load_all()
    N, L1, H = X.shape
    print(f"loaded: N={N} layers={L1} dim={H}")

    # ---------- 1. per-layer anisotropy (label-free)
    rows = []
    for li in range(L1):
        Xi = X[:, li, :]
        mu = Xi.mean(axis=0)
        mean_norm = float(np.linalg.norm(mu))
        avg_norm = float(np.mean(np.linalg.norm(Xi, axis=1)))
        ratio = mean_norm / (avg_norm + 1e-9)
        pc = pairwise_cos_sample(Xi)
        rows.append({"layer": li, "mean_cos_random_pairs": round(pc, 4),
                     "norm_of_mean": round(mean_norm, 3),
                     "mean_norm": round(avg_norm, 3),
                     "mean_over_avg": round(ratio, 4)})
        print(f"  layer {li:2d}  cos_random={pc:.4f}  ||mu||={mean_norm:7.2f}  "
              f"mean||x||={avg_norm:7.2f}  ratio={ratio:.4f}")

    import csv
    p1 = os.path.join(C.TABLES, "anisotropy_by_layer.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("->", p1)

    # ---------- 2. final representation = mean of requested layers
    layers = [int(s) for s in a.layers.split(",") if s.strip() != ""]
    print(f"\nfinal representation: mean of layers {layers}")
    Xr = X[:, layers, :].mean(axis=1)          # (N, H)

    # ---------- 3. center
    mu = Xr.mean(axis=0, keepdims=True)
    Xc_mean = Xr - mu

    # abtt: remove top-k principal directions of the centered data
    Xc = Xc_mean.copy()
    U, S, Vt = np.linalg.svd(Xc_mean, full_matrices=False)
    k = a.abtt_k
    if k > 0:
        comp = Vt[:k]                          # (k, H)
        Xc = Xc - (Xc @ comp.T) @ comp
    print(f"removed {k} principal direction(s); "
          f"top singular values: {np.round(S[:5], 2)}")

    # ---------- 4. effect report
    def stats(Z, tag):
        return {
            "variant": tag,
            "cos_random_pairs": round(pairwise_cos_sample(Z), 4),
            "norm_of_mean": round(float(np.linalg.norm(Z.mean(0))), 3),
            "mean_norm": round(float(np.mean(np.linalg.norm(Z, axis=1))), 3),
        }

    rep = [stats(Xr, "raw_last4mean"),
           stats(Xc_mean, "mean_centered"),
           stats(Xc, f"abtt_k{k}")]
    p2 = os.path.join(C.TABLES, "centering_effect.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
        w.writeheader()
        w.writerows(rep)
    print(f"\n{'variant':18}{'cos_random':>12}{'||mu||':>10}{'mean||x||':>12}")
    for r in rep:
        print(f"{r['variant']:18}{r['cos_random_pairs']:>12}{r['norm_of_mean']:>10}"
              f"{r['mean_norm']:>12}")
    print("->", p2)

    # ---------- 5. persist
    np.savez_compressed(os.path.join(EMB_DIR, "centered.npz"),
                        X=Xc.astype(np.float32), ids=np.array(ids))
    np.savez_compressed(os.path.join(EMB_DIR, "center_params.npz"),
                        mu=mu.astype(np.float32),
                        pca_components=(Vt[:k].astype(np.float32) if k > 0
                                        else np.zeros((0, H), np.float32)),
                        layers=np.array(layers))
    meta_p = os.path.join(EMB_DIR, "centering_meta.json")
    json.dump({
        "n_items": int(N),
        "layers_averaged": layers,
        "hidden": int(H),
        "variant": f"mean-center + abtt(k={k})",
        "mean_vector_saved": "center_params.npz:mu",
        "pca_components_saved": "center_params.npz:pca_components",
        "note": "apply the SAME mu and PCA components to any new item before "
                "cosine/clustering; never re-fit per subset",
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n->", os.path.join(EMB_DIR, "centered.npz"))
    print("->", os.path.join(EMB_DIR, "center_params.npz"))
    print("->", meta_p)


if __name__ == "__main__":
    main()
