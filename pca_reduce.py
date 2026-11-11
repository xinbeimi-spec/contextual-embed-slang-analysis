# -*- coding: utf-8 -*-
"""Stage 4a: PCA dimensionality reduction.

The original representation is never overwritten. Everything here goes into a
NEW folder (data/processed/pca/) so both versions can be compared later, which
is also what makes the "reduce-then-cluster" choice auditable.

Design decision recorded here and reused downstream:
  * clustering runs in a MEDIUM-dimensional space (default: the first 50 PCs,
    which retain most of the variance) -- this denoises without flattening the
    geometry;
  * the 2-D projection produced here is FOR PLOTTING ONLY. Clustering is never
    done in 2-D, because 2 components discard most of the variance and the
    resulting clusters would be an artifact of the projection.

Outputs (data/processed/pca/)
  pca_scores_{k}.npz   scores (N,k) float32 + ids + explained ratio
  pca_2d.npz           first 2 PC scores (plotting only)
  pca_params.npz       components_, mean_, explained_variance_ratio_
  explained_variance.csv
  pca_meta.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

EMB = C.EMB_DIR
OUT_DIR = os.path.join(C.PROCESSED, "pca")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50, help="PCs kept for clustering")
    ap.add_argument("--source", default=os.path.join(EMB, "centered.npz"))
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.load(a.source)
    X, ids = z["X"].astype(np.float64), list(z["ids"])
    N, D = X.shape
    print(f"source: {a.source}")
    print(f"X = {X.shape}  (N={N}, D={D})")

    mu = X.mean(axis=0)
    Xc = X - mu

    # full SVD-derived PCA (cheap for 2663 x 768)
    t0 = time.time()
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / max(1, N - 1)
    ratio = ev / ev.sum()
    cum = np.cumsum(ratio)
    print(f"PCA computed in {time.time()-t0:.1f}s")

    def k_for(t):
        return int(np.searchsorted(cum, t) + 1)

    k80, k90 = k_for(0.80), k_for(0.90)
    print(f"components for 80% variance: {k80}")
    print(f"components for 90% variance: {k90}")
    print(f"variance explained by first {a.k} PCs: {cum[a.k-1]:.4f}")
    print(f"variance explained by first 2 PCs: {cum[1]:.4f}")

    scores = Xc @ Vt.T                        # (N, D)
    k = min(a.k, D)

    np.savez_compressed(os.path.join(OUT_DIR, f"pca_scores_{k}.npz"),
                        scores=scores[:, :k].astype(np.float32),
                        ids=np.array(ids),
                        explained=ratio[:k].astype(np.float32))
    np.savez_compressed(os.path.join(OUT_DIR, "pca_2d.npz"),
                        scores=scores[:, :2].astype(np.float32),
                        ids=np.array(ids),
                        explained=ratio[:2].astype(np.float32))
    np.savez_compressed(os.path.join(OUT_DIR, "pca_params.npz"),
                        components=Vt[:k].astype(np.float32),
                        mu=mu.astype(np.float32),
                        explained_variance_ratio=ratio[:k].astype(np.float32),
                        explained_variance=ev[:k].astype(np.float32))

    import csv
    p = os.path.join(C.TABLES, "explained_variance.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pc", "explained_ratio", "cumulative"])
        for i in range(min(100, D)):
            w.writerow([i + 1, round(float(ratio[i]), 6), round(float(cum[i]), 6)])
    print("->", p)

    meta = {
        "source": a.source,
        "n_items": int(N),
        "input_dim": int(D),
        "k_for_clustering": int(k),
        "k_for_80pct": int(k80),
        "k_for_90pct": int(k90),
        "var_explained_k": round(float(cum[k - 1]), 4),
        "var_explained_2d": round(float(cum[1]), 4),
        "rule": "cluster in k dimensions; 2-D is for plotting only",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(meta, open(os.path.join(OUT_DIR, "pca_meta.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n=>", OUT_DIR)
    for fn in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, fn)
        print(f"   {fn:28} {os.path.getsize(fp)/1024:9.1f} KB")


if __name__ == "__main__":
    main()
