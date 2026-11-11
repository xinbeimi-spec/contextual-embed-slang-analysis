# -*- coding: utf-8 -*-
"""Is it defensible to drop cluster 6 (the 578-item residual)?

Checks
  1. length distribution vs the other clusters
  2. how the residual lines up with context_sufficiency (the pre-existing,
     non-circular "can a human judge this?" flag)
  3. does it contain informative long items that dropping would lose?
  4. does the data reproduce a 5-way structure after the residual is removed
     (refit), or is "merge 5 into 0" purely our interpretive choice?
"""
import os
import sys
import json
import collections

import numpy as np
import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")
jieba.setLogLevel(60)

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
ROOT = C.EMB_MASK_DIR
LAYERS = (9, 10, 11, 12)


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load():
    index = rj(os.path.join(ROOT, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(ROOT, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    Z = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    Z = Z - (Z @ Vt[:1].T) @ Vt[:1]
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12), ids


def main():
    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    Z, ids = load()
    texts = [meta[i].get("text_norm", "") for i in ids]
    suff = [int(meta[i].get("context_sufficiency") or 0) for i in ids]

    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=7, n_init=30,
                 random_state=C.RANDOM_SEED).fit_predict(Z)

    print("=== length and sufficiency per cluster ===")
    print(f"{'cl':>4}{'n':>7}{'mean_len':>10}{'median':>8}"
          f"{'sufficiency=1':>15}{'<=10 chars':>12}")
    for c in range(7):
        idx = np.where(lab == c)[0]
        L = [len(texts[i]) for i in idx]
        s1 = sum(suff[i] for i in idx)
        short = sum(1 for i in idx if len(texts[i]) <= 10)
        print(f"{c:>4}{len(idx):>7}{np.mean(L):>10.1f}{np.median(L):>8.0f}"
              f"{s1:>15}{short:>12}")

    # ---- residual cluster
    c6 = np.where(lab == 6)[0]
    print(f"\n=== residual cluster 6: n={len(c6)} ===")
    print(f"  sufficiency=1: {sum(suff[i] for i in c6)} "
          f"({100*sum(suff[i] for i in c6)/len(c6):.1f}%)")
    L = [len(texts[i]) for i in c6]
    print(f"  length: mean={np.mean(L):.1f} median={np.median(L):.0f} "
          f"p90={np.percentile(L, 90):.0f} max={max(L)}")
    print(f"  <=10 chars: {sum(1 for x in L if x <= 10)} "
          f"({100*sum(1 for x in L if x <= 10)/len(L):.1f}%)")
    print("  longest 5:")
    for i in sorted(c6, key=lambda i: -len(texts[i]))[:5]:
        print("    ·", texts[i][:92])
    print("  random 8 from the informative subset (len>=12):")
    inf = [i for i in c6 if len(texts[i]) >= 12]
    print(f"    informative count = {len(inf)}")
    rng = np.random.default_rng(0)
    for i in rng.choice(inf, size=min(8, len(inf)), replace=False):
        print("    ~", texts[i][:92])

    # ---- do the two 绷住 clusters really differ? compare centroid similarity
    c0 = np.where(lab == 0)[0]
    c5 = np.where(lab == 5)[0]
    c2 = np.where(lab == 2)[0]
    def centroid(idx):
        v = Z[idx].mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)
    print("\n=== centroid cosine between the 绷住 clusters ===")
    for a, b, na, nb in ((c0, c5, "c0", "c5"), (c0, c2, "c0", "c2"),
                         (c5, c2, "c5", "c2")):
        print(f"  {na} vs {nb}: {float(centroid(a) @ centroid(b)):.4f}")

    # ---- refit k=5 without the residual: does it reproduce the 4 forms + 2?
    keep = np.setdiff1d(np.arange(len(ids)), c6)
    lab5 = KMeans(n_clusters=5, n_init=50,
                  random_state=C.RANDOM_SEED).fit_predict(Z[keep])
    print(f"\n=== refit k=5 on {len(keep)} items (residual removed) ===")
    tk = [texts[i] for i in keep]
    for c in range(5):
        idx = np.where(lab5 == c)[0]
        forms = collections.Counter()
        for i in idx:
            t = tk[i]
            for f in ("绷不住", "绷带", "难绷", "绷紧", "紧绷"):
                if f in t:
                    forms[f] += 1
                    break
            else:
                if "绷住" in t:
                    forms["绷住"] += 1
                else:
                    forms["其他"] += 1
        L = [len(tk[i]) for i in idx]
        print(f"  cluster {c}: n={len(idx):5d} mean_len={np.mean(L):5.1f} | "
              + ", ".join(f"{k}×{v}" for k, v in forms.most_common(4)))
        # does this cluster match c0 / c5 / c2?
        ids_c = keep[idx]
        for name, ref in (("c0", c0), ("c5", c5), ("c2", c2), ("c4", np.where(lab == 4)[0])):
            ov = len(set(ids_c.tolist()) & set(ref.tolist()))
            if ov > 100:
                print(f"      overlaps {name} by {ov} "
                      f"({100*ov/len(idx):.0f}% of this cluster)")


if __name__ == "__main__":
    main()
