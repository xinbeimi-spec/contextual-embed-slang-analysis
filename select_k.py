# -*- coding: utf-8 -*-
"""Stage 4b-alt: how many senses does the data actually support?

Compares two representations on identical items
  UNMASKED : mean over the target-span tokens   (encodes the surface form)
  MASKED   : hidden state at a [MASK] replacing the target (context only)

and answers "how many clusters" with THREE independent criteria instead of one:

  1. silhouette (cosine)          -- internal cohesion/separation
  2. bootstrap stability (mean ARI over subsamples) -- does k reproduce?
     This is the criterion a reader cannot accuse of being hand-tuned: if a
     partition does not reappear under resampling, the "number of senses" is
     not a property of the data.
  3. form entropy per cluster     -- DIAGNOSTIC OF CIRCULARITY.
     Clusters that merely track spelling are form-pure (low entropy); clusters
     that track meaning mix the forms (high entropy). This is what tells us
     whether a representation has escaped the surface-form confound.

Output
  results/tables/select_k_comparison.csv
  results/tables/select_k_forms_masked.csv
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
LAYERS = (9, 10, 11, 12)
FORMS = ["绷不住", "绷带", "绷紧", "紧绷", "绷直", "难绷", "绷绷炸弹", "绷住"]


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_repr(root, name):
    index = rj(os.path.join(root, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(root, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    print(f"{name}: N={X.shape[0]} D={X.shape[1]}")
    return X, ids


def prep(X, abtt=1):
    mu = X.mean(axis=0, keepdims=True)
    Z = X - mu
    if abtt > 0:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        comp = Vt[:abtt]
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def form_of(t):
    for f in FORMS:
        if f in t:
            return f
    return "other"


def entropy(cnt):
    p = np.array(list(cnt.values()), dtype=float)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def analyse(Z, ids, texts, k_max=10, boots=8, seed=C.RANDOM_SEED):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        lab = km.fit_predict(Z)
        sil = silhouette_score(Z, lab, metric="cosine",
                               sample_size=min(1500, n), random_state=seed)
        # bootstrap stability
        aris = []
        for _ in range(boots):
            idx = rng.choice(n, size=int(0.8 * n), replace=False)
            l1 = KMeans(n_clusters=k, n_init=5,
                        random_state=seed).fit_predict(Z[idx])
            l2 = KMeans(n_clusters=k, n_init=5,
                        random_state=seed + 1).fit_predict(Z[idx])
            aris.append(adjusted_rand_score(l1, l2))
        # form entropy
        per_cluster = []
        for c in range(k):
            m = lab == c
            if m.sum() < 5:
                continue
            cnt = collections.Counter(form_of(texts[i])
                                      for i in np.where(m)[0])
            per_cluster.append(entropy(cnt))
        rows.append({"k": k, "silhouette": round(float(sil), 4),
                     "stability_ari": round(float(np.mean(aris)), 4),
                     "form_entropy": round(float(np.mean(per_cluster)), 4)
                     if per_cluster else 0.0,
                     "min_cluster": int(min(np.bincount(lab)))})
        print(f"  k={k:2d}  sil={sil:.4f}  stab={np.mean(aris):.4f}  "
              f"formH={rows[-1]['form_entropy']:.3f}  "
              f"min_cluster={rows[-1]['min_cluster']}")
    return rows, lab


def main():
    rows_meta = {r["clean_id"]: r for r in rj(CLEAN)}
    Xu, ids_u = load_repr(C.EMB_DIR, "unmasked")
    Xm, ids_m = load_repr(C.EMB_MASK_DIR, "masked")
    assert ids_u == ids_m, "id order mismatch between the two representations"
    ids = ids_u
    texts = [rows_meta[i].get("text_norm", "") for i in ids]

    import csv
    all_rows = []
    for tag, X in (("unmasked", Xu), ("masked", Xm)):
        Z = prep(X, abtt=1)
        print(f"\n=== {tag} (mean-centred + 1 PC removed, L2-normalised) ===")
        rows, lab = analyse(Z, ids, texts)
        for r in rows:
            r["representation"] = tag
            all_rows.append(r)

    p = os.path.join(C.TABLES, "select_k_comparison.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["representation", "k", "silhouette",
                                          "stability_ari", "form_entropy",
                                          "min_cluster"])
        w.writeheader()
        w.writerows(all_rows)
    print("\n->", p)

    print("\n=== summary: which k wins on each criterion ===")
    for tag in ("unmasked", "masked"):
        sub = [r for r in all_rows if r["representation"] == tag]
        b_stab = max(sub, key=lambda r: r["stability_ari"])
        b_sil = max(sub, key=lambda r: r["silhouette"])
        b_form = max(sub, key=lambda r: r["form_entropy"])
        print(f"  {tag:9} best stability k={b_stab['k']} "
              f"({b_stab['stability_ari']:.3f}) | "
              f"best silhouette k={b_sil['k']} ({b_sil['silhouette']:.3f}) | "
              f"max form-entropy k={b_form['k']} ({b_form['form_entropy']:.3f})")


if __name__ == "__main__":
    main()
