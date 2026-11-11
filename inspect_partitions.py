# -*- coding: utf-8 -*-
"""Inspect the two candidate partitions (k=2 and k=6) on the MASKED space,
so the choice between them rests on interpretability rather than on a metric.

Prints, per cluster: size, top target-forms, mean length, sample sentences.
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
    return np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1), ids


def prep(X, abtt=1):
    Z = X - X.mean(axis=0, keepdims=True)
    if abtt:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        comp = Vt[:abtt]
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def forms_of(text):
    out = []
    for w in jieba.lcut(text):
        if C.TARGET_CHAR in w:
            out.append(w)
    for i, ch in enumerate(text):
        if ch == C.TARGET_CHAR:
            for L in (1, 2):
                if i + L + 1 <= len(text):
                    out.append(text[i:i + L + 1])
    return out or ["绷"]


CORE = ["绷不住", "绷带", "绷紧", "紧绷", "绷直", "难绷", "绷绷炸弹",
        "绷铁", "破绷", "绷住"]


def form_bucket(text):
    for f in CORE:
        if f in text:
            return f
    return "其他单绷"


def main():
    X, ids = load()
    Z = prep(X)
    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    texts = [meta[i].get("text_norm", "") for i in ids]

    from sklearn.cluster import KMeans
    for k in (2, 6):
        lab = KMeans(n_clusters=k, n_init=20,
                     random_state=C.RANDOM_SEED).fit_predict(Z)
        sizes = collections.Counter(lab.tolist())
        print(f"\n{'='*78}\n=== masked representation, k={k} ===")
        for c in sorted(sizes):
            idx = np.where(lab == c)[0]
            b = collections.Counter(form_bucket(texts[i]) for i in idx)
            top = ", ".join(f"{f}×{n}" for f, n in b.most_common(4))
            lens = np.mean([len(texts[i]) for i in idx])
            print(f"\n[cluster {c}]  n={sizes[c]}  mean_len={lens:.1f}")
            print(f"   forms: {top}")
            # longest 3 as representative, plus 2 random mid-length
            order = sorted(idx, key=lambda i: -len(texts[i]))
            for i in order[:3]:
                print("    ·", texts[i][:88])
            rng = np.random.default_rng(0)
            mids = [i for i in idx if 20 <= len(texts[i]) <= 40]
            if mids:
                for i in rng.choice(mids, size=min(2, len(mids)), replace=False):
                    print("    ~", texts[i][:88])


if __name__ == "__main__":
    main()
