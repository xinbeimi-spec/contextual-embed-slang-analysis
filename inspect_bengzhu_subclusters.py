# -*- coding: utf-8 -*-
"""What are the three 绷住 sub-clusters (619 / 242 / 155)?

All three are ~100% 绷住 by form, so their difference is NOT orthographic.
Dump their content to characterise them.
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


def main():
    X, ids = load()
    Z = prep(X)
    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    texts = [meta[i].get("text_norm", "") for i in ids]

    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=7, n_init=30,
                 random_state=C.RANDOM_SEED).fit_predict(Z)

    # clusters that are >=95% 绷住
    for c in range(7):
        idx = np.where(lab == c)[0]
        n_bz = sum(1 for i in idx if "绷住" in texts[i] and "绷不住" not in texts[i])
        if n_bz / max(1, len(idx)) < 0.9:
            continue
        print(f"\n{'='*74}\n[cluster {c}]  n={len(idx)}  绷住占比={100*n_bz/len(idx):.1f}%")
        lens = [len(texts[i]) for i in idx]
        print(f"  平均长度={np.mean(lens):.1f}  中位数={np.median(lens):.0f}")
        # high-frequency neighbouring words (excluding target)
        cnt = collections.Counter()
        for i in idx:
            for w in jieba.lcut(texts[i]):
                if len(w) >= 2 and C.TARGET_CHAR not in w:
                    cnt[w] += 1
        print("  高频共现词:", ", ".join(f"{w}×{n}" for w, n in cnt.most_common(12)))
        order = sorted(idx, key=lambda i: -len(texts[i]))
        print("  最长例句:")
        for i in order[:3]:
            print("    ·", texts[i][:92])
        mids = [i for i in idx if 15 <= len(texts[i]) <= 30]
        if mids:
            rng = np.random.default_rng(1)
            print("  中等长度例句:")
            for i in rng.choice(mids, size=min(4, len(mids)), replace=False):
                print("    ~", texts[i][:92])


if __name__ == "__main__":
    main()
