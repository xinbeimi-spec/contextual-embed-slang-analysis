# -*- coding: utf-8 -*-
"""Sanity check: did centering destroy the class structure?

KEEP THIS IN PERSPECTIVE: the "classes" here are SURFACE hints (绷带 / 绷紧 /
绷不住 / 绷绷炸弹), i.e. lexical form, not meaning. This check answers only
"is the geometry still sane after centering", it is NOT an RQ3 result.
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

EMB = C.EMB_DIR
CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_raw_repr():
    index = rj(os.path.join(EMB, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(EMB, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)
    return X[:, 9:13, :].mean(axis=1), ids


def cos_mat_vs_proto(Z, y, classes):
    out = {}
    for c in classes:
        m = y == c
        if m.sum() < 5:
            continue
        proto = Z[m].mean(0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        Zc = Z[m] / (np.linalg.norm(Z[m], axis=1, keepdims=True) + 1e-9)
        out[c] = float(np.mean(Zc @ proto))
    return out


def main():
    Xr, ids = load_raw_repr()
    Z = np.load(os.path.join(EMB, "centered.npz"))
    Zc, zids = Z["X"], list(Z["ids"])
    assert ids == zids, "id order mismatch"
    print(f"N={len(ids)}")

    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    y = np.array([meta[i].get("sense_hint", "unknown") for i in ids])
    suff = np.array([meta[i].get("context_sufficiency", 0) for i in ids])
    classes = [c for c, k in collections.Counter(y).items()
               if k >= 30 and c != "unknown"]
    sel = np.isin(y, classes) & (suff == 1)
    print(f"subset: {sel.sum()} items, classes={classes}")

    from sklearn.metrics import silhouette_score
    for tag, M in (("raw", Xr), ("centered", Zc)):
        sub = M[sel]
        s = silhouette_score(sub, y[sel], metric="cosine",
                             sample_size=min(1000, len(sub)),
                             random_state=C.RANDOM_SEED)
        intra = cos_mat_vs_proto(sub, y[sel], classes)
        print(f"\n[{tag}]  silhouette(cosine) = {s:.4f}")
        for c, v in sorted(intra.items(), key=lambda kv: -kv[1]):
            print(f"    intra-cosine {c:18} {v:.4f}")
        # inter: mean cosine between class prototypes
        protos = {}
        for c in classes:
            m = y[sel] == c
            if m.sum() >= 5:
                p = sub[m].mean(0)
                protos[c] = p / (np.linalg.norm(p) + 1e-9)
        names = list(protos)
        vals = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                vals.append(float(protos[names[i]] @ protos[names[j]]))
        print(f"    inter-proto cosine: mean={np.mean(vals):.4f} "
              f"max={np.max(vals):.4f}")


if __name__ == "__main__":
    main()
