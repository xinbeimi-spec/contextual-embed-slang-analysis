# -*- coding: utf-8 -*-
"""Stage 3c: layer probe + sanity checks.

Two questions:
  (1) which of the 12 layers best separates the surface usage classes?
      -> gives an evidence-based layer choice instead of "last layer by default"
  (2) does the vector space behave at all? cheap sanity checks before clustering.

CAVEAT that must appear in the paper: labels here are `sense_hint`, i.e. derived
from surface cues, NOT human annotation. This probe therefore validates the
pipeline and picks a layer; the reported "representation ability" figure for RQ3
must be recomputed on gold labels from the pilot annotation.

Outputs
  results/tables/layer_probe.csv
  results/tables/embed_sanity.csv
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

EMB_DIR = C.EMB_DIR
SHARD_DIR = os.path.join(EMB_DIR, "shards")
CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")

MIN_CLASS = 30


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_all():
    index = read_jsonl(os.path.join(EMB_DIR, "index.jsonl"))
    meta = {r["clean_id"]: r for r in read_jsonl(CLEAN)}
    by_shard = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by_shard[r["shard"]].append(r)

    xs, ids = [], []
    for si in sorted(by_shard):
        p = os.path.join(SHARD_DIR, f"shard_{si:03d}.npz")
        emb = np.load(p)["emb"]                      # (n, L+1, H) float16
        for r in sorted(by_shard[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)              # (N, L+1, H)
    return X, ids, meta


def main():
    X, ids, meta = load_all()
    n, L1, H = X.shape
    print(f"loaded embeddings: N={n} layers={L1} dim={H}")

    hints = np.array([meta[i].get("sense_hint", "unknown") for i in ids])
    years = np.array([(meta[i].get("ctime_iso") or "")[:4] for i in ids])
    suffic = np.array([meta[i].get("context_sufficiency", 0) for i in ids])

    print("\nclass counts:", dict(collections.Counter(hints)))

    # ---- keep classes with enough members, judgeable items only
    keep_cls = [c for c, k in collections.Counter(hints).items()
                if k >= MIN_CLASS and c != "unknown"]
    sel = np.isin(hints, keep_cls) & (suffic == 1)
    print(f"probe subset: {sel.sum()} items, classes={keep_cls}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rows = []
    Xs = X[sel]
    ys = hints[sel]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    for li in range(L1):
        Xi = Xs[:, li, :]
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=2000, C=1.0, random_state=C.RANDOM_SEED),
        )
        acc = cross_val_score(clf, Xi, ys, cv=cv, scoring="accuracy")
        f1 = cross_val_score(clf, Xi, ys, cv=cv, scoring="f1_macro")
        rows.append({"layer": li, "acc_mean": acc.mean(), "acc_std": acc.std(),
                     "f1_macro_mean": f1.mean(), "f1_macro_std": f1.std()})
        print(f"  layer {li:2d}  acc={acc.mean():.4f}±{acc.std():.4f}  "
              f"macroF1={f1.mean():.4f}")

    import csv
    p1 = os.path.join(C.TABLES, "layer_probe.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["f1_macro_mean"])
    print(f"\nbest layer by macro-F1: {best['layer']} "
          f"(f1={best['f1_macro_mean']:.4f})")
    print("->", p1)

    # ---- sanity checks on the last-4-mean representation
    last4 = X[:, -4:, :].mean(axis=1)
    def cos(a, b):
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(a @ b)

    # prototype per class, then mean intra / inter cosine
    protos = {}
    for c in keep_cls:
        m = (hints == c) & (suffic == 1)
        if m.sum() >= 5:
            protos[c] = last4[m].mean(axis=0)
    intra = {}
    for c in protos:
        m = (hints == c) & (suffic == 1)
        intra[c] = float(np.mean([cos(v, protos[c]) for v in last4[m]]))
    names = list(protos)
    inter = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter[f"{names[i]}|{names[j]}"] = cos(protos[names[i]], protos[names[j]])

    print("\nintra-class mean cosine (higher = tighter):")
    for c, v in sorted(intra.items(), key=lambda kv: -kv[1]):
        print(f"  {c:18} {v:.4f}")
    print("\ninter-class prototype cosine (lower = more separable):")
    for k, v in sorted(inter.items(), key=lambda kv: -kv[1]):
        print(f"  {k:38} {v:.4f}")

    p2 = os.path.join(C.TABLES, "embed_sanity.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "pair_or_class", "value"])
        for k, v in intra.items():
            w.writerow(["intra_cosine", k, round(v, 4)])
        for k, v in inter.items():
            w.writerow(["inter_cosine", k, round(v, 4)])
        w.writerow(["n_items", "-", n])
        w.writerow(["probe_items", "-", int(sel.sum())])
        w.writerow(["best_layer", "-", best["layer"]])
    print("\n->", p2)


if __name__ == "__main__":
    main()
