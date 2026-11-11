# -*- coding: utf-8 -*-
"""Can the geometry be sharpened into sense-level clusters?

Follow-up to diag_ta_cluster_tendency.py. t-SNE showed many SMALL islands whose
membership is topical (food / engineering / fandom / beauty), with the sense
categories cutting across them. Two levers are tested here:

  A. stronger anisotropy removal: the leading principal directions of comment
     embeddings are usually topic/style, not sense. ABTT with k=2,3,5,10 removes
     those directions before the cosine density peaks.
  B. form-controlled clustering: the sense question is asked WITHIN a form
     (e.g. inside 坍塌 items, is there more than one sense?), which removes the
     form-vs-form comparison that dominates the global geometry.

Output: logs/<slug>/diag_sharpen.txt
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, assign, kmedoids_refine
from cdpc_pipeline import load_masked, form_label, item_category

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(C.LOGS, "diag_sharpen.txt")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sil(Z, lab):
    from sklearn.metrics import silhouette_score
    if len(set(lab.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(Z, lab, metric="cosine",
                                 sample_size=min(1500, len(lab)),
                                 random_state=C.RANDOM_SEED))


def abtt(Zraw, k):
    Z = Zraw - Zraw.mean(axis=0, keepdims=True)
    if k > 0:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        comp = Vt[:k]
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def cluster(Z, k, dcp=2.0):
    S = Z @ Z.T
    rho, delta, gamma, parent, dc = density_peaks(S, dcp)
    lab = assign(parent, np.argsort(-gamma)[:k], rho)
    lab, _m = kmedoids_refine(Z, lab, iters=4)
    return lab


def main():
    Z1, ids = load_masked()          # already mean-centred + ABTT(1) + L2
    rows = {r["clean_id"]: r for r in rj(
        os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl"))}
    forms = [form_label(rows[c]["text_norm"]) for c in ids]
    cats = [item_category(rows[c]["text_norm"], c, {}) for c in ids]
    L = [f"items={len(ids)}", ""]

    # Z1 is already ABTT(1); reconstruct the un-centred matrix for higher k by
    # re-loading the shards through the same loader is expensive, so we
    # approximate by applying extra ABTT directions on top of Z1.
    L.append("=== A. 在现有表示上继续去掉主成分（k=5, dc=2%）===")
    L.append(f"{'extra_abtt':>11}{'silhouette':>12}{'form_purity':>13}")
    for extra in (0, 1, 2, 4, 9):
        Zx = abtt(Z1, extra)
        lab = cluster(Zx, 5)
        tot = 0
        for c in set(lab.tolist()):
            sub = collections.Counter(f for f, l in zip(forms, lab.tolist()) if l == c)
            tot += sub.most_common(1)[0][1]
        L.append(f"{extra:>11}{sil(Zx, lab):>12.4f}{tot/len(ids):>13.3f}")

    L.append("")
    L.append("=== B. 分词组内部聚类（form-controlled）===")
    L.append("  问题：每个词形内部，是否还能分出 >1 个语义团？"
             "（silhouette 高 = 该词形内部有细分结构）")
    L.append(f"{'form':<8}{'n':>6}{'k':>4}{'silhouette':>12}  sub-cluster top categories")
    for f in ("塌房", "坍塌", "倒塌", "塌方", "塌陷", "塌了", "其他"):
        idx = [i for i, x in enumerate(forms) if x == f]
        if len(idx) < 60:
            continue
        sub = Z1[idx]
        k = 3 if len(idx) >= 180 else 2
        lab = cluster(sub, k)
        catsub = collections.Counter(cats[i] for i, l in zip(idx, lab.tolist())
                                     if True)
        parts = []
        for c in sorted(set(lab.tolist())):
            cc = collections.Counter(cats[i] for i, l in zip(idx, lab.tolist()) if l == c)
            parts.append(f"S{c}(n={sum(cc.values())}):" + ",".join(
                f"{k2}:{v}" for k2, v in cc.most_common(2)))
        L.append(f"{f:<8}{len(idx):>6}{k:>4}{sil(sub, lab):>12.4f}  " + "  ".join(parts))

    txt = "\n".join(L)
    os.makedirs(C.LOGS, exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)
    print("\n->", OUT)


if __name__ == "__main__":
    main()
