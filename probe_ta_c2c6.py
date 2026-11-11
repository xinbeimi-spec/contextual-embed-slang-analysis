# -*- coding: utf-8 -*-
"""Probe: are result1's residual basins C2 ∪ C6 internally clusterable?

Motivation (user, 2026-10-31): in round 1 (k=8) clusters C2 (n=256, purity 0.38,
majority "成语义") and C6 (n=654, purity 0.63, majority "非义项用法") were both
mixed bags. The question is whether, taken TOGETHER, they contain sub-groups that
are semantically coherent enough to carry a category of their own.

Method: same CDPC maths as the main pipeline (cosine density peaks + k-medoids),
fitted on the 910 items of C2 ∪ C6 only, with the same centring as load_masked().
We sweep k=2..6 and report, per sub-cluster: size, form composition, dominant
surface cue, and 3 example texts. No labels are invented here; the output is a
reading aid.

Output: logs/<slug>/probe_c2c6.txt  (+ .csv in the current result round)
Run:    python src/analyze/probe_ta_c2c6.py
"""
import os
import sys
import json
import csv
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, choose_k, assign, kmedoids_refine
from cdpc_pipeline import load_masked, form_label

sys.stdout.reconfigure(encoding="utf-8")

ROUND1 = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_result1.jsonl")
OUT = os.path.join(C.LOGS, "probe_c2c6.txt")
CSV_OUT = os.path.join(C.TABLES, "probe_c2c6_clusters.csv")
CUE = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    rows = {r["clean_id"]: r for r in rj(CUE)}
    r1 = rj(ROUND1)
    sel = [r for r in r1 if r.get("cdpc_cluster") in (2, 6)]
    print(f"round1 C2+C6 items: {len(sel)}")
    keep_ids = [r["clean_id"] for r in sel]

    Z_all, ids_all = load_masked()
    pos = {cid: i for i, cid in enumerate(ids_all)}
    idx = [pos[c] for c in keep_ids if c in pos]
    missing = len(keep_ids) - len(idx)
    Z = Z_all[idx]
    ids = [ids_all[i] for i in idx]
    print(f"with vectors: {len(ids)} (missing {missing})")

    S = Z @ Z.T
    rho, delta, gamma, parent, dc = density_peaks(S, 2.0)
    k_auto, _r, _g = choose_k(gamma, 0)
    print(f"dc={dc:.4f}  auto k on this subset = {k_auto}")

    L = [f"round1 C2+C6 subset: n={len(ids)}  dc={dc:.4f}  auto k={k_auto}", ""]
    L.append("=== k sweep on the subset ===")
    L.append(f"{'k':>3}{'min_basin':>11}{'form_purity':>13}  top forms")
    best = {}
    for k in range(2, 7):
        centers = np.argsort(-gamma)[:k]
        lab = assign(parent, centers, rho)
        lab, med = kmedoids_refine(Z, lab, iters=4)
        sizes = collections.Counter(lab.tolist())
        forms = [form_label(rows[c]["text_norm"]) for c in ids]
        tot = 0
        for c in sizes:
            tot += collections.Counter(f for f, l in zip(forms, lab.tolist())
                                      if l == c).most_common(1)[0][1]
        purity = tot / len(ids)
        top = collections.Counter(forms).most_common(4)
        L.append(f"{k:>3}{min(sizes.values()):>11}{purity:>13.3f}  "
                 + " ".join(f"{f}:{n}" for f, n in top))
        best[k] = (lab, med, sizes, purity, forms)

    for k in (3, 4, 5):
        if k not in best:
            continue
        lab, med, sizes, purity, forms = best[k]
        L.append("")
        L.append(f"=== k={k} detail (purity {purity:.3f}) ===")
        recs = []
        for c in sorted(sizes):
            w = np.where(lab == c)[0]
            fc = collections.Counter(forms[i] for i in w)
            L.append(f"  [S{c}] n={sizes[c]}  top forms: "
                     + " ".join(f"{f}:{n}" for f, n in fc.most_common(5)))
            for i in w[:3]:
                L.append(f"        - {rows[ids[i]]['text_norm'][:70]}")
            recs.append({"k": k, "subcluster": int(c), "size": int(sizes[c]),
                         "share": round(sizes[c] / len(ids), 4),
                         "top_forms": "|".join(f"{f}:{n}" for f, n in
                                               fc.most_common(5))})
        if k == 4:
            with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)

    txt = "\n".join(L)
    os.makedirs(C.LOGS, exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)
    print("\n->", OUT)


if __name__ == "__main__":
    main()
