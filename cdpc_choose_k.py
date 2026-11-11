# -*- coding: utf-8 -*-
"""Why k=7 for the CDPC clustering? A documented k sweep.

The automatic rule inside cluster_cdpc.choose_k (largest multiplicative drop in
sorted gamma, restricted to peaks above mean+1sd) returns k=4 on this corpus.
k=4 is too coarse here: one basin absorbs 绷带 together with 绷不住, so 物理义
and 非义项用法 disappear from the typology entirely (purity of that cluster 0.53).
That is a property of the density landscape (two big basins dominate), not a
property of the data having four senses.

This script recomputes the same {rho, delta, gamma, parent} once and evaluates
k = 4..10 on three criteria that do NOT use the category labels:

  * weighted cluster-level form purity  — how well clusters separate surface forms
  * silhouette (cosine)                 — geometric separation
  * category coverage                   — how many of the six senses survive
  * min basin size                      — no cluster should be a single point

Output: results/result4/cdpc_k_choice.csv
Run:    python src/analyze/cdpc_choose_k.py
"""
import os
import sys
import csv
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, assign, kmedoids_refine
from cdpc_pipeline import load_masked, form_label, item_category
from jinbeng_cue import load_adjudication

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
OUT = os.path.join(C.TABLES, "cdpc_k_choice.csv")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    Z, ids = load_masked()
    rows = {r["clean_id"]: r for r in rj(CLEAN)}
    jinbeng = load_adjudication()
    N = len(ids)
    forms = [form_label(rows[c].get("text_norm") or "") for c in ids]
    cats = [item_category(rows[c].get("text_norm") or "", c, jinbeng) for c in ids]
    print(f"items={N}")

    S = Z @ Z.T
    rho, delta, gamma, parent, dc = density_peaks(S, 2.0)
    print(f"dc={dc:.4f}  (no k is fixed in the density computation)")
    # the automatic rule (largest multiplicative drop in sorted gamma)
    from cluster_cdpc import choose_k
    k_auto, _r, _g = choose_k(gamma, 0)
    print(f"automatic rule -> k={k_auto}")

    from sklearn.metrics import silhouette_score
    out = []
    print(f"\n{'k':>3}{'form_purity':>13}{'silhouette':>12}{'n_categories':>14}"
          f"{'min_basin':>11}  categories present")
    for k in range(3, 13):
        centers = np.argsort(-gamma)[:k]
        lab = assign(parent, centers, rho)
        lab, med = kmedoids_refine(Z, lab, iters=4)
        sizes = collections.Counter(lab.tolist())
        # cluster -> majority form (label-free proxy for "did the split work")
        tot = 0
        for c in sizes:
            sub = collections.Counter(f for f, l in zip(forms, lab.tolist()) if l == c)
            tot += sub.most_common(1)[0][1]
        purity = tot / N
        sil = silhouette_score(Z, lab, metric="cosine",
                               sample_size=min(1500, N),
                               random_state=C.RANDOM_SEED)
        # category coverage (uses the surface-form mapping only to name clusters)
        cl_cat = {}
        for c in sizes:
            sub = collections.Counter(cc for cc, l in zip(cats, lab.tolist()) if l == c)
            cl_cat[c] = sub.most_common(1)[0][0]
        present = sorted(set(cl_cat.values()))
        out.append({"k": k, "form_purity": round(float(purity), 4),
                    "silhouette": round(float(sil), 4),
                    "n_categories": len(present),
                    "min_basin": int(min(sizes.values())),
                    "categories": "/".join(present),
                    "chosen": 1 if k == k_auto else 0})
        print(f"{k:>3}{purity:>13.4f}{sil:>12.4f}{len(present):>14}"
              f"{min(sizes.values()):>11}  {'/'.join(present)}")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["k", "form_purity", "silhouette",
                                          "n_categories", "min_basin",
                                          "categories", "chosen"])
        w.writeheader()
        w.writerows(out)
    print(f"\n-> {OUT}")
    print("\n如何读这张表（第一遍，先看结构再定 k）：")
    print(f"  * 自动 gamma 规则给出 k={k_auto}，它只保证峰分断层最大，不保证类别齐全；")
    print("  * form_purity = 每个簇的主导词形占比的加权和，越高说明簇越贴词形；")
    print("  * min_basin 是「最小盆地的条目数」，个位数就是碎片簇，不能拿来当义项；")
    print("  * silhouette 只作参考：它奖励「合并成一团」，不能用来选 k；")
    print("  * n_categories 用的是词形→类别的临时映射（塌房/物理/下垂/成语/安定），"
          "只用来读第一遍结构，不是最终义项表。")


if __name__ == "__main__":
    main()
