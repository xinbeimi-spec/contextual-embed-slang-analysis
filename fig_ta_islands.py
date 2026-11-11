# -*- coding: utf-8 -*-
"""Density islands of the 塌 space (HDBSCAN) drawn on the t-SNE view.

Why this figure exists (2026-10-31): the user's reading of the PCA scatter was
"the points look dispersed, it did not cluster". The diagnostics agree that the
global geometry is a CONTINUUM (silhouette rises monotonically with k, only two
real gamma peaks), but they also show sharp DENSITY ISLANDS whose membership is
almost perfectly lexical: 塌房 530/530, 坍塌 278/278, 塌陷 131/130, 一塌糊涂
96/102. CDPC with k=5 cannot show them, because k peaks are pulled into the two
big masses; HDBSCAN finds the islands and labels the rest as noise, which is the
honest picture.

Outputs
  results/<slug>/<round>/cdpc_fig1c_islands.png/.pdf
  results/<slug>/<round>/hdbscan_islands.csv
  logs/<slug>/hdbscan_islands.txt
Run: python src/analyze/fig_ta_islands.py
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
from cdpc_pipeline import load_masked, form_label, item_category, CAT_COLOR

sys.stdout.reconfigure(encoding="utf-8")

OUT_TXT = os.path.join(C.LOGS, "hdbscan_islands.txt")
OUT_CSV = os.path.join(C.TABLES, "hdbscan_islands.csv")
FIG = os.path.join(C.FIGURES, "cdpc_fig1c_islands")
COORDS = os.path.join(C.FIGURES, "tsne_coords.npz")
MIN_SIZE = 30


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    Z, ids = load_masked()
    rows = {r["clean_id"]: r for r in rj(
        os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl"))}
    forms = [form_label(rows[c]["text_norm"]) for c in ids]
    cats = [item_category(rows[c]["text_norm"], c, {}) for c in ids]

    h = HDBSCAN(min_cluster_size=MIN_SIZE, metric="euclidean",
                cluster_selection_method="eom", copy=True)
    lab = h.fit_predict(Z)
    islands = [c for c in sorted(set(lab.tolist())) if c != -1]
    print(f"islands={len(islands)} noise={int((lab == -1).sum())}")

    # t-SNE once, cached (it is the slow part)
    if os.path.exists(COORDS):
        ts = np.load(COORDS)["ts"]
        print("t-SNE coords loaded from cache")
    else:
        Zp = PCA(n_components=50, random_state=C.RANDOM_SEED).fit_transform(Z)
        ts = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=C.RANDOM_SEED, metric="cosine").fit_transform(Zp)
        np.savez_compressed(COORDS, ts=ts)
        print("t-SNE computed")

    # table
    recs, L = [], [f"items={len(ids)}  islands={len(islands)}  "
                   f"noise={int((lab == -1).sum())} "
                   f"({100*(lab == -1).sum()/len(ids):.1f}%)", ""]
    for c in islands:
        idx = np.where(lab == c)[0]
        if len(idx) < 20:
            continue
        fc = collections.Counter(forms[i] for i in idx)
        cc = collections.Counter(cats[i] for i in idx)
        top_f, n_f = fc.most_common(1)[0]
        top_c, n_c = cc.most_common(1)[0]
        recs.append({"island": int(c), "size": len(idx),
                     "form": top_f, "form_share": round(n_f / len(idx), 3),
                     "category": top_c, "category_share": round(n_c / len(idx), 3),
                     "forms": "|".join(f"{k}:{v}" for k, v in fc.most_common(3))})
    recs.sort(key=lambda r: -r["size"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    for r in recs:
        L.append(f"  island {r['island']:>2} n={r['size']:>4}  form={r['form']}"
                 f"({r['form_share']:.2f})  cat={r['category']}"
                 f"({r['category_share']:.2f})")
    open(OUT_TXT, "w", encoding="utf-8").write("\n".join(L))

    # figure: islands coloured, continuum grey
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    palette = ["#E7721B", "#C0392B", "#2C6FB5", "#2E8B57", "#7B52A1",
               "#B8860B", "#D35400", "#1F6F8B", "#8E44AD", "#C0392B",
               "#16A085", "#A04000"]
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    noise = lab == -1
    ax.scatter(ts[noise, 0], ts[noise, 1], s=5, c="#DDDDDD", alpha=0.7,
               edgecolors="none", label=f"连续体/未成岛 (n={int(noise.sum())})")
    for j, r in enumerate(recs):
        c = r["island"]
        idx = lab == c
        ax.scatter(ts[idx, 0], ts[idx, 1], s=9, alpha=0.85,
                   color=palette[j % len(palette)], edgecolors="none",
                   label=f"岛{r['island']} {r['form']} (n={r['size']})")
    ax.set_title(f"「{C.TARGET_WORD}」密度岛（HDBSCAN, min_cluster_size={MIN_SIZE}）"
                 f"—— 岛由词形构成，而非义项")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, loc="best", markerscale=2)
    ax.text(0.01, 0.01, "注：t-SNE 仅作二维展示；点色=密度岛，灰=未成岛的连续体。",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=300, bbox_inches="tight")
    print("\n".join(L))
    print("->", f"{FIG}.png")
    print("->", OUT_CSV)


if __name__ == "__main__":
    main()
