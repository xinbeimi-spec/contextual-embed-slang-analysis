# -*- coding: utf-8 -*-
"""The clustering figure for the typology (file name is historical).

Produces the 2-D semantic-space scatter coloured by the FINAL categories
(物理义 / 忍住义 / 忍不住义 / 评价义 / 紧绷义 / 非义项用法), with category
centroids marked. Since 2026-10-21 there are SIX categories: 紧绷义 was added,
so the output file is now fig1_six_category_space.* and the five-category
figure lives in results/final_5cat/ as the archived version.

Notes that go into the caption
  * the projection is PCA on the SAME centred, ABTT-filtered space in which the
    (masked) clustering was performed, so the picture is not a different space
    from the analysis;
  * PC1/PC2 explain only ~28% of the variance -- that is exactly why clustering
    was NOT done in 2-D. The plot is a view, not the geometry;
  * 紧绷义 is a small category (n in the tens), so its points are drawn larger
    than the rest or they would be invisible.

Output
  results/figures/fig1_six_category_space.png/.pdf
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

ROOT = C.EMB_MASK_DIR
TYPOLOGY = os.path.join(C.PROCESSED, "beng_typology.jsonl")
LAYERS = (9, 10, 11, 12)
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
CAT_COLOR = {"物理义": "#4C72B0", "忍住义": "#DD8452",
             "忍不住义": "#C44E52", "评价义": "#55A868",
             "紧绷义": "#8172B3", "非义项用法": "#9E9E9E"}
CAT_SIZE = {"紧绷义": 34}


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_repr():
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


def main():
    X, ids = load_repr()
    typ = {r["clean_id"]: r for r in rj(TYPOLOGY)}
    cat = {r["clean_id"]: r["category"] for r in rj(TYPOLOGY)}
    excluded = {i for i, r in typ.items() if r.get("excluded_from_typology")}
    y = np.array([cat.get(i, "非义项用法") for i in ids])
    keep = np.array([i not in excluded for i in ids])
    print("items:", len(ids), "category counts:",
          dict(collections.Counter(y.tolist())))
    print(f"excluded_from_typology items drawn grey but not scored: "
          f"{len(ids) - int(keep.sum())}")

    # same transformation chain as the clustering
    Z = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    comp = Vt[:1]
    Zc = Z - (Z @ comp.T) @ comp                       # ABTT k=1
    ev = (S[:60] ** 2) / (len(Z) - 1)
    ratio = ev / ev.sum()
    Zn = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12)

    # 2-D projection of the SAME space
    _, _, Vt2 = np.linalg.svd(Zn - Zn.mean(0, keepdims=True), full_matrices=False)
    coords = (Zn - Zn.mean(0, keepdims=True)) @ Vt2[:2].T
    var2 = float(np.var(coords, axis=0).sum() / np.var(Zn, axis=0).sum())
    print(f"2-D projection explains {100*var2:.1f}% of the variance of the "
          f"clustering space")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    for c in CAT_ORDER:
        m = (y == c) & keep
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1],
                   s=CAT_SIZE.get(c, 7), alpha=0.55,
                   color=CAT_COLOR[c], edgecolors="none",
                   label=f"{c} (n={int(m.sum())})")
    # centroids
    for c in CAT_ORDER:
        m = (y == c) & keep
        if not m.any():
            continue
        cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
        ax.scatter([cx], [cy], s=190, marker="X", color=CAT_COLOR[c],
                   edgecolors="black", linewidths=0.9, zorder=5)
        ax.annotate(c, (cx, cy), textcoords="offset points", xytext=(9, 8),
                    fontsize=9, fontweight="bold", color="black", zorder=6)
    ax.set_xlabel("投影维度 1")
    ax.set_ylabel("投影维度 2")
    ax.set_title("「绷」六类用法在语义空间中的分布")
    ax.legend(fontsize=8, frameon=False, markerscale=3.0,
              loc="best")
    ax.text(0.01, 0.01,
            f"注：聚类在 768 维中心化空间完成，此二维投影仅解释其方差的 "
            f"{100*var2:.1f}%，仅用于展示。",
            transform=ax.transAxes, fontsize=7.5, color="#555555",
            va="bottom")
    fig.tight_layout()

    os.makedirs(C.FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(C.FIGURES, f"fig1_six_category_space.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("->", p)


if __name__ == "__main__":
    main()
