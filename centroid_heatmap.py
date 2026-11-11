# -*- coding: utf-8 -*-
"""Category-centroid similarity heatmap (the quantitative backbone of 图1).

Why this exists
  Pairwise similarity between two hand-picked sentences turned out to be a
  dangerous measure: v1's "same-sense" pair scored 0.537 mostly because the two
  sentences share a meme construction, not because they share a sense
  (see HANDOVER 3.5). A centroid is far more stable than a single sentence: it
  averages over every item in the category, so template repetition inside one
  category cannot drive the result.

What is computed
  * vectors: masked representation ([MASK] replaces 绷), last-4 layers mean
  * transform: mean + All-but-the-Top (k=1), then L2 normalise -- exactly the
    space the k=7 clustering was computed in, refitted on the SAME embedded
    item set so the numbers are comparable to the clustering
  * centroid per category = mean of the (normalised) item vectors
  * cosine between centroids, plus each category's mean internal cosine
    (cohesion) -- a category whose cohesion is near 0 is not a category
  * items flagged `excluded_from_typology` are left out, like everywhere else

Outputs
  results/tables/category_centroid_cosine.csv
  results/figures/fig3_category_centroid_heatmap.png/.pdf

Caveat that must appear in the caption: the vectors are centred, so absolute
values are not interpretable; only the ordering / contrast is.
"""
import os
import sys
import json
import csv
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


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_masked():
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
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    Z = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    comp = Vt[:1]
    Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12), ids


def main():
    Z, ids = load_masked()
    typ = {r["clean_id"]: r for r in rj(TYPOLOGY)}
    pos = {cid: i for i, cid in enumerate(ids)}
    print(f"embedded items: {len(ids)}  typology items: {len(typ)}")

    members = collections.defaultdict(list)
    for cid, r in typ.items():
        if r.get("excluded_from_typology"):
            continue
        cat = r.get("category")
        if cat in CAT_ORDER and cid in pos:
            members[cat].append(pos[cid])

    cents, names, ns = [], [], []
    for c in CAT_ORDER:
        idx = members.get(c) or []
        if not idx:
            print(f"  ! no items for {c}")
            continue
        V = Z[idx]
        cent = V.mean(axis=0)
        cents.append(cent / (np.linalg.norm(cent) + 1e-12))
        names.append(c)
        ns.append(len(idx))
        within = float((V @ (cent / (np.linalg.norm(cent) + 1e-12))).mean())
        print(f"  {c:8} n={len(idx):5d}  mean cosine to centroid = {within:+.3f}")

    S = np.stack(cents) @ np.stack(cents).T
    p = os.path.join(C.TABLES, "category_centroid_cosine.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "n"] + names)
        for i, c in enumerate(names):
            w.writerow([c, ns[i]] + [round(float(S[i, j]), 4)
                                     for j in range(len(names))])

    off = [S[i, j] for i in range(len(names)) for j in range(len(names))
           if i != j]
    print("\n=== centroid cosine matrix ===")
    print(f"{'':10}" + "".join(f"{c:>10}" for c in names))
    for i, c in enumerate(names):
        print(f"{c:10}" + "".join(f"{S[i, j]:>10.3f}" for j in range(len(names))))
    print(f"\nbetween-category mean cosine: {np.mean(off):+.3f}")

    # plain-text evidence for the write-up: which pair is most / least similar
    pairs = sorted(((S[i, j], names[i], names[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)
    print("\nmost similar pairs:")
    for v, a, b in pairs[:3]:
        print(f"  {a} ~ {b}: {v:+.3f}")
    print("least similar pairs:")
    for v, a, b in pairs[-3:]:
        print(f"  {a} ~ {b}: {v:+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(names, ns)], fontsize=8)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(names, ns)], fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(S[i, j]) > 0.55 else "black")
    ax.set_title("「绷」六类质心的余弦相似度")
    fig.colorbar(im, ax=ax, shrink=0.82, label="cosine")
    ax.text(0.5, -0.22,
            "注：向量已去均值并移除第一主成分，故绝对数值不可作阈值，只有相对高低有意义。",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    os.makedirs(C.FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"fig3_category_centroid_heatmap.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    print("->", p)


if __name__ == "__main__":
    main()
