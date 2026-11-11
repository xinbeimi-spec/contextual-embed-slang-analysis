# -*- coding: utf-8 -*-
"""CDPC with the laughter layer MERGED into one pale background class.

2026-11-09, user decision: 笑声 is one class, not a set of clusters. It is drawn
in light blue *underneath* the other clusters (drawn first, small + pale), and
the CDPC peaks are computed on the remaining items only, so the other senses are
not crowded out by 55.7% of the corpus.

Outputs go to results/<slug>/result2 (run with $env:LING_RESULT="result2"):
    cdpc_ha_fig1_clusters.png/.pdf   scatter, laughter as pale layer
    cdpc_ha_fig4_stacked.png/.pdf    yearly composition (laughter at bottom)
    cdpc_ha_fig4b_line.png/.pdf      yearly share lines + n
    cdpc_cluster_sizes.csv, cdpc_cluster_x_form.csv
    data/processed/ha/ha_cdpc_result2.jsonl
"""
import os
import sys
import csv
import json
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, assign, kmedoids_refine
from cdpc_pipeline import load_masked
from cdpc_ha_diag import form_label, FORMS

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
LAUGH_COLOR = "#A6CEE3"          # light blue, the background layer
PALETTE = ["#D62728", "#FF7F0E", "#2CA02C", "#9467BD", "#8C564B",
           "#E377C2", "#7F7F7F", "#BCBD22"]


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="clusters for the non-laughter layer")
    ap.add_argument("--dc-pct", type=float, default=2.0)
    a = ap.parse_args()

    Z, ids = load_masked()
    rows = {r["clean_id"]: r for r in rj(CLEAN)}
    N = len(ids)
    forms = [form_label(rows[c].get("text_norm")) for c in ids]
    is_laugh = np.array([f == "笑声" for f in forms])

    Zi = Z[~is_laugh]
    idx_rest = np.where(~is_laugh)[0]
    fi = [forms[i] for i in idx_rest]
    print(f"total={N}  laughter(merged)={int(is_laugh.sum())}  "
          f"clustered={len(idx_rest)}  k={a.k}")

    S = Zi @ Zi.T
    rho, delta, gamma, parent, dc = density_peaks(S, a.dc_pct)
    centers = np.argsort(-gamma)[:a.k]
    lab_sub = assign(parent, centers, rho)
    lab_sub, _med = kmedoids_refine(Zi, lab_sub, iters=8)
    lab_sub = np.asarray(lab_sub)

    # order clusters by size, remap so C0 is the biggest
    sizes = collections.Counter(lab_sub.tolist())
    order = sorted(sizes, key=lambda c: -sizes[c])
    remap = {c: i for i, c in enumerate(order)}
    lab_sub = np.array([remap[x] for x in lab_sub])

    lab = np.full(N, a.k, dtype=int)          # label a.k = laughter
    lab[idx_rest] = lab_sub

    # ---- structure
    lines = [f"k={a.k} (+ merged laughter)  total={N}"]
    for c in range(a.k + 1):
        idx = [i for i in range(N) if lab[i] == c]
        if not idx:
            continue
        sub = collections.Counter(forms[i] for i in idx)
        tag = "笑声（合并）" if c == a.k else f"C{c}"
        pur = sub.most_common(1)[0][1] / len(idx)
        lines.append(f"{tag:8} n={len(idx):5d} ({len(idx)/N*100:4.1f}%)  "
                     f"purity={pur:.2f}  " +
                     " ".join(f"{k}:{v}" for k, v in sub.most_common(5)))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(C.LOGS, f"cdpc_merge_k{a.k}.txt"), "w",
         encoding="utf-8").write(txt)

    # ---- tables
    with open(os.path.join(C.TABLES, "cdpc_cluster_sizes.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "label", "n", "share", "top_form", "purity"])
        for c in range(a.k + 1):
            idx = [i for i in range(N) if lab[i] == c]
            if not idx:
                continue
            sub = collections.Counter(forms[i] for i in idx)
            w.writerow([c, "笑声（合并）" if c == a.k else f"C{c}", len(idx),
                        round(len(idx) / N, 4), sub.most_common(1)[0][0],
                        round(sub.most_common(1)[0][1] / len(idx), 3)])
    with open(os.path.join(C.TABLES, "cdpc_cluster_x_form.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster"] + [n for n, _ in FORMS] + ["其他"])
        for c in range(a.k + 1):
            idx = [i for i in range(N) if lab[i] == c]
            sub = collections.Counter(forms[i] for i in idx)
            w.writerow([c] + [sub.get(n, 0) for n, _ in FORMS] + [sub.get("其他", 0)])

    with open(os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_result2.jsonl"),
              "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            r = rows[cid]
            f.write(json.dumps({"clean_id": cid, "cluster": int(lab[i]),
                                "cluster_label": "笑声（合并）" if lab[i] == a.k
                                else f"C{lab[i]}",
                                "form": forms[i],
                                "text_norm": r.get("text_norm"),
                                "ctime_iso": r.get("ctime_iso")},
                               ensure_ascii=False) + "\n")

    draw(Z, Zi, ids, rows, lab, forms, a.k)
    print("->", C.TABLES)


def draw(Z, Zi, ids, rows, lab, forms, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    N = len(ids)
    # The projection axes are fitted on the CLUSTERED subset only (the laughter
    # layer is 57% of the points and would otherwise set the axes, squeezing
    # every real cluster into the middle). All points are projected with the
    # same basis, so the picture stays an honest 2-D view of one space.
    pca = PCA(n_components=2, random_state=C.RANDOM_SEED)
    pca.fit(Zi)
    coords = pca.transform(Z)
    var2 = pca.explained_variance_ratio_.sum() * 100

    # medoids computed on the clustered (non-laughter) subset only
    med = {}
    for c in range(K):
        idx = np.where(lab == c)[0]
        cent = Z[idx].mean(axis=0)
        cent /= (np.linalg.norm(cent) + 1e-12)
        med[c] = int(idx[np.argmax(Z[idx] @ cent)])

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    # 1) laughter FIRST, pale, small: it is the background layer
    m = lab == K
    ax.scatter(coords[m, 0], coords[m, 1], s=2.5, alpha=0.12, color=LAUGH_COLOR,
               edgecolors="none", zorder=1, rasterized=True)
    # 2) the actual clusters on top
    for c in range(K):
        mm = lab == c
        ax.scatter(coords[mm, 0], coords[mm, 1], s=9, alpha=0.70,
                   color=PALETTE[c % len(PALETTE)], edgecolors="none",
                   zorder=3, label=f"C{c}")
    for c, i in med.items():
        ax.scatter(coords[i, 0], coords[i, 1], s=150, marker="X", c="#222222",
                   edgecolors="white", linewidths=0.8, zorder=6)
        ax.annotate(f"C{c}", (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(9, 7), fontsize=9,
                    fontweight="bold", zorder=7)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      color=LAUGH_COLOR, alpha=0.6,
                      label=f"笑声（合并，n={int((lab == K).sum())}）")]
    for c in range(K):
        idx = [i for i in range(N) if lab[i] == c]
        sub = collections.Counter(forms[i] for i in idx)
        top, n = sub.most_common(1)[0]
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                              color=PALETTE[c % len(PALETTE)],
                              label=f"C{c} {top} {n/len(idx)*100:.0f}% (n={len(idx)})"))
    handles.append(Line2D([], [], marker="X", linestyle="", markersize=8,
                          color="#222222", label="密度峰中心（medoid）"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title(f"「{C.TARGET_WORD}」CDPC 聚类：笑声合并为背景层，其余 {K} 簇")
    ax.text(0.01, 0.01,
            f"注：聚类在 768 维余弦空间完成（笑声层不参与聚类）；PCA 主轴拟合于非笑声子集，"
            f"所有点用同一投影，二维图仅解释 {var2:.1f}% 方差，仅用于展示。",
            transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_ha_fig1_clusters.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    plt.close(fig)

    tab = collections.defaultdict(lambda: collections.Counter())
    for i, cid in enumerate(ids):
        y = (rows[cid].get("ctime_iso") or "")[:4]
        if y and y.isdigit():
            tab[int(y)][int(lab[i])] += 1
    years = sorted(tab)
    ns = [sum(tab[y].values()) for y in years]
    # stack order: laughter at the BOTTOM, then C0..C{K-1}
    order = [K] + list(range(K))
    shares = {c: [tab[y].get(c, 0) / max(1, sum(tab[y].values())) for y in years]
              for c in order}
    labels = ["笑声（合并）"] + [f"C{c}" for c in range(K)]
    colors = [LAUGH_COLOR] + [PALETTE[c % len(PALETTE)] for c in range(K)]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.stackplot(years, [shares[c] for c in order], labels=labels,
                 colors=colors, alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_xlabel("年份")
    ax.set_ylabel("占比")
    ax.set_title(f"「{C.TARGET_WORD}」历时构成：笑声（底）＋ {K} 簇（CDPC）")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_ha_fig4_stacked.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.4),
                                 height_ratios=[3, 1], sharex=True)
    for c, lab_txt, col in zip(order, labels, colors):
        a1.plot(years, shares[c], marker="o", ms=3.5, color=col, label=lab_txt)
    a1.set_ylabel("占比")
    a1.set_ylim(0, 1)
    a1.set_title(f"「{C.TARGET_WORD}」各簇占比的历时变化（笑声为背景层）")
    a1.legend(fontsize=8, frameon=False, ncol=4)
    a2.bar(years, ns, color="#999999")
    a2.set_ylabel("样本量 n")
    a2.set_xlabel("年份")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_ha_fig4b_line.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    plt.close(fig)


if __name__ == "__main__":
    main()
