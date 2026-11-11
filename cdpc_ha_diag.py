# -*- coding: utf-8 -*-
"""First look at CDPC on 哈 (2026-11-09): k sweep + cluster structure.

The full cdpc_pipeline.py is still 拉-specific in its category tables, so this
script only answers the question "what does the automatic clustering look like
on 哈?" without committing to a sense inventory:

  * the automatic gamma rule (choose_k) and a k = 3..12 sweep on
    form purity / silhouette / min basin / categories present;
  * for the automatic k: cluster sizes, cluster x form cross-tab, per-cluster
    examples (so a human can see what each basin actually contains).

Form labels here are PROVISIONAL reading aids, not the final sense table.

Run: python src/analyze/cdpc_ha_diag.py [--k 0]
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
from cluster_cdpc import density_peaks, choose_k, assign, kmedoids_refine
from cdpc_pipeline import load_masked

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
TABLE = C.TABLES

FORMS = [
    ("哈基米", ["哈基米", "hjm"]),
    ("哈皮", ["哈皮", "哈批"]),
    ("哈人", ["哈人"]),
    ("哈喽", ["哈喽", "哈啰", "哈罗"]),
    ("打哈哈", ["打哈哈"]),
    ("哈哈镜", ["哈哈镜"]),
    ("哈欠", ["哈欠"]),
    ("哈气", ["哈气", "哈手"]),
    ("哈腰", ["哈腰"]),
    ("马大哈", ["马大哈"]),
    ("二哈", ["二哈", "哈士奇"]),
    ("疑问", ["哈？", "哈?"]),
    ("笑声", ["哈哈哈", "哈哈"]),
]
CAT_OF = {"哈基米": "网络新义", "哈人": "网络新义", "哈皮": "方言义",
          "哈喽": "问候义", "打哈哈": "词典义", "哈哈镜": "词典义",
          "哈欠": "词典义", "哈气": "词典义", "哈腰": "词典义",
          "马大哈": "词典义", "二哈": "非义项/其他", "疑问": "疑问义",
          "笑声": "笑声义", "其他": "非义项/其他"}
CAT_ORDER = ["笑声义", "疑问义", "网络新义", "方言义", "词典义", "问候义",
             "非义项/其他"]


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def form_label(t):
    t = t or ""
    for name, pats in FORMS:
        for p in pats:
            if p in t:
                return name
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--dc-pct", type=float, default=2.0)
    ap.add_argument("--no-sweep", action="store_true")
    # 2026-11-09 (route C->A): the laughter layer is 55.7% of the corpus and
    # owns 4 of the 7 density basins, so the other senses never get a peak.
    # --exclude-laugh runs the SAME clustering on the non-laughter subset,
    # where 哈基米 / 哈欠 / 哈气 / 哈皮 / 疑问 / 哈喽 have room to separate.
    ap.add_argument("--exclude-laugh", action="store_true")
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args()

    Z, ids = load_masked()
    rows = {r["clean_id"]: r for r in rj(CLEAN)}
    assert all(c in rows for c in ids), "clean_id mismatch between vectors and clean table"
    N = len(ids)
    forms = [form_label(rows[c].get("text_norm")) for c in ids]
    cats = [CAT_OF.get(f, "非义项/其他") for f in forms]
    if a.exclude_laugh:
        keep = [f != "笑声" for f in forms]
        Z = Z[np.array(keep)]
        ids = [i for i, k in zip(ids, keep) if k]
        forms = [f for f, k in zip(forms, keep) if k]
        cats = [c for c, k in zip(cats, keep) if k]
        N = len(ids)
        print(f"[two-stage] laughter excluded -> items={N}")
    print(f"items={N} dim={Z.shape[1]}")

    S = Z @ Z.T
    off = S[np.triu_indices_from(S, 1)]
    print(f"cosine: mean={off.mean():.4f} p1={np.percentile(off,1):.4f} "
          f"p99={np.percentile(off,99):.4f}")

    rho, delta, gamma, parent, dc = density_peaks(S, a.dc_pct)
    print(f"dc={dc:.4f}")
    k_auto, _r, _g = choose_k(gamma, 0)
    print(f"automatic gamma rule -> k={k_auto}")

    from sklearn.metrics import silhouette_score
    sweep = []
    if a.no_sweep:
        print("(sweep skipped)")
    print(f"\n{'k':>3}{'form_pur':>10}{'sil':>9}{'n_cat':>7}{'min_basin':>11}  forms")
    for k in ([] if a.no_sweep else range(3, 13)):
        centers = np.argsort(-gamma)[:k]
        lab = assign(parent, centers, rho)
        lab, _med = kmedoids_refine(Z, lab, iters=4)
        sizes = collections.Counter(lab.tolist())
        tot = 0
        for c in sizes:
            sub = collections.Counter(f for f, l in zip(forms, lab.tolist()) if l == c)
            tot += sub.most_common(1)[0][1]
        pur = tot / N
        sil = silhouette_score(Z, lab, metric="cosine",
                               sample_size=min(1500, N), random_state=C.RANDOM_SEED)
        cl_cat = {}
        for c in sizes:
            sub = collections.Counter(cc for cc, l in zip(cats, lab.tolist()) if l == c)
            cl_cat[c] = sub.most_common(1)[0][0]
        present = sorted(set(cl_cat.values()))
        sweep.append({"k": k, "form_purity": round(float(pur), 4),
                      "silhouette": round(float(sil), 4),
                      "n_categories": len(present), "min_basin": int(min(sizes.values())),
                      "categories": "/".join(present),
                      "chosen": 1 if k == k_auto else 0})
        print(f"{k:>3}{pur:>10.4f}{sil:>9.4f}{len(present):>7}"
              f"{min(sizes.values()):>11}  {'/'.join(present)}")

    os.makedirs(TABLE, exist_ok=True)
    if sweep:
        p = os.path.join(TABLE, "cdpc_k_choice.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)
        print(f"\n-> {p}")

    K = a.k or k_auto
    centers = np.argsort(-gamma)[:K]
    lab = assign(parent, centers, rho)
    lab, med = kmedoids_refine(Z, lab, iters=8)
    lab = np.asarray(lab)

    # order clusters by size
    sizes = collections.Counter(lab.tolist())
    order = sorted(sizes, key=lambda c: -sizes[c])
    remap = {c: i for i, c in enumerate(order)}
    lab = np.array([remap[x] for x in lab])

    print(f"\n=== k={K}: cluster structure ===")
    lines = [f"k={K}  n={N}"]
    for c in range(len(order)):
        idx = [i for i in range(N) if lab[i] == c]
        sub = collections.Counter(forms[i] for i in idx)
        cat = collections.Counter(cats[i] for i in idx)
        purity = sub.most_common(1)[0][1] / len(idx)
        lines.append(f"\nC{c}  n={len(idx)} ({len(idx)/N*100:.1f}%)  "
                     f"purity={purity:.2f}  form={sub.most_common(1)[0][0]}  "
                     f"cat={cat.most_common(1)[0][0]}")
        lines.append("   forms: " + " ".join(f"{k}:{v}" for k, v in sub.most_common(6)))
        for i in idx[:5]:
            lines.append(f"   - [{rows[ids[i]]['ctime_iso'][:10]}] "
                         f"{rows[ids[i]]['text_norm'][:60]}")
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(C.LOGS, f"cdpc_diag_k{K}.txt")
    open(out, "w", encoding="utf-8").write(txt)
    print(f"\n-> {out}")

    # tables
    with open(os.path.join(TABLE, "cdpc_cluster_sizes.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n", "share", "top_form", "purity", "top_category"])
        for c in range(len(order)):
            idx = [i for i in range(N) if lab[i] == c]
            sub = collections.Counter(forms[i] for i in idx)
            cat = collections.Counter(cats[i] for i in idx)
            w.writerow([c, len(idx), round(len(idx) / N, 4), sub.most_common(1)[0][0],
                        round(sub.most_common(1)[0][1] / len(idx), 3),
                        cat.most_common(1)[0][0]])
    with open(os.path.join(TABLE, "cdpc_cluster_x_form.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster"] + [n for n, _ in FORMS] + ["其他"])
        for c in range(len(order)):
            idx = [i for i in range(N) if lab[i] == c]
            sub = collections.Counter(forms[i] for i in idx)
            w.writerow([c] + [sub.get(n, 0) for n, _ in FORMS] + [sub.get("其他", 0)])

    # per-item labels for inspection
    pj = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_diag.jsonl")
    with open(pj, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            r = rows[cid]
            f.write(json.dumps({"clean_id": cid, "cluster": int(lab[i]),
                                "form": forms[i], "category": cats[i],
                                "text_norm": r.get("text_norm"),
                                "ctime_iso": r.get("ctime_iso")},
                               ensure_ascii=False) + "\n")
    print(f"-> {pj}")

    if a.fig:
        draw(Z, ids, rows, lab, forms, len(order))


def draw(Z, ids, rows, lab, forms, K):
    """PCA scatter + yearly stacked + yearly line, same style as cdpc_pipeline."""
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
    pca = PCA(n_components=2, random_state=C.RANDOM_SEED)
    coords = pca.fit_transform(Z)
    var2 = pca.explained_variance_ratio_.sum() * 100

    # medoid of each (remapped) cluster: member closest to the cluster centroid
    med = {}
    for c in range(K):
        idx = np.where(lab == c)[0]
        cent = Z[idx].mean(axis=0)
        cent /= (np.linalg.norm(cent) + 1e-12)
        med[c] = int(idx[np.argmax(Z[idx] @ cent)])

    colors = list(plt.get_cmap("tab10").colors)
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for c in range(K):
        m = lab == c
        ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.55,
                   color=colors[c % 10], edgecolors="none", label=f"C{c}")
    for c, i in med.items():
        ax.scatter(coords[i, 0], coords[i, 1], s=140, marker="X",
                   c="#222222", edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(f"C{c}", (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    # legend with cluster size + dominant form
    handles = []
    for c in range(K):
        idx = [i for i in range(N) if lab[i] == c]
        sub = collections.Counter(forms[i] for i in idx)
        top, n = sub.most_common(1)[0]
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                              color=colors[c % 10],
                              label=f"C{c} {top} {n/len(idx)*100:.0f}% (n={len(idx)})"))
    handles.append(Line2D([], [], marker="X", linestyle="", markersize=8,
                          color="#222222", label="密度峰中心（medoid）"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title(f"「{C.TARGET_WORD}」CDPC 密度峰聚类（k={K}，余弦相似度）")
    ax.text(0.01, 0.01,
            f"注：聚类在 768 维余弦空间完成，此 PCA 二维投影仅解释该空间方差的 "
            f"{var2:.1f}%，仅用于展示。",
            transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_ha_fig1_clusters_k{K}.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    plt.close(fig)

    # ---- yearly composition
    tab = collections.defaultdict(lambda: collections.Counter())
    for i, cid in enumerate(ids):
        y = (rows[cid].get("ctime_iso") or "")[:4]
        if y and y.isdigit():
            tab[int(y)][int(lab[i])] += 1
    years = sorted(tab)
    ns = [sum(tab[y].values()) for y in years]
    shares = {c: [tab[y].get(c, 0) / max(1, sum(tab[y].values())) for y in years]
              for c in range(K)}
    labels = []
    for c in range(K):
        idx = [i for i in range(N) if lab[i] == c]
        sub = collections.Counter(forms[i] for i in idx)
        labels.append(f"C{c} {sub.most_common(1)[0][0]}")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.stackplot(years, [shares[c] for c in range(K)], labels=labels,
                 colors=[colors[c % 10] for c in range(K)], alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_xlabel("年份")
    ax.set_ylabel("占比")
    ax.set_title(f"「{C.TARGET_WORD}」{K} 簇的历时构成（CDPC，百分比堆叠）")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_ha_fig4_stacked_k{K}.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.4),
                                 height_ratios=[3, 1], sharex=True)
    for c in range(K):
        a1.plot(years, shares[c], marker="o", ms=3.5, color=colors[c % 10],
                label=labels[c])
    a1.set_ylabel("占比")
    a1.set_ylim(0, 1)
    a1.set_title(f"「{C.TARGET_WORD}」各簇占比的历时变化（CDPC）")
    a1.legend(fontsize=8, frameon=False, ncol=3)
    a2.bar(years, ns, color="#999999")
    a2.set_ylabel("样本量 n")
    a2.set_xlabel("年份")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_ha_fig4b_line_k{K}.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    plt.close(fig)


if __name__ == "__main__":
    main()
