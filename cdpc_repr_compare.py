# -*- coding: utf-8 -*-
"""Cluster 塌 with the TARGET representation and compare it with MASKED.

Context (2026-11-01): the project has moved on to a third word (拉), and
cdpc_pipeline.py now carries 拉's form/category tables. To compare the two
representations for 塌 without touching that code, this script is SELF-
CONTAINED: it re-declares 塌's form inventory / category rules locally, and
reuses only the stable maths in cluster_cdpc.py.

Protocol (identical for both representations, so the comparison is fair):
  shards -> mean of layers 9-12 -> mean-centre -> remove 1st PC (ABTT k=1)
        -> L2-normalise -> cosine -> CDPC (dc = 2nd pct) -> k-medoids refine
Only the vector SOURCE differs: emb2/masked (塌 replaced by [MASK]) vs
emb2/target (mean over the 塌 span).

Outputs (results/<slug>/result6/)
  cdpc_fig1_clusters_masked.png/.pdf, cdpc_fig1_clusters_target.png/.pdf
  cdpc_sizes_{masked,target}.csv
  cdpc_cluster_x_form_{masked,target}.csv
  repr_compare.csv
  logs/<slug>/repr_compare.txt
Run:
  python src/analyze/cdpc_repr_compare.py --k 8
"""
import os
import re
import sys
import json
import csv
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, assign, kmedoids_refine

sys.stdout.reconfigure(encoding="utf-8")

LAYERS = (9, 10, 11, 12)

# ---------------------------------------------------------------- 塌 as a word
# Kept LOCAL on purpose: the shared pipeline now belongs to 拉.
FORMS_TA = [("塌房", ["塌房", "房子塌了", "人设塌"]),
            ("塌方", ["塌方"]),
            ("塌陷", ["塌陷"]),
            ("坍塌", ["坍塌", "崩塌"]),
            ("倒塌", ["倒塌"]),
            ("塌鼻梁", ["塌鼻梁"]),
            ("一塌糊涂", ["一塌糊涂"]),
            ("塌实", ["塌实", "塌心"]),
            ("塌了", ["塌了", "塌下来", "塌掉", "塌垮", "塌楼", "塌落"])]
PHYSICAL_FRAME = (r"山|山坡|山体|矿|矿洞|煤矿|隧道|桥|桥梁|楼|楼房|房子|房屋|屋|"
                  r"屋顶|墙|墙体|围墙|地|地面|地基|路面|道路|马路|工地|支架|"
                  r"顶棚|天花板|棚|坝|堤|坑|洞|岩石|土|泥|废墟|建筑|楼板|"
                  r"楼梯|电梯|管道|水管|操场|看台|舞台|架子|货架")
FANDOM_CUE = re.compile(
    r"塌不塌|塌没塌|塌啥房|内娱|爱豆|偶像|饭圈|人设|吃瓜|塌一个|一塌一个|"
    r"脱粉|翻车|纸片人|还能塌|塌过|塌得了|塌不塌房|"
    r"明星|艺人|网红|追星|打榜|应援|官宣|绯闻|恋爱塌")

CAT_ORDER = ["物理·倒塌", "物理·坍塌", "物理·塌方", "物理·塌陷", "物理·塌了",
             "物理·其他", "塌房义项", "成语义项", "下垂凹陷义项", "非义项用法"]
CAT_COLOR = {"物理·倒塌": "#1F77B4", "物理·坍塌": "#2CA02C",
             "物理·塌方": "#FF7F0E", "物理·塌陷": "#D62728",
             "物理·塌了": "#9467BD", "物理·其他": "#8C564B",
             "塌房义项": "#E377C2", "成语义项": "#BCBD22",
             "下垂凹陷义项": "#17BECF", "非义项用法": "#7F7F7F"}


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def form_label(t):
    for f, pats in FORMS_TA:
        for p in pats:
            if p in t:
                return f
    return "其他"


def item_category(t):
    t = t or ""
    if FANDOM_CUE.search(t) and not re.search(r"倒塌|坍塌|塌方|塌陷|崩塌", t):
        return "塌房义项"
    f = form_label(t)
    if f == "塌房":
        return "塌房义项"
    if f in ("塌方", "塌陷", "坍塌", "倒塌", "塌了"):
        return f"物理·{f}"
    if f == "塌鼻梁":
        return "下垂凹陷义项"
    if f == "一塌糊涂":
        return "成语义项"
    if f == "塌实":
        return "下垂凹陷义项"          # 3 items, excluded from typology anyway
    if re.search(PHYSICAL_FRAME, t):
        return "物理·其他"
    return "非义项用法"


# ------------------------------------------------------------------ vectors
def load_repr(root):
    """shards -> layer mean -> centre -> ABTT(1) -> L2. Same for both sources."""
    index = rj(os.path.join(root, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(root, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    mu = X.mean(axis=0, keepdims=True)
    Z = X - mu
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    comp = Vt[:1]
    Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12), ids


def cluster(Z, k, dc_pct=2.0):
    S = Z @ Z.T
    rho, delta, gamma, parent, dc = density_peaks(S, dc_pct)
    labels = assign(parent, np.argsort(-gamma)[:k], rho)
    labels, medoids = kmedoids_refine(Z, labels)
    return labels, medoids, S, rho, gamma, dc


def silhouette(Z, labels):
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(Z, labels, metric="cosine",
                                 sample_size=min(1500, len(labels)),
                                 random_state=C.RANDOM_SEED))


def pca2(Z):
    Zm = Z - Z.mean(axis=0, keepdims=True)
    _, _, V = np.linalg.svd(Zm, full_matrices=False)
    coords = Zm @ V[:2].T
    var2 = float(np.var(coords, axis=0).sum() / np.var(Z, axis=0).sum())
    return coords, var2


def plot(Z, labels, medoids, texts, tag, k, sil, var2, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    coords, _v = pca2(Z)
    ycat = np.array([item_category(t) for t in texts])
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for c in CAT_ORDER:
        m = ycat == c
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.55,
                   color=CAT_COLOR[c], edgecolors="none")
    mids = [medoids[c] for c in sorted(medoids)]
    ax.scatter(coords[mids, 0], coords[mids, 1], s=130, marker="X",
               c="#222222", edgecolors="white", linewidths=0.8, zorder=5)
    for c in sorted(medoids):
        i = medoids[c]
        ax.annotate(f"C{c}", (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(7, 6), fontsize=8)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      color=CAT_COLOR[c],
                      label=f"{c} (n={int((ycat == c).sum())})")
               for c in CAT_ORDER if (ycat == c).any()]
    handles.append(Line2D([], [], marker="X", linestyle="", markersize=8,
                          color="#222222", label="密度峰中心（medoid）"))
    # legend OUTSIDE the axes: 10 classes + medoid would cover the C-labels
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title(f"「{C.TARGET_WORD}」{tag} 表示 · CDPC 聚类"
                 f"（k={k}，cosine，silhouette={sil:.3f}）", fontsize=10.5)
    ax.text(0.01, 0.01,
            f"注：聚类在 768 维余弦空间完成，此 PCA 二维投影仅解释方差的 "
            f"{100*var2:.1f}%，仅用于展示。",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = f"{out_base}.{ext}"
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--round", default="result6")
    a = ap.parse_args()

    clean = {r["clean_id"]: r for r in rj(C.PROCESSED and
              os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl"))}
    outdir = os.path.join(C.RESULT_ROOT, a.round)
    os.makedirs(outdir, exist_ok=True)

    res = {}
    for tag, root in (("masked", C.EMB_MASK_DIR), ("target", C.EMB_DIR)):
        Z, ids = load_repr(root)
        labels, medoids, S, rho, gamma, dc = cluster(Z, a.k)
        sizes = collections.Counter(labels.tolist())
        texts = [clean[c]["text_norm"] for c in ids]
        sil = silhouette(Z, labels)
        coords, var2 = pca2(Z)
        print(f"[{tag}] N={len(ids)} k={len(sizes)} silhouette={sil:.4f} "
              f"dc={dc:.4f} PCA2={100*var2:.1f}% "
              f"min_basin={min(sizes.values())}")

        # tables
        with open(os.path.join(outdir, f"cdpc_sizes_{tag}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "size", "share", "silhouette_global",
                        "medoid_clean_id", "medoid_text"])
            for c in sorted(sizes):
                m = medoids.get(c)
                w.writerow([c, sizes[c], round(sizes[c] / len(ids), 4),
                            round(sil, 4), ids[m] if m is not None else "",
                            clean[ids[m]]["text_norm"][:150] if m is not None else ""])
        with open(os.path.join(outdir, f"cdpc_cluster_x_form_{tag}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "size"] + [f for f, _ in FORMS_TA] + ["其他"])
            for c in sorted(sizes):
                idx = np.where(labels == c)[0]
                cnt = collections.Counter(form_label(texts[i]) for i in idx)
                w.writerow([c, sizes[c]] + [cnt[f] for f, _ in FORMS_TA]
                           + [cnt["其他"]])

        purity = 0
        for c in sizes:
            idx = np.where(labels == c)[0]
            purity += collections.Counter(form_label(texts[i]) for i in idx).most_common(1)[0][1]
        res[tag] = dict(Z=Z, ids=ids, labels=labels, medoids=medoids, sizes=sizes,
                        texts=texts, sil=sil, var2=var2, purity=purity / len(ids))
        plot(Z, labels, medoids, texts, "masked" if tag == "masked" else "target",
             len(sizes), sil, var2,
             os.path.join(outdir, f"cdpc_fig1_clusters_{tag}"))

    # ---- agreement between the two solutions
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    assert res["masked"]["ids"] == res["target"]["ids"], "id order mismatch"
    ari = adjusted_rand_score(res["masked"]["labels"], res["target"]["labels"])
    nmi = normalized_mutual_info_score(res["masked"]["labels"],
                                       res["target"]["labels"])
    ct = collections.Counter(zip(res["masked"]["labels"].tolist(),
                                 res["target"]["labels"].tolist()))
    best = {}
    for (cm, ctn), n in ct.items():
        if n > best.get(cm, (None, 0))[1]:
            best[cm] = (ctn, n)
    match_rate = sum(v[1] for v in best.values()) / len(res["masked"]["ids"])

    rows = []
    for tag in ("masked", "target"):
        r = res[tag]
        rows.append({"representation": tag, "k": a.k, "n": len(r["ids"]),
                     "silhouette": round(r["sil"], 4),
                     "form_purity": round(r["purity"], 4),
                     "pca2_var": round(r["var2"], 4),
                     "min_basin": int(min(r["sizes"].values())),
                     "max_basin": int(max(r["sizes"].values())),
                     "n_clusters": len(r["sizes"])})
    rows.append({"representation": "ARI", "k": a.k, "n": len(res["masked"]["ids"]),
                 "silhouette": round(ari, 4), "form_purity": round(nmi, 4),
                 "pca2_var": "", "min_basin": "", "max_basin": "",
                 "n_clusters": round(match_rate, 4)})
    with open(os.path.join(outdir, "repr_compare.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [f"「{C.TARGET_WORD}」masked vs target 聚类对比（k={a.k}）", "",
             f"{'repr':<8}{'silhouette':>12}{'form_purity':>13}{'PCA2':>8}"
             f"{'min_basin':>11}{'max_basin':>11}"]
    for r in rows[:2]:
        lines.append(f"{r['representation']:<8}{r['silhouette']:>12.4f}"
                     f"{r['form_purity']:>13.4f}{r['pca2_var']*100:>7.1f}%"
                     f"{r['min_basin']:>11}{r['max_basin']:>11}")
    lines += ["", f"ARI={ari:.4f}  NMI={nmi:.4f}  同簇最佳匹配率={match_rate:.3f}",
              "", "簇级对应（masked -> target, n）:"]
    for cm in sorted(best):
        lines.append(f"  C{cm} -> C{best[cm][0]}  ({best[cm][1]})")
    txt = "\n".join(lines)
    open(os.path.join(C.LOGS, "repr_compare.txt"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
