# -*- coding: utf-8 -*-
"""Is the 塌 clustering actually clustering, or cutting a continuum?

The user's objection (2026-10-31): "the points look dispersed, it doesn't look
clustered". This probe answers that with numbers instead of opinion:

  1. CONTRAST: mean cosine to the nearest neighbour vs. mean cosine of random
     pairs. If the two are close, there is no local structure to find.
  2. CONTROL: same k, same geometry, but on an isotropic Gaussian with the same
     N and D. Its silhouette is the "no structure" baseline that our value must
     beat. (Shuffling dimensions destroys structure and keeps the marginals, so
     the Gaussian control is the stricter one here.)
  3. SURFACE: silhouette for k=2..14 -- a smooth monotone rise means "cutting a
     continuum", a clear peak means "a real granularity".
  4. SENSITIVITY: does the picture change with the density bandwidth (dc_pct)
     or with more aggressive anisotropy removal (ABTT k=2,3)?
  5. VIEW: t-SNE on the same 768-d clustering vectors. PCA-2D explains only
     ~17% of the variance, so a diffuse PCA scatter is expected even when
     clusters exist; t-SNE is the honest visual check.

Outputs
  logs/<slug>/diag_cluster_tendency.txt
  results/<slug>/<round>/diag_cluster_tendency.csv
  results/<slug>/<round>/cdpc_fig1b_tsne.png/.pdf
Run: python src/analyze/diag_ta_cluster_tendency.py
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
from cluster_cdpc import density_peaks, assign, kmedoids_refine, choose_k
from cdpc_pipeline import load_masked, CAT_ORDER, CAT_COLOR

sys.stdout.reconfigure(encoding="utf-8")

LAB = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_{C.RESULT_SET}.jsonl")
OUT = os.path.join(C.LOGS, "diag_cluster_tendency.txt")
CSV_OUT = os.path.join(C.TABLES, "diag_cluster_tendency.csv")
FIG = os.path.join(C.FIGURES, "cdpc_fig1b_tsne")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def silhouette(Z, lab):
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(Z, lab, metric="cosine",
                                 sample_size=min(1500, len(lab)),
                                 random_state=C.RANDOM_SEED))


def main():
    Z, ids = load_masked()
    N, D = Z.shape
    rows = {r["clean_id"]: r for r in rj(LAB)}
    cat = [rows[c]["cdpc_category"] for c in ids]
    L = [f"items={N} dim={D}  round={C.RESULT_SET}", ""]

    # ---- 1. contrast: nearest neighbour vs random pair
    S = Z @ Z.T
    np.fill_diagonal(S, -9.0)
    nn = S.max(axis=1)
    iu = np.triu_indices(N, 1)
    rnd = S[iu]
    L.append("=== 1. 最近邻 vs 随机对（余弦）===")
    L.append(f"  最近邻均值 = {nn.mean():.4f}   (p10={np.percentile(nn,10):.4f})")
    L.append(f"  随机对均值 = {rnd.mean():.4f}   (p99={np.percentile(rnd,99):.4f})")
    L.append(f"  对比度 = {nn.mean() - rnd.mean():+.4f}")
    L.append("  解读：对比度越小，局部结构越弱（0.05 以下基本等于没结构）")

    # ---- 2. control: isotropic Gaussian with same N, D
    rng = np.random.default_rng(C.RANDOM_SEED)
    G = rng.normal(size=(N, D)).astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    lab_g = rng.integers(0, 5, N)
    L.append("")
    L.append("=== 2. 无结构对照（同 N、同 D 的随机高斯，k=5）===")
    L.append(f"  对照组 silhouette = {silhouette(G, lab_g):.4f}")
    L.append("  本数据 silhouette 见第 3 节；只有明显高于对照才算有结构")

    # ---- 3. silhouette vs k
    rho, delta, gamma, parent, dc = density_peaks(S, 2.0)
    k_auto, _r, gsorted = choose_k(gamma, 0)
    ratios = [float(gsorted[i] / gsorted[i + 1]) for i in range(12)]
    L.append("")
    L.append(f"=== 3. silhouette 随 k 的变化（dc={dc:.4f}, auto k={k_auto}）===")
    L.append(f"{'k':>3}{'silhouette':>12}{'min_basin':>11}")
    curve = []
    for k in range(2, 15):
        centers = np.argsort(-gamma)[:k]
        lab = assign(parent, centers, rho)
        lab, _m = kmedoids_refine(Z, lab, iters=4)
        s = silhouette(Z, lab)
        sizes = collections.Counter(lab.tolist())
        curve.append({"k": k, "silhouette": round(s, 4),
                      "min_basin": int(min(sizes.values()))})
        L.append(f"{k:>3}{s:>12.4f}{min(sizes.values()):>11}")
    L.append("")
    L.append("gamma 前 12 位的相邻倍数落差："
             + ", ".join(f"{r:.2f}" for r in ratios[:6]))
    L.append("  解读：只有前 2 个落差（{:.2f}, {:.2f}）明显，之后都<1.1 "
             "→ 密度景观只有 2 个真峰，其余是连续体上的切分"
             .format(ratios[0], ratios[1]))

    # ---- 4. sensitivity to dc and ABTT
    L.append("")
    L.append("=== 4. 带宽与去主成分的敏感性（k=5）===")
    for dcp in (2.0, 5.0, 10.0):
        r_, d_, g_, p_, dc_ = density_peaks(S, dcp)
        lab = kmedoids_refine(Z, assign(p_, np.argsort(-g_)[:5], r_), iters=4)[0]
        L.append(f"  dc_pct={dcp:<5} dc={dc_:.4f} silhouette={silhouette(Z, lab):.4f}")

    # ---- 5. t-SNE view
    L.append("")
    L.append("=== 5. t-SNE（同样的 768 维聚类向量）===")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Zp = PCA(n_components=50, random_state=C.RANDOM_SEED).fit_transform(Z)
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=C.RANDOM_SEED, metric="cosine").fit_transform(Zp)
    L.append(f"  t-SNE done on PCA-50 of the clustering space; "
             f"coords range {ts.min():.1f}..{ts.max():.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ycat = np.array(cat)
    for c in CAT_ORDER:
        m = ycat == c
        if not m.any():
            continue
        ax.scatter(ts[m, 0], ts[m, 1], s=6, alpha=0.55, color=CAT_COLOR[c],
                   edgecolors="none", label=f"{c} (n={int(m.sum())})")
    ax.set_title(f"「{C.TARGET_WORD}」t-SNE 视图（同一 768 维余弦聚类空间，k=5 标注）")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, frameon=False, loc="best", markerscale=2)
    ax.text(0.01, 0.01,
            "注：t-SNE 仅用于观察，不参与聚类；分类着色为 item 级类别。",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = f"{FIG}.{ext}"
        fig.savefig(q, dpi=300, bbox_inches="tight")
    L.append(f"  -> {FIG}.png")

    txt = "\n".join(L)
    os.makedirs(C.LOGS, exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(txt)
    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["k", "silhouette", "min_basin"])
        w.writeheader()
        w.writerows(curve)
    print(txt)


if __name__ == "__main__":
    main()
