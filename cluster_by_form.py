# -*- coding: utf-8 -*-
"""Stage 4e: cluster into the five surface-form groups and measure recovery.

Question being tested
  Instead of the binary traditional-vs-new split, treat the five frequent forms
    绷带 / 绷紧(含紧绷) / 难绷 / 绷住 / 绷不住
  as the category set. Does an unsupervised k=5 (or k=6) partition recover them,
  and with what purity? This tells us how much of the geometric structure is
  form-driven, which is the honest framing for a keyword-stratified corpus.

Metrics reported
  * purity / ARI / NMI between clusters and form groups  (recovery strength)
  * per-form recall (how often each form lands in its own dominant cluster)
  * silhouette + bootstrap stability for each k, so k is still not arbitrary

Outputs
  results/tables/form_cluster_recovery.csv
  results/tables/form_x_cluster_k{k}.csv
  results/figures/fig1b_form_clusters.png/.pdf
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
LAYERS = (9, 10, 11, 12)

# (label, patterns) -- checked in order, earliest occurrence wins,
# ties broken by longer pattern
FORMS = [
    ("绷不住", ["绷不住"]),
    ("绷带", ["绷带"]),
    ("绷紧", ["绷紧", "紧绷"]),
    ("难绷", ["难绷"]),
    ("绷住", ["绷住"]),
]


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_repr(root):
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
    return np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1), ids


def prep(X, abtt=1):
    Z = X - X.mean(axis=0, keepdims=True)
    if abtt:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        comp = Vt[:abtt]
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def form_label(text):
    best = None
    for lab, pats in FORMS:
        for p in pats:
            i = text.find(p)
            if i < 0:
                continue
            if best is None or i < best[0] or (i == best[0] and len(p) > best[2]):
                best = (i, lab, len(p))
    return best[1] if best else "其他"


def purity(y_true, y_pred):
    total = 0
    for c in set(y_pred):
        idx = [i for i, p in enumerate(y_pred) if p == c]
        cnt = collections.Counter(y_true[i] for i in idx)
        total += cnt.most_common(1)[0][1]
    return total / len(y_true)


def main():
    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    Xu, ids_u = load_repr(C.EMB_DIR)
    Xm, ids_m = load_repr(C.EMB_MASK_DIR)
    assert ids_u == ids_m
    ids = ids_u
    texts = [meta[i].get("text_norm", "") for i in ids]
    y = np.array([form_label(t) for t in texts])
    print("form distribution:", dict(collections.Counter(y.tolist())))

    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 silhouette_score)
    rng = np.random.default_rng(C.RANDOM_SEED)

    rows = []
    best = None
    for tag, X in (("unmasked", Xu), ("masked", Xm)):
        Z = prep(X)
        for k in (4, 5, 6, 7):
            km = KMeans(n_clusters=k, n_init=30,
                        random_state=C.RANDOM_SEED).fit(Z)
            lab = km.labels_
            pur = purity(y.tolist(), lab.tolist())
            ari = adjusted_rand_score(y, lab)
            nmi = normalized_mutual_info_score(y, lab)
            sil = silhouette_score(Z, lab, metric="cosine",
                                   sample_size=min(1500, len(Z)),
                                   random_state=C.RANDOM_SEED)
            aris = []
            n = len(Z)
            for _ in range(6):
                idx = rng.choice(n, size=int(0.8 * n), replace=False)
                l1 = KMeans(n_clusters=k, n_init=5,
                            random_state=C.RANDOM_SEED).fit_predict(Z[idx])
                l2 = KMeans(n_clusters=k, n_init=5,
                            random_state=C.RANDOM_SEED + 1).fit_predict(Z[idx])
                aris.append(adjusted_rand_score(l1, l2))
            row = {"representation": tag, "k": k,
                   "purity_vs_forms": round(pur, 4),
                   "ari_vs_forms": round(ari, 4),
                   "nmi_vs_forms": round(nmi, 4),
                   "silhouette": round(float(sil), 4),
                   "stability_ari": round(float(np.mean(aris)), 4)}
            rows.append(row)
            print(f"  {tag:9} k={k}  purity={pur:.4f}  ARI={ari:.4f}  "
                  f"NMI={nmi:.4f}  sil={sil:.4f}  stab={np.mean(aris):.4f}")
            if best is None or pur > best["purity_vs_forms"]:
                best = row
                best_lab = lab
                best_Z = Z
                best_rep = tag

    import csv
    p1 = os.path.join(C.TABLES, "form_cluster_recovery.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nbest recovery: {best_rep} k={best['k']} "
          f"purity={best['purity_vs_forms']}")

    # contingency for the best partition
    k = best["k"]
    lab = best_lab
    order = [f[0] for f in FORMS] + ["其他"]
    tab = collections.defaultdict(collections.Counter)
    for l_, f_ in zip(lab.tolist(), y.tolist()):
        tab[l_][f_] += 1
    p2 = os.path.join(C.TABLES, f"form_x_cluster_k{k}.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size"] + order)
        for c in sorted(tab):
            w.writerow([c, sum(tab[c].values())] + [tab[c][o] for o in order])

    print(f"\n--- contingency: cluster x form (k={k}, {best_rep}) ---")
    print(f"{'cl':>4}{'size':>7}" + "".join(f"{o:>9}" for o in order) + "   dominant")
    for c in sorted(tab):
        dom = tab[c].most_common(1)[0]
        print(f"{c:>4}{sum(tab[c].values()):>7}"
              + "".join(f"{tab[c][o]:>9}" for o in order)
              + f"   {dom[0]} ({dom[1]/sum(tab[c].values())*100:.0f}%)")

    # per-form recall (does each form have its own cluster?)
    print("\n--- per-form recall ---")
    for f_ in order:
        idx = [i for i, v in enumerate(y.tolist()) if v == f_]
        if not idx:
            continue
        cnt = collections.Counter(lab[i] for i in idx)
        top, n = cnt.most_common(1)[0]
        print(f"  {f_:8} n={len(idx):5d}  dominant cluster={top} "
              f"({n}/{len(idx)} = {100*n/len(idx):.1f}%)")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    Xc = best_Z - best_Z.mean(axis=0, keepdims=True)
    _, S, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (len(Xc) - 1)
    ratio = ev / ev.sum()
    coords = Xc @ Vt2[:2].T

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, c in enumerate(sorted(tab)):
        m = lab == c
        ax.scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.65,
                   color=cmap(i % 10), label=f"cluster {c} (n={int(m.sum())})",
                   edgecolors="none")
    ax.set_xlabel(f"PC1（{ratio[0]*100:.1f}%）")
    ax.set_ylabel(f"PC2（{ratio[1]*100:.1f}%）")
    ax.set_title(f"「绷」按词形聚成 {k} 类（{best_rep} 表示，"
                 f"纯度 {best['purity_vs_forms']:.3f}）")
    ax.legend(fontsize=8, frameon=False, markerscale=2.5, ncol=2)
    fig.tight_layout()
    os.makedirs(C.FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIGURES, f"fig1b_form_clusters.{ext}"),
                    dpi=300, bbox_inches="tight")
    for p in (p1, p2):
        print("->", p)
    print("-> fig1b_form_clusters.png/.pdf")


if __name__ == "__main__":
    main()
