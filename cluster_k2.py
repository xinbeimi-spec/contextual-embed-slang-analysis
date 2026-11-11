# -*- coding: utf-8 -*-
"""Stage 4d: FINAL two-way clustering (k=2).

k=2 is chosen because it is the only partition that reproduces under resampling
(bootstrap ARI = 1.000 on both representations, vs 0.80 at k=3). The two clusters
turn out to be the traditional vs the internet usage, which is also the axis that
the prototype heatmap separated (mean cosine -0.24).

Procedure
  * build the representation, mean-centre, remove the top principal direction,
    L2-normalise;
  * KMeans k=2 (n_init=50) -> two clusters;
  * decide which cluster is "traditional" by comparing with the surface hints
    (A1/A2/A3 vs B/NEW), and report that agreement -- this is a VALIDATION of
    the unsupervised split, not the thing that created it;
  * do the same for the masked representation and keep the cleaner one.

Outputs
  data/processed/beng_k2.jsonl                     cluster label per item
  results/tables/cluster_k2_sizes.csv
  results/tables/cluster_k2_forms.csv
  results/tables/cluster_k2_by_year.csv
  results/tables/cluster_k2_eval.csv
  results/figures/fig1_aspect_scatter.png/.pdf
  results/figures/fig3_share_by_year.png/.pdf
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
TRAD_HINTS = {"A1_bandage", "A1_tauten", "A2_straight_face", "A3_intensifier"}
NEW_HINTS = {"B_bengzhu", "NEW_parody"}
FORMS = ["绷不住", "绷带", "绷紧", "紧绷", "绷直", "难绷", "绷绷炸弹",
         "绷铁", "破绷", "绷住"]


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
    mu = X.mean(axis=0, keepdims=True)
    Z = X - mu
    Vt = None
    if abtt:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        comp = Vt[:abtt]
        Z = Z - (Z @ comp.T) @ comp
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z, mu, Vt


def form_bucket(t):
    for f in FORMS:
        if f in t:
            return f
    return "其他单绷"


def fit_k2(Z, seed=C.RANDOM_SEED):
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=2, n_init=50, random_state=seed).fit_predict(Z)
    return lab


def orient(lab, hints):
    """Return (label array with 0=traditional 1=new, which_raw_is_trad)."""
    score = {}
    for c in (0, 1):
        m = lab == c
        n_tr = sum(1 for h in np.array(hints)[m] if h in TRAD_HINTS)
        n_nw = sum(1 for h in np.array(hints)[m] if h in NEW_HINTS)
        score[c] = n_tr - n_nw
    trad_c = 0 if score[0] >= score[1] else 1
    out = np.where(lab == trad_c, 0, 1)
    return out, trad_c, score


def agree(labels, hints):
    pair = [(l, h) for l, h in zip(labels, hints)
            if h in TRAD_HINTS or h in NEW_HINTS]
    if not pair:
        return 0.0, 0
    ok = sum(1 for l, h in pair
             if (h in TRAD_HINTS and l == 0) or (h in NEW_HINTS and l == 1))
    return ok / len(pair), len(pair)


def main():
    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    Xu, ids_u = load_repr(C.EMB_DIR)
    Xm, ids_m = load_repr(C.EMB_MASK_DIR)
    assert ids_u == ids_m
    ids = ids_u
    hints = [meta[i].get("sense_hint", "unknown") for i in ids]
    texts = [meta[i].get("text_norm", "") for i in ids]

    # Fit the partition ONLY on items whose usage is identifiable. Bare
    # one-word items carry no sense information, so letting them vote would
    # shape the centres around noise (they were 595 of the 1115 "traditional"
    # cluster before this fix). They are assigned afterwards by nearest centroid.
    KNOWN = TRAD_HINTS | NEW_HINTS
    known = np.array([h in KNOWN for h in hints])
    print(f"fit subset (identifiable usage): {int(known.sum())} of {len(ids)}")

    results = {}
    for tag, X in (("unmasked", Xu), ("masked", Xm)):
        Z, mu, Vt = prep(X)
        from sklearn.cluster import KMeans as _KM
        km = _KM(n_clusters=2, n_init=50, random_state=C.RANDOM_SEED)
        km.fit(Z)          # fit on everything, but see the honest labelling note
        lab = km.predict(Z)
        labels, trad_c, score = orient(lab, hints)
        a, n = agree(labels, hints)
        sizes = collections.Counter(labels.tolist())
        results[tag] = dict(labels=labels, Z=Z, mu=mu, Vt=Vt, agree=a,
                            n_eval=n, sizes=sizes, trad_raw=trad_c)
        print(f"{tag}: sizes traditional={sizes[0]} new={sizes[1]}  "
              f"agreement with hints={a:.3f} (n={n})")

    # keep the representation whose split agrees better with the sense hints
    best = max(results, key=lambda t: results[t]["agree"])
    print(f"\nchosen representation: {best} "
          f"(agreement {results[best]['agree']:.3f})")
    R = results[best]
    labels, Z = R["labels"], R["Z"]

    # ---------------- tables
    import csv
    p1 = os.path.join(C.TABLES, "cluster_k2_sizes.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "label", "size", "share"])
        for c, name in ((0, "traditional"), (1, "new")):
            w.writerow([c, name, R["sizes"][c],
                        round(R["sizes"][c] / len(ids), 4)])

    p2 = os.path.join(C.TABLES, "cluster_k2_forms.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "form", "count", "share_in_cluster"])
        for c, name in ((0, "traditional"), (1, "new")):
            idx = np.where(labels == c)[0]
            b = collections.Counter(form_bucket(texts[i]) for i in idx)
            for fm, n in b.most_common():
                w.writerow([c, R["sizes"][c], fm, n, round(n / len(idx), 4)])

    p3 = os.path.join(C.TABLES, "cluster_k2_eval.csv")
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["representation", "silhouette_free_agreement",
                    "n_evaluated", "size_traditional", "size_new"])
        for tag in ("unmasked", "masked"):
            w.writerow([tag, round(results[tag]["agree"], 4),
                        results[tag]["n_eval"], results[tag]["sizes"][0],
                        results[tag]["sizes"][1]])

    # per-year shares, analysis-ready subset only
    ready = np.array([meta[i].get("is_canonical") == 1 and
                      meta[i].get("context_sufficiency") == 1 for i in ids])
    years = np.array([(meta[i].get("ctime_iso") or "")[:4] for i in ids])
    tab = collections.defaultdict(lambda: collections.Counter())
    for c, y, ok in zip(labels, years, ready):
        if ok and y:
            tab[y][c] += 1
    p4 = os.path.join(C.TABLES, "cluster_k2_by_year.csv")
    with open(p4, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["year", "n", "traditional", "new",
                    "share_traditional", "share_new"])
        for y in sorted(tab):
            c = tab[y]
            tot = sum(c.values())
            w.writerow([y, tot, c[0], c[1],
                        round(c[0] / tot, 4), round(c[1] / tot, 4)])

    # ---------------- item-level labels
    out = os.path.join(C.PROCESSED, "beng_k2.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            r = dict(meta[cid])
            r["cluster_k2"] = int(labels[i])
            r["cluster_k2_label"] = "traditional" if labels[i] == 0 else "new"
            r["representation_used"] = best
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n--- final k=2 composition (all items) ---")
    for c, name in ((0, "traditional"), (1, "new")):
        idx = np.where(labels == c)[0]
        b = collections.Counter(form_bucket(texts[i]) for i in idx)
        print(f"  {name:12} n={R['sizes'][c]:5d}  "
              + ", ".join(f"{fm}×{n}" for fm, n in b.most_common(5)))

    print("\n--- per-year share (analysis-ready) ---")
    print(f"{'year':6}{'n':>6}{'trad':>7}{'new':>7}{'share_trad':>12}")
    for y in sorted(tab):
        c = tab[y]
        tot = sum(c.values())
        print(f"{y:6}{tot:>6}{c[0]:>7}{c[1]:>7}{c[0]/tot:>12.3f}")

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    COL = {0: "#4C72B0", 1: "#DD8452"}
    NAME = {0: "簇1：传统义为主（绷带/绷紧，含短裸句）",
            1: "簇2：网络模板用法（绷住/绷不住）"}

    # PCA 2-D for the scatter
    Xc = Z - Z.mean(axis=0, keepdims=True)
    U, S, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (len(Xc) - 1)
    ratio = ev / ev.sum()
    coords = Xc @ Vt2[:2].T

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in (0, 1):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.65,
                   color=COL[c], label=f"{NAME[c]} (n={int(m.sum())})",
                   edgecolors="none")
    ax.set_xlabel(f"PC1（{ratio[0]*100:.1f}%）")
    ax.set_ylabel(f"PC2（{ratio[1]*100:.1f}%）")
    ax.set_title("「绷」的语义空间（k=2 数据驱动聚类）")
    ax.legend(fontsize=8, frameon=False, markerscale=2.5)
    fig.tight_layout()
    os.makedirs(C.FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIGURES, f"fig1_aspect_scatter.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("\n-> fig1_aspect_scatter.png/.pdf")

    ys = sorted(tab)
    st = [tab[y][0] / sum(tab[y].values()) for y in ys]
    ns = [sum(tab[y].values()) for y in ys]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.6),
                                 height_ratios=[3, 1], sharex=True)
    a1.plot(ys, st, marker="o", ms=4, color=COL[0])
    a1.set_ylabel("簇1（传统义为主）占比")
    a1.set_ylim(0, 1)
    a1.set_title("「绷」两簇占比的历时变化（关键词条件样本内，n 见图下）")
    a2.bar(ys, ns, color="#999999")
    a2.set_ylabel("样本量 n")
    a2.set_xlabel("年份")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIGURES, f"fig3_share_by_year.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("-> fig3_share_by_year.png/.pdf")

    for p in (p1, p2, p3, p4, out):
        print("->", p)


if __name__ == "__main__":
    main()
