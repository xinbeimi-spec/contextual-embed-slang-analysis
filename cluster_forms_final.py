# -*- coding: utf-8 -*-
"""Stage 4f: the FINAL form-based clusterings, two granularities.

  k=5  : what was asked for -- 绷带 / 绷紧 / 难绷 / 绷住 / 绷不住
         Reality: four forms come out clean, but 绷住 + other items collapse
         into one 1323-item blob, because 绷住 alone needs three clusters.
  k=7  : what the data prefers (purity 0.913, stability 0.949)
         five form clusters + three 绷住 sub-clusters + one residual cluster.

Both are written out so the two granularities can be compared item-by-item.

Outputs
  data/processed/beng_forms.jsonl          cluster_k5 + cluster_k7 per item
  results/tables/forms_k5_contingency.csv
  results/tables/forms_k7_contingency.csv
  results/tables/forms_k5_topforms.csv
  results/tables/forms_k7_topforms.csv
  results/tables/forms_k5_by_year.csv
  results/figures/fig1b_k5_clusters.png/.pdf
  results/figures/fig1c_k7_clusters.png/.pdf
"""
import os
import sys
import json
import collections

import numpy as np
import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")
jieba.setLogLevel(60)

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
ROOT = C.EMB_MASK_DIR     # best recovery was the masked space
LAYERS = (9, 10, 11, 12)
FORMS = [("绷不住", ["绷不住"]), ("绷带", ["绷带"]),
         ("绷紧", ["绷紧", "紧绷"]), ("难绷", ["难绷"]), ("绷住", ["绷住"])]
ORDER = [f[0] for f in FORMS] + ["其他"]


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load():
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
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12), ids, Vt


def form_label(t):
    best = None
    for lab, pats in FORMS:
        for p in pats:
            i = t.find(p)
            if i < 0:
                continue
            if best is None or i < best[0] or (i == best[0] and len(p) > best[2]):
                best = (i, lab, len(p))
    return best[1] if best else "其他"


def purity(y, lab):
    tot = 0
    for c in set(lab):
        idx = [i for i, p in enumerate(lab) if p == c]
        tot += collections.Counter(y[i] for i in idx).most_common(1)[0][1]
    return tot / len(y)


def top_forms(texts, topn=8, min_count=3):
    cnt = collections.Counter()
    for t in texts:
        for w in jieba.lcut(t):
            if C.TARGET_CHAR in w:
                cnt[w] += 1
        for i, ch in enumerate(t):
            if ch == C.TARGET_CHAR:
                for L in (1, 2):
                    if i + L + 1 <= len(t):
                        cnt[t[i:i + L + 1]] += 1
    items = [(g, c) for g, c in cnt.items() if c >= min_count and len(g) >= 2]
    items.sort(key=lambda x: (-x[1], -len(x[0])))
    out, seen = [], []
    for g, c in items:
        if any(g in s for s in seen):
            continue
        out.append((g, c))
        seen.append(g)
        if len(out) >= topn:
            break
    return out


def main():
    import csv
    meta = {r["clean_id"]: r for r in rj(CLEAN)}
    Z, ids, Vt = load()
    texts = [meta[i].get("text_norm", "") for i in ids]
    y = [form_label(t) for t in texts]
    years = [(meta[i].get("ctime_iso") or "")[:4] for i in ids]
    print(f"items={len(ids)}  form distribution="
          f"{dict(collections.Counter(y))}")

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    labs = {}
    for k in (5, 7):
        lab = KMeans(n_clusters=k, n_init=30,
                     random_state=C.RANDOM_SEED).fit_predict(Z)
        labs[k] = lab
        pur = purity(y, lab.tolist())
        ari = adjusted_rand_score(y, lab)
        nmi = normalized_mutual_info_score(y, lab)
        print(f"\nk={k}: purity_vs_forms={pur:.4f} ARI={ari:.4f} NMI={nmi:.4f}")

        tab = collections.defaultdict(collections.Counter)
        for l_, f_ in zip(lab.tolist(), y):
            tab[l_][f_] += 1
        p = os.path.join(C.TABLES, f"forms_k{k}_contingency.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "size"] + ORDER + ["dominant"])
            for c in sorted(tab):
                dom = tab[c].most_common(1)[0]
                w.writerow([c, sum(tab[c].values())] + [tab[c][o] for o in ORDER]
                           + [f"{dom[0]} ({100*dom[1]/sum(tab[c].values()):.0f}%)"])

        p = os.path.join(C.TABLES, f"forms_k{k}_topforms.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "size", "rank", "form", "count", "per_1000"])
            for c in sorted(tab):
                idx = np.where(lab == c)[0]
                ts = [texts[i] for i in idx]
                for r_, (g, n) in enumerate(top_forms(ts), 1):
                    w.writerow([c, len(idx), r_, g, n,
                                round(1000 * n / len(idx), 1)])

        print(f"{'cl':>4}{'size':>7}" + "".join(f"{o:>9}" for o in ORDER)
              + "   dominant / top forms")
        for c in sorted(tab):
            idx = np.where(lab == c)[0]
            ts = [texts[i] for i in idx]
            dom = tab[c].most_common(1)[0]
            tf = ", ".join(f"{g}×{n}" for g, n in top_forms(ts, 4))
            print(f"{c:>4}{len(idx):>7}"
                  + "".join(f"{tab[c][o]:>9}" for o in ORDER)
                  + f"   {dom[0]}({100*dom[1]/len(idx):.0f}%) | {tf}")

    # k=5 yearly composition
    lab5 = labs[5]
    tab = collections.defaultdict(lambda: collections.Counter())
    for l_, yr in zip(lab5.tolist(), years):
        if yr:
            tab[yr][l_] += 1
    p = os.path.join(C.TABLES, "forms_k5_by_year.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["year", "n"] + [f"cluster_{c}" for c in range(5)])
        for yr in sorted(tab):
            w.writerow([yr, sum(tab[yr].values())]
                       + [tab[yr][c] for c in range(5)])

    out = os.path.join(C.PROCESSED, "beng_forms.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            r = dict(meta[cid])
            r["form_group"] = y[i]
            r["cluster_k5"] = int(labs[5][i])
            r["cluster_k7"] = int(labs[7][i])
            r["representation"] = "masked_last4mean"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    Xc = Z - Z.mean(axis=0, keepdims=True)
    _, S, Vt2 = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (len(Xc) - 1)
    ratio = ev / ev.sum()
    coords = Xc @ Vt2[:2].T
    cmap = plt.get_cmap("tab10")
    for k, name in ((5, "fig1b_k5_clusters"), (7, "fig1c_k7_clusters")):
        lab = labs[k]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for i, c in enumerate(sorted(set(lab.tolist()))):
            m = lab == c
            ax.scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.65,
                       color=cmap(i % 10),
                       label=f"cluster {c} (n={int(m.sum())})",
                       edgecolors="none")
        ax.set_xlabel(f"PC1（{ratio[0]*100:.1f}%）")
        ax.set_ylabel(f"PC2（{ratio[1]*100:.1f}%）")
        ax.set_title(f"「绷」按词形聚成 {k} 类"
                     f"（遮罩表示，词形纯度 {purity(y, lab.tolist()):.3f}）")
        ax.legend(fontsize=8, frameon=False, markerscale=2.5, ncol=2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(C.FIGURES, f"{name}.{ext}"), dpi=300,
                        bbox_inches="tight")
        print(f"-> {name}.png/.pdf")

    print("->", out)


if __name__ == "__main__":
    main()
