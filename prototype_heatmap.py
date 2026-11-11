# -*- coding: utf-8 -*-
"""Stage 4c: representative-sentence similarity heatmap (traditional vs new).

Selection rule (must be stated in the paper):
  * "traditional" pool  = surface hints A1_bandage + A1_tauten  (绷带 / 绷紧)
  * "new" pool          = surface hint  B_bengzhu             (绷住 / 绷不住)
  * a representative sentence is the one closest to its pool centroid by
    cosine, but picks are forced to be mutually dissimilar (pairwise cosine
    below DIVERSITY), so the panel is not five paraphrases of one line.
  * similarity is computed on the SAME centred 768-d vectors used for
    clustering (not on the 2-D projection).

Outputs
  results/tables/prototype_5v5_sentences.csv
  results/tables/prototype_5v5_cosine.csv
  results/figures/fig2_similarity_heatmap.png / .pdf
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

EMB = C.EMB_DIR
CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")

TRAD_HINTS = ("A1_bandage", "A1_tauten")
NEW_HINTS = ("B_bengzhu",)
DIVERSITY = 0.90


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pick(Xn, cand_idx, n, texts):
    if len(cand_idx) <= n:
        return list(cand_idx)
    C0 = Xn[cand_idx].mean(axis=0)
    C0 /= np.linalg.norm(C0) + 1e-12
    sims = Xn[cand_idx] @ C0
    order = np.argsort(-sims)
    picked = []
    for k in order:
        i = int(cand_idx[k])
        if all(float(Xn[i] @ Xn[j]) < DIVERSITY for j in picked):
            picked.append(i)
        if len(picked) >= n:
            break
    if len(picked) < n:                      # relax if the pool is too tight
        for k in order:
            i = int(cand_idx[k])
            if i not in picked:
                picked.append(i)
            if len(picked) >= n:
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    z = np.load(os.path.join(EMB, "centered.npz"))
    X, ids = z["X"].astype(np.float32), list(z["ids"])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rows = {r["clean_id"]: r for r in rj(CLEAN)}

    hint = np.array([rows[i].get("sense_hint", "unknown") for i in ids])
    suff = np.array([rows[i].get("context_sufficiency", 0) for i in ids])

    trad = np.where(np.isin(hint, TRAD_HINTS) & (suff == 1))[0]
    new = np.where(np.isin(hint, NEW_HINTS) & (suff == 1))[0]
    print(f"pools: traditional={len(trad)}  new={len(new)}")

    i_tr = pick(Xn, trad, a.n, rows)
    i_new = pick(Xn, new, a.n, rows)

    sel = i_tr + i_new
    labels = [f"T{j+1}" for j in range(len(i_tr))] + \
             [f"N{j+1}" for j in range(len(i_new))]
    S = Xn[sel] @ Xn[sel].T

    import csv
    p1 = os.path.join(C.TABLES, "prototype_5v5_sentences.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "sense_hint", "ctime", "n_chars",
                    "text", "clean_id"])
        for lab, i in zip(labels, sel):
            r = rows[ids[i]]
            w.writerow([lab, "traditional" if lab.startswith("T") else "new",
                        r.get("sense_hint"), (r.get("ctime_iso") or "")[:10],
                        r.get("n_chars_norm"), r.get("text_norm"),
                        r.get("clean_id")])

    p2 = os.path.join(C.TABLES, "prototype_5v5_cosine.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label"] + labels)
        for r_, lab in enumerate(labels):
            w.writerow([lab] + [round(float(v), 4) for v in S[r_]])

    print(f"\n{'':6}" + "".join(f"{l:>8}" for l in labels))
    for r_, lab in enumerate(labels):
        print(f"{lab:6}" + "".join(f"{S[r_, c]:8.3f}" for c in range(len(labels))))

    # block means
    ntr = len(i_tr)
    blk_tt = S[:ntr, :ntr][np.triu_indices(ntr, 1)].mean()
    blk_nn = S[ntr:, ntr:][np.triu_indices(len(i_new), 1)].mean()
    blk_tn = S[:ntr, ntr:].mean()
    print(f"\nmean cosine  traditional-internal = {blk_tt:.4f}")
    print(f"mean cosine  new-internal         = {blk_nn:.4f}")
    print(f"mean cosine  traditional-vs-new   = {blk_tn:.4f}")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for r_ in range(len(labels)):
        for c_ in range(len(labels)):
            ax.text(c_, r_, f"{S[r_, c_]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(S[r_, c_]) > 0.55 else "black")
    ax.axhline(ntr - 0.5, color="k", lw=1.2)
    ax.axvline(ntr - 0.5, color="k", lw=1.2)
    ax.set_xlabel("代表性句子（T=传统义, N=网络新义）")
    ax.set_ylabel("代表性句子")
    ax.set_title("绷：传统义 vs 网络新义的上下文化向量余弦相似度")
    fig.colorbar(im, ax=ax, shrink=0.82, label="cosine similarity")
    fig.tight_layout()

    os.makedirs(C.FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = os.path.join(C.FIGURES, f"fig2_similarity_heatmap.{ext}")
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        print("->", fp)
    for p in (p1, p2):
        print("->", p)


if __name__ == "__main__":
    main()
