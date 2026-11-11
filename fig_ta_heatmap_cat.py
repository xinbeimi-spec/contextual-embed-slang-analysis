# -*- coding: utf-8 -*-
"""Single-panel masked heatmap for N random sentences of ONE category.

Same style as cdpc_fig2_semantic_heatmap10 (the 10 「塌了」 sentences): MASKED
representation only (the space the clustering used), sentences running down the
left edge one per row, cosine values printed in the cells.

Different purpose: here the 10 sentences come from a SINGLE item-level category,
so the matrix answers "how homogeneous is this category inside the clustering
space?" — if the block is uniformly warm, the category is a tight constructional
class; if it is patchy, the category label is covering different contexts.

Usage
  python src/analyze/fig_ta_heatmap_cat.py --category "物理·坍塌" --n 10 --tag tanta
Outputs
  results/<slug>/<round>/cdpc_fig2b_semantic_heatmap10_<tag>.png/.pdf
  results/<slug>/<round>/semantic_heatmap10_<tag>.csv
  logs/<slug>/samples_<tag>.txt
"""
import os
import sys
import json
import csv
import random
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cdpc_pipeline import load_masked

sys.stdout.reconfigure(encoding="utf-8")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wrap(t, width=17, maxlines=2):
    t = (t or "").strip()
    lines, i = [], 0
    while i < len(t) and len(lines) < maxlines:
        lines.append(t[i:i + width])
        i += width
    if i < len(t):
        lines[-1] = lines[-1][:max(width - 1, 1)] + "…"
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--round", default="result4",
                    help="which round's item labels to sample from")
    a = ap.parse_args()

    lab = rj(os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_{a.round}.jsonl"))
    pool = [r for r in lab if r.get("cdpc_category") == a.category]
    if len(pool) < a.n:
        raise SystemExit(f"only {len(pool)} items in {a.category}")
    random.seed(a.seed)
    sample = random.sample(pool, a.n)
    print(f"category={a.category}  pool={len(pool)}  sampled={len(sample)}")

    ids = [r["clean_id"] for r in sample]
    texts = [r["text_norm"] for r in sample]

    Zm, mids = load_masked()
    pos = {c: i for i, c in enumerate(mids)}
    Z10 = Zm[[pos[c] for c in ids]]
    S = Z10 @ Z10.T

    labels = [f"S{i+1}" for i in range(len(ids))]
    outp = os.path.join(C.TABLES, f"semantic_heatmap10_{a.tag}.csv")
    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "cosine", "text_row"])
        for i in range(len(ids)):
            for j in range(len(ids)):
                w.writerow([labels[i], labels[j], round(float(S[i, j]), 4), texts[i]])
    sample_log = os.path.join(C.LOGS, f"samples_{a.tag}.txt")
    with open(sample_log, "w", encoding="utf-8") as f:
        f.write(f"category={a.category}  seed={a.seed}  n={len(ids)}\n\n")
        for i, (cid, t) in enumerate(zip(ids, texts), 1):
            f.write(f"S{i}\t{cid}\t{t}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-0.8, vmax=1.0)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticklabels([f"S{i+1}  " + wrap(texts[i]) for i in range(len(ids))],
                       fontsize=7.0, linespacing=1.25)
    ax.tick_params(axis="y", length=0, pad=4)
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = S[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if (v > 0.45 or v < -0.5) else "#222222")
    off = (S.sum() - np.trace(S)) / (len(ids) ** 2 - len(ids))
    ax.set_title(f"「{C.TARGET_WORD}」随机 {len(ids)} 条「{a.category}」句的两两余弦相似度\n"
                 f"masked 表示·CDPC 聚类空间（该组内平均 {off:.3f}）", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("余弦相似度", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES,
                         f"cdpc_fig2b_semantic_heatmap10_{a.tag}.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    print("->", outp)
    print("->", sample_log)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(np.round(S, 2))


if __name__ == "__main__":
    main()
