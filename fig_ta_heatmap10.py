# -*- coding: utf-8 -*-
"""Single-panel semantic heatmap for the 10 sampled 「塌了」 sentences.

Only the MASKED representation is drawn (2026-10-31, user decision): the
clustering was done in the masked space, so the figure must show the same
space. The target-space matrix is still written to the CSV for reference but
is no longer plotted.

Layout: sentence text runs down the left edge, one entry per row, so the figure
needs no caption block at the bottom.

Outputs
  results/<slug>/<round>/cdpc_fig2_semantic_heatmap10.png/.pdf
  results/<slug>/<round>/semantic_heatmap10.csv
Run: python src/analyze/fig_ta_heatmap10.py
"""
import os
import re
import sys
import json
import csv
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cdpc_pipeline import load_masked

sys.stdout.reconfigure(encoding="utf-8")

SAMPLES = os.path.join(C.LOGS, "pending_tale_samples.txt")
TARGET_DIR = C.EMB_DIR


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def read_sample_ids():
    ids = []
    for line in open(SAMPLES, encoding="utf-8"):
        m = re.search(r"sid=([0-9a-f]{16})", line)
        if m:
            ids.append(m.group(1))
    return ids


def load_target_rows(ids):
    index = {r["clean_id"]: r for r in rj(os.path.join(TARGET_DIR, "index.jsonl"))
             if r.get("status") == "ok"}
    by_shard = collections.defaultdict(list)
    for cid in ids:
        r = index[cid]
        by_shard[r["shard"]].append(r)
    out = {}
    for si, recs in by_shard.items():
        emb = np.load(os.path.join(TARGET_DIR, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in recs:
            out[r["clean_id"]] = emb[r["row"]].astype(np.float32)
    return np.stack([out[c] for c in ids])


def apply_centering(X):
    p = np.load(os.path.join(TARGET_DIR, "center_params.npz"))
    Z = X - p["mu"]
    comp = p["pca_components"]
    if comp.shape[0]:
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


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
    ids = read_sample_ids()
    rows = {r["clean_id"]: r for r in rj(
        os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_{C.RESULT_SET}.jsonl"))}
    texts = [rows[c]["text_norm"] for c in ids]

    Zm, mids = load_masked()
    pos = {c: i for i, c in enumerate(mids)}
    Zm10 = Zm[[pos[c] for c in ids]]
    Sm = Zm10 @ Zm10.T

    # target matrix is kept in the CSV only
    Zt10 = apply_centering(load_target_rows(ids)[:, 9:13, :].mean(axis=1))
    St = Zt10 @ Zt10.T

    labels = [f"S{i+1}" for i in range(len(ids))]
    outp = os.path.join(C.TABLES, "semantic_heatmap10.csv")
    with open(outp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["panel", "row", "col", "cosine"])
        for tag, M in (("masked", Sm), ("target", St)):
            for i in range(len(ids)):
                for j in range(len(ids)):
                    w.writerow([tag, labels[i], labels[j], round(float(M[i, j]), 4)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(Sm, cmap="RdBu_r", vmin=-0.8, vmax=1.0)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ylabels = [f"S{i+1}  " + wrap(texts[i]) for i in range(len(ids))]
    ax.set_yticklabels(ylabels, fontsize=7.0, linespacing=1.25)
    ax.tick_params(axis="y", length=0, pad=4)
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = Sm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if (v > 0.45 or v < -0.5) else "#222222")
    ax.set_title("「塌」10 条同形句（均含「塌了」）的两两余弦相似度\n"
                 "masked 表示·CDPC 聚类空间（减均值 + 去首主成分 + L2 归一化）",
                 fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("余弦相似度", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_fig2_semantic_heatmap10.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    print("->", outp)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(np.round(Sm, 2))


if __name__ == "__main__":
    main()
