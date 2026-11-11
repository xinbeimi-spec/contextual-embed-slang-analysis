# -*- coding: utf-8 -*-
"""Stratified 10-sentence heatmap for the 「塌了」 sense.

Motivation (user, 2026-11-03): the earlier 「塌了」 heatmap was a plain random
draw and happened to be 9 physical + 1 odd one. This time the sample is
DELIBERATELY stratified so the figure can show whether the representation
separates senses inside one surface form:

  4 sentences  : fandom / 塌房-like or other metaphorical uses of 塌了
  6 sentences  : physical collapse (楼/山/墙/桥/棚/平台…)

Selection uses cues for *sampling only*; the categories are not changed.
Rows are ordered metaphor-first so the block structure is readable.

Outputs (results/<slug>/result7/)
  cdpc_fig2d_semantic_heatmap10_tale_strat.png/.pdf
  semantic_heatmap10_tale_strat.csv
  logs/<slug>/samples_tale_strat.txt
Run: python src/analyze/fig_ta_heatmap_tale_strat.py --seed 11
"""
import os
import re
import sys
import json
import csv
import random
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cdpc_repr_compare import load_repr, PHYSICAL_FRAME

sys.stdout.reconfigure(encoding="utf-8")

# cues used ONLY to pick candidates (not to label the corpus)
METAPHOR_CUE = re.compile(
    r"爱豆|偶像|明星|艺人|网红|粉丝|追星|打榜|应援|人设|内娱|塌房|塌不塌|塌没塌|"
    r"翻车|脱粉|糊|吃瓜|恋情|出轨|绯闻|私生活|官宣|嗑|路人|黑料|公关|洗白|"
    r"市场|行情|口碑|联盟|王朝|帝国|神话|信仰|价值观|三观")
PHYS_CUE = re.compile(PHYSICAL_FRAME)

N_META = 4
N_PHYS = 6


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sanitize(t):
    """Drop characters the CJK font cannot render (math symbols, emoji-faces)."""
    keep = []
    for ch in t or "":
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF or 0x3000 <= o <= 0x303F
                or 0xFF00 <= o <= 0xFFEF or o < 0x80
                or ch in "，。！？、；：“”‘’（）…—《》【】"):
            keep.append(ch)
    return "".join(keep)


def wrap(t, width=17, maxlines=2):
    t = sanitize(t)
    lines, i = [], 0
    while i < len(t) and len(lines) < maxlines:
        lines.append(t[i:i + width])
        i += width
    if i < len(t):
        lines[-1] = lines[-1][:max(width - 1, 1)] + "…"
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--round", default="result7")
    ap.add_argument("--label-round", default="result4")
    a = ap.parse_args()

    clean = {r["clean_id"]: r for r in rj(
        os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl"))}
    lab = rj(os.path.join(C.PROCESSED,
                          f"{C.WORD_SLUG}_cdpc_{a.label_round}.jsonl"))
    pool = [r for r in lab if r.get("cdpc_category") == "物理·塌了"]
    print(f"「物理·塌了」池子: {len(pool)}")

    meta, phys = [], []
    for r in pool:
        t = clean[r["clean_id"]]["text_norm"]
        if "塌了" not in t:
            continue
        has_m, has_p = bool(METAPHOR_CUE.search(t)), bool(PHYS_CUE.search(t))
        if has_m and not has_p and 12 <= len(t) <= 120:
            meta.append(r["clean_id"])
        elif has_p and not has_m and 12 <= len(t) <= 120:
            phys.append(r["clean_id"])
    print(f"  隐喻/塌房候选: {len(meta)}   物理候选: {len(phys)}")

    random.seed(a.seed)
    pick_m = random.sample(meta, N_META)
    pick_p = random.sample(phys, N_PHYS)
    ids = pick_m + pick_p
    groups = ["塌房/隐喻"] * N_META + ["物理"] * N_PHYS
    texts = [clean[c]["text_norm"] for c in ids]

    # ---- vectors (masked = the clustering space)
    Zm, mids = load_repr(C.EMB_MASK_DIR)
    pos = {c: i for i, c in enumerate(mids)}
    Z = Zm[[pos[c] for c in ids]]
    S = Z @ Z.T

    outdir = os.path.join(C.RESULT_ROOT, a.round)
    os.makedirs(outdir, exist_ok=True)
    tabs = C.TABLES

    labels = [f"S{i+1}" for i in range(len(ids))]
    csv_p = os.path.join(tabs, "semantic_heatmap10_tale_strat.csv")
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "cosine", "group_row", "text_row"])
        for i in range(len(ids)):
            for j in range(len(ids)):
                w.writerow([labels[i], labels[j], round(float(S[i, j]), 4),
                            groups[i], texts[i]])

    log_p = os.path.join(C.LOGS, "samples_tale_strat.txt")
    with open(log_p, "w", encoding="utf-8") as f:
        f.write(f"seed={a.seed}  池 {len(pool)}  隐喻 {len(meta)}  物理 {len(phys)}\n\n")
        for i, (cid, g, t) in enumerate(zip(ids, groups, texts), 1):
            f.write(f"S{i}\t{g}\t{cid}\t{(clean[cid].get('ctime_iso') or '')[:10]}\t{t}\n")

    # ---- group means
    im = list(range(N_META))
    ip = list(range(N_META, len(ids)))
    def msub(aa, bb):
        v = [S[i, j] for i in aa for j in bb if i != j]
        return float(np.mean(v)) if v else float("nan")
    mm, pp, mp = msub(im, im), msub(ip, ip), msub(im, ip)
    print(f"within 隐喻={mm:.3f}  within 物理={pp:.3f}  cross={mp:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    imshow = ax.imshow(S, cmap="RdBu_r", vmin=-0.8, vmax=1.0)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticklabels([f"S{i+1}  " + wrap(texts[i]) for i in range(len(ids))],
                       fontsize=7.0, linespacing=1.25)
    # colour-code the row labels so the two strata are legible without a legend
    for i, tl in enumerate(ax.get_yticklabels()):
        tl.set_color("#C0392B" if i < N_META else "#1F4E79")
    ax.tick_params(axis="y", length=0, pad=4)
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = S[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if (v > 0.45 or v < -0.5) else "#222222")
    # separators between the two strata
    for x in (N_META - 0.5,):
        ax.axhline(x, color="#333333", lw=1.1, ls="--")
        ax.axvline(x, color="#333333", lw=1.1, ls="--")
    ax.set_title("「塌」10 条「塌了」句：4 条塌房/隐喻（红）+ 6 条物理（蓝）\n"
                 f"masked 余弦相似度；组内均值 隐喻={mm:.3f} / 物理={pp:.3f}，跨组={mp:.3f}",
                 fontsize=9.5)
    cb = fig.colorbar(imshow, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("余弦相似度", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(tabs, f"cdpc_fig2d_semantic_heatmap10_tale_strat.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    print("->", csv_p)
    print("->", log_p)
    for i, (g, t) in enumerate(zip(groups, texts), 1):
        print(f"S{i} [{g}] {t[:60]}")


if __name__ == "__main__":
    main()
