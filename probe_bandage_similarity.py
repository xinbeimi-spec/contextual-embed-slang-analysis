# -*- coding: utf-8 -*-
"""诊断：为什么 target 表示聚不出「绷带」，而 masked 可以？

背景（2026-11-01 用户提问）
  result5 的对比里，target 表示的「物理义」变成 0，原因是 323 条「绷带」在 target 的
  密度地形里没有形成独立盆地，被并进了以「其他/提及型」为主的大簇。
  但**词形层面**它们当然还在：物理义本来就是按「绷带 / 绷紧」这两个词形定义的。
  所以这里要回答的是三个可量化的问题：

  Q1 同一条「绷带」句子，它的**最近邻**是不是另一条绷带句？（邻近率）
     邻近率高 = 该词形在自己的邻域里自成一团（和 k 无关，纯几何）
  Q2 绷带组**内部**的平均余弦 vs 它和「绷住/绷不住/难绷」的平均余弦（可分性）
  Q3 两种表示的「绷带组中心」与「绷住组中心」有多像？
     若 target 显著更近，说明 target 向量被共享的「绷」字分量拉在一起了。

  另外记录：两种表示下，323 条绷带句分别被分到哪个语义类别（这是「物理义=0」的直接原因）。

输入：data/processed/beng/emb2/{masked,target}
输出：logs/beng/probe_bandage_similarity.txt
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cdpc_repr_compare_beng import load_repr, form_label, BENG_PROC, LOGDIR

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(BENG_PROC, "beng_clean.jsonl")
FORMS = ["绷带", "绷紧", "绷住", "绷不住", "难绷", "其他"]


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def nn_same_form_rate(Z, forms):
    """每条句子的最近邻（余弦最大，排除自己）是否同一词形。"""
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)        # 排除自己
    nn = np.argmax(S, axis=1)
    same = sum(1 for i, j in enumerate(nn) if forms[i] == forms[j])
    per = collections.defaultdict(list)
    for i, j in enumerate(nn):
        per[forms[i]].append(1 if forms[i] == forms[j] else 0)
    return same / len(forms), {f: float(np.mean(v)) for f, v in per.items()}


def group_cosines(Z, forms):
    """返回：每组的组内平均余弦、以及两两组的中心余弦。"""
    idx = {f: [i for i, x in enumerate(forms) if x == f] for f in FORMS}
    S = Z @ Z.T
    within = {}
    for f in FORMS:
        ii = idx[f]
        if len(ii) < 5:
            continue
        sub = S[np.ix_(ii, ii)]
        m = ~np.eye(len(ii), dtype=bool)
        within[f] = float(sub[m].mean())
    cents = {f: Z[idx[f]].mean(axis=0) for f in FORMS if len(idx[f]) >= 5}
    cent = {f: c / (np.linalg.norm(c) + 1e-12) for f, c in cents.items()}
    pairs = {}
    fs = sorted(cent)
    for a in range(len(fs)):
        for b in range(a + 1, len(fs)):
            pairs[(fs[a], fs[b])] = float(cent[fs[a]] @ cent[fs[b]])
    return within, pairs, {f: len(idx[f]) for f in FORMS}


def main():
    clean = {r["clean_id"]: r for r in rj(CLEAN)}
    L = []

    # ---- 两种表示下，323 条绷带句分别被分到哪个类别
    L.append("=== Q0 323 条「绷带」句在两次聚类里被分到哪个类别 ===")
    for name, path in (("masked", os.path.join(BENG_PROC, "beng_cdpc.jsonl")),
                       ("target", os.path.join(BENG_PROC, "beng_cdpc_target.jsonl"))):
        if not os.path.exists(path):
            L.append(f"  {name}: 缺少 {os.path.basename(path)}，跳过")
            continue
        rows = [r for r in rj(path) if form_label(r.get("text_norm") or "") == "绷带"]
        cnt = collections.Counter(r.get("cdpc_category") for r in rows)
        L.append(f"  {name}: 绷带条目 {len(rows)} 条 -> {dict(cnt)}")
    L.append("")

    # ---- 几何诊断
    for tag, sub in (("masked", "masked"), ("target", "target")):
        root = os.path.join(BENG_PROC, "emb2", sub)
        Z, ids = load_repr(root)
        forms = [form_label(clean[c]["text_norm"]) for c in ids]
        L.append(f"=== {tag}（N={len(ids)}）===")

        rate, per = nn_same_form_rate(Z, forms)
        L.append(f"  Q1 最近邻同词形率（全体）：{rate:.3f}")
        for f in FORMS:
            if f in per:
                L.append(f"       {f:<5} {per[f]:.3f}")

        within, pairs, sizes = group_cosines(Z, forms)
        L.append("  Q2 组内平均余弦：")
        for f in FORMS:
            if f in within:
                L.append(f"       {f:<5} n={sizes[f]:<5} {within[f]:+.3f}")
        L.append("  Q3 组中心两两余弦（越小越分得开）：")
        for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
            L.append(f"       {a:<5} ~ {b:<5} {v:+.3f}")
        # 关键可分性：绷带 vs 其余
        bd = [i for i, f in enumerate(forms) if f == "绷带"]
        rest = [i for i, f in enumerate(forms) if f != "绷带"]
        S = Z @ Z.T
        cross = float(S[np.ix_(bd, rest)].mean())
        L.append(f"  绷带 组内 {within['绷带']:+.3f}   绷带×其余 {cross:+.3f}   "
                 f"可分性差 {within['绷带'] - cross:+.3f}")
        L.append("")

    txt = "\n".join(L)
    os.makedirs(LOGDIR, exist_ok=True)
    p = os.path.join(LOGDIR, "probe_bandage_similarity.txt")
    open(p, "w", encoding="utf-8").write(txt)
    print(txt)
    print("->", p)


if __name__ == "__main__":
    main()
