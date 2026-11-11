# -*- coding: utf-8 -*-
"""从「绷」的 5 个义项里各随机抽 1 句，做语义相似度热力图（写进 results/beng/result4）。

为什么要这样做
  result4 的主结果里有 6 个类别，但其中「非义项用法」不是词义（是"提及而非使用"，
  例如复读、玩梗、错别字）。所以这次只在**真正的 5 个义项**里抽样：
      物理义 / 忍住义 / 忍不住义 / 评价义 / 紧绷义
  每个义项随机抽 1 条，得到 5 句，两两算余弦相似度，画 5×5 热力图。
  热力图上的坐标轴只能放短标签（句子太长挤不下），所以**完整句子单独写进 Excel**。

抽样规则（可复现）
  * 来源：data/processed/beng/beng_cdpc.jsonl（result4 同一批标注，masked 表示 k=7）
  * 只要「可用」的句子：is_canonical=1（去重代表）、context_sufficiency=1（人能判定）、
    长度 12–80 字（太短没语境，太长轴标签放不下）
  * 排除 excluded_from_typology 的极零散义项
  * 随机数种子固定为 SEED（见下），所以抽到哪 5 句是**可复现的随机**，不是人工挑选

向量与相似度
  直接用 result4 聚类时用的那份 masked 向量（[MASK] 表示）：
  第 9–12 层平均 -> 减均值 -> 去掉第一主成分(ABTT k=1) -> L2 归一化
  余弦相似度 = 两个归一化向量的点积。
  ⚠️ 因为做过去均值，**矩阵的绝对值不能当阈值**，只能看相对高低（这一点必须写进图注）。

输出（results/beng/result4/）
  fig2_semantic_heatmap_5senses.png/.pdf   5×5 热力图
  semantic_heatmap_5senses_samples.csv     5 条样本句的完整信息（供 Excel 用）
  semantic_heatmap_5senses_matrix.csv      余弦矩阵（供 Excel 用）
  semantic_heatmap_5senses_meta.json       抽样参数与候选池大小（留痕）

运行
  $PY src\\analyze\\make_semantic_heatmap.py
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
from cdpc_repr_compare_beng import load_repr, BENG_PROC

sys.stdout.reconfigure(encoding="utf-8")

SEED = 20261101                     # 随机种子：改它就会抽到另一组句子（但同样可复现）
SENSES = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义"]   # 非义项用法不算词义
MIN_LEN, MAX_LEN = 12, 80           # 句子长度过滤（字数）
# 公开汇报用：候选池里排除含下列词的条目（性暗示/低俗），剔完再随机抽。
# 这不是"挑好看的句子"，而是一条固定的、可复现的内容过滤规则；
# 被剔除的条数会打印出来，写进 Excel「说明」表。
RISKY_WORDS = ("屁股", "内衣", "那啥", "胸", "涩", "色情", "裸", "下体",
               "搞黄", "黄图", "淫", "骚", "床", "上她", "舔", "揉")
OUTDIR = os.path.join(C.RESULTS, "beng", "result4")
SRC = os.path.join(BENG_PROC, "beng_cdpc.jsonl")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def short_label(text, win=18):
    """轴标签：以「绷」为中心截一段，保证目标字一定出现在标签里。"""
    idx = text.find("绷")
    if idx < 0 or len(text) <= win:
        return text if len(text) <= win else text[:win] + "…"
    half = win // 2
    lo = max(0, min(idx - half, len(text) - win))
    hi = lo + win
    return ("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def most_representative(pool, Z, pos, topn=8):
    """挑「最具代表性」的一条：到该义项质心的余弦最高。

    硬约束（先过滤再取最大，避免又抽到混多种表达的句子）：
      * 句子里「绷」只出现 1 次（排除 绷带+绷中绷+绷一绷 这种混用）
      * 单一词形：form_label 必须是 绷带 或 绷紧
      * 不含书名号《》（排除视频标题式玩梗）
      * 长度 12–60 字（自足、可读）
    返回 (best_record, [(score, text), ...] 前 topn 个候选)
    """
    import numpy as np
    from cdpc_repr_compare_beng import form_label as repr_form_label

    idx = [pos[r["clean_id"]] for r in pool if r["clean_id"] in pos]
    if not idx:
        return None, []
    C = Z[idx].mean(axis=0)                       # 该义项的质心
    C = C / (np.linalg.norm(C) + 1e-12)

    cands = []
    for r in pool:
        cid = r["clean_id"]
        if cid not in pos:
            continue
        t = r.get("text_norm") or ""
        if t.count("绷") != 1:                    # 只允许一个「绷」
            continue
        if "《" in t or "》" in t:
            continue
        if not (12 <= len(t) <= 60):
            continue
        if repr_form_label(t) not in ("绷带", "绷紧"):
            continue
        cands.append((float(Z[pos[cid]] @ C), r))
    if not cands:
        return None, []
    cands.sort(key=lambda x: -x[0])
    return cands[0][1], [(round(sc, 4), r.get("text_norm") or "") for sc, r in cands[:topn]]


def main():
    ap = argparse.ArgumentParser(description="5 义项各抽 1 句，画语义热力图")
    ap.add_argument("--resample", default="",
                    help="只重抽某一个义项（如 物理义），其余保持原样")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"随机种子（默认 {SEED}）")
    ap.add_argument("--keep-file", default="",
                    help="要保留的那份旧 samples.csv（默认用本目录已有的）")
    ap.add_argument("--representative-sense", default="",
                    help="这个义项不用随机抽，改为挑「最接近该义项质心」的句子")
    a = ap.parse_args()

    rows = rj(SRC)
    print(f"标注表：{len(rows)} 条")

    # ---- 每个义项各自的候选池（先按 clean_id 排序，保证抽样与文件顺序无关）
    pools = {}
    for s in SENSES:
        pool = [r for r in rows
                if r.get("cdpc_category") == s
                and not r.get("excluded_from_typology")
                and r.get("is_canonical") == 1
                and r.get("context_sufficiency") == 1
                and MIN_LEN <= len(r.get("text_norm") or "") <= MAX_LEN]
        pool.sort(key=lambda r: r["clean_id"])
        raw_n = len(pool)
        pool = [r for r in pool
                if not any(w in (r.get("text_norm") or "") for w in RISKY_WORDS)]
        pools[s] = pool
        print(f"  {s}: 候选 {len(pool)} 条（内容过滤剔除 {raw_n - len(pool)} 条）")

    # ---- 读出上一版样本（用于「只换一个义项」）
    keep_path = a.keep_file or os.path.join(OUTDIR, "semantic_heatmap_5senses_samples.csv")
    prev = {}
    if a.resample:
        if not os.path.exists(keep_path):
            raise SystemExit(f"要保留旧样本，但找不到 {keep_path}")
        with open(keep_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                prev[row["义项"]] = row["clean_id"]
        print(f"沿用旧样本的义项：{[x for x in SENSES if x in prev and x != a.resample]}")

    # 向量在这里就加载：代表性挑选要用质心（后面算矩阵时复用，不再重复加载）
    Z, ids = load_repr(os.path.join(BENG_PROC, "emb2", "masked"))
    pos = {cid: i for i, cid in enumerate(ids)}

    rng = random.Random(a.seed)
    picked = []
    for sense in SENSES:
        pool = pools[sense]
        if not pool:
            raise SystemExit(f"{sense} 没有候选句，无法抽样")

        # ⓪ 需要「最具代表性」的义项：先按质心余弦挑，且必须满足单一词形约束
        if sense == a.representative_sense:
            best, top = most_representative(pool, Z, pos)
            if best is None:
                raise SystemExit(f"{sense} 找不到满足「单一词形」约束的代表句")
            print(f"\n{sense}：按「到质心余弦最高」挑出代表句（候选 {len(pool)} 条）")
            for sc, t in top:
                print(f"    {sc:+.4f}  {t[:60]}")
            picked.append(best)
            continue

        # ① 不需要重抽、且旧样本仍在候选池里 -> 原样保留
        if a.resample and sense != a.resample and sense in prev:
            hit = [r for r in pool if r["clean_id"] == prev[sense]]
            if hit:
                picked.append(hit[0])
                continue

        # ② 需要重抽的义项 -> 从候选池里排除上一次用过的句子，保证一定换掉
        cand = pool
        if a.resample and sense == a.resample and sense in prev:
            old_id = prev[sense]
            cand = [r for r in pool if r["clean_id"] != old_id] or pool
            old = [r for r in rows if r["clean_id"] == old_id]
            print(f"重抽 {sense}：排除旧句 {old_id} -> 旧句为 "
                  f"{(old[0].get('text_norm') or '')[:40] if old else '(已不在表里)'}")
        picked.append(rng.choice(cand))           # 每个义项随机抽 1 条

    print(f"\n抽样完成（seed={a.seed}）")

    # ---- 取这 5 条句子的向量（Z/pos 已在上面加载）
    vecs, labels, cats = [], [], []
    for r in picked:
        cid = r["clean_id"]
        if cid not in pos:
            raise SystemExit(f"句子 {cid} 不在向量表里（可能 no_mask_token）")
        vecs.append(Z[pos[cid]])
        labels.append(short_label(r["text_norm"]))
        cats.append(r["cdpc_category"])
    V = np.stack(vecs)
    S = V @ V.T                                   # 归一化向量的点积 = 余弦相似度

    # ---- 保存样本句与矩阵（Excel 用）
    p_samp = os.path.join(OUTDIR, "semantic_heatmap_5senses_samples.csv")

    def _write_samples(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["序号", "义项", "短标签", "完整句子（清洗后）", "原始文本",
                        "clean_id", "发送时间", "年份", "簇号", "词形", "字数", "点赞"])
            for i, r in enumerate(picked, 1):
                t = r.get("text_norm") or ""
                w.writerow([i, r["cdpc_category"], short_label(t), t,
                            r.get("text_raw") or "", r["clean_id"],
                            (r.get("ctime_iso") or "")[:19],
                            (r.get("ctime_iso") or "")[:4],
                            r.get("cdpc_cluster"), r.get("sense_hint"),
                            len(t), r.get("like_count")])

    try:
        _write_samples(p_samp)
    except PermissionError:
        # 该 csv 被 Excel/记事本等程序打开时会锁住；改用备用名，不阻塞出图
        alt = p_samp.replace(".csv", ".new.csv")
        _write_samples(alt)
        print(f"!! {os.path.basename(p_samp)} 被占用（大概是用 Excel 打开了），"
              f"本次写到 {os.path.basename(alt)}；关掉那个文件后重跑即可覆盖正式文件")
        p_samp = alt

    p_mat = os.path.join(OUTDIR, "semantic_heatmap_5senses_matrix.csv")
    with open(p_mat, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["义项"] + [f"句{i}({c})" for i, c in enumerate(cats, 1)])
        for i, r in enumerate(picked):
            w.writerow([f"句{i+1}·{cats[i]}"] + [round(float(S[i, j]), 4)
                                                 for j in range(len(picked))])

    meta = {"seed": a.seed, "representation": "masked ([MASK]) last-4 mean, "
            "mean-centred + ABTT(k=1) + L2", "source": SRC,
            "pool_sizes": {s: len(pools[s]) for s in SENSES},
            "length_filter": [MIN_LEN, MAX_LEN],
            "nonsense_excluded": "非义项用法不是词义，未参与抽样",
            "content_filter": f"候选池先剔除含性暗示/低俗词的条目：{list(RISKY_WORDS)}",
            "selection_mode": (f"{a.representative_sense}=最具代表性（到该义项质心余弦最高；"
                               "约束：只含 1 个「绷」、单一词形绷带/绷紧、无书名号、12–60 字）"
                               if a.representative_sense else "全部随机")}
    with open(os.path.join(OUTDIR, "semantic_heatmap_5senses_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 画热力图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(picked)))
    ax.set_yticks(range(len(picked)))
    # 坐标轴只放「编号 + 义项 + 短片段」：完整句子放进 Excel，否则轴标签会互相压住
    ytick = [f"句{i+1}·{c}\n{lab[:12]}" for i, (c, lab) in enumerate(zip(cats, labels))]
    xtick = [f"句{i+1}\n{c}" for i, c in enumerate(cats)]
    ax.set_xticklabels(xtick, fontsize=8)
    ax.set_yticklabels(ytick, fontsize=8)
    for i in range(len(picked)):
        for j in range(len(picked)):
            ax.text(j, i, f"{S[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(S[i, j]) > 0.55 else "black")
    ax.set_title("「绷」5 个义项各随机 1 句的语义相似度（余弦）", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine")
    # 脚注用 fig.text 放在画布底部，避免和刻度标签重叠
    fig.text(0.5, 0.02,
             "注：句向量已去均值并移除第一主成分，矩阵绝对值不可作阈值，只有相对高低有意义。\n"
             + (f"抽样：其余义项随机 1 句（seed={a.seed}）；"
                f"{a.representative_sense}按「最接近该义项质心」选取（须为单一词形）。"
                if a.representative_sense else
                f"抽样：每个义项从候选库随机 1 句（seed={a.seed}）。")
             + "完整句子见同名 Excel。",
             ha="center", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.24)
    for ext in ("png", "pdf"):
        q = os.path.join(OUTDIR, f"fig2_semantic_heatmap_5senses.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    # ---- 控制台报告
    print("\n抽到的 5 句：")
    for i, r in enumerate(picked, 1):
        print(f"  {i}. [{r['cdpc_category']}] {(r.get('text_norm') or '')[:60]}")
    print("\n余弦矩阵：")
    print("      " + "".join(f"{c[:4]:>8}" for c in cats))
    for i in range(len(picked)):
        print(f"  句{i+1} " + "".join(f"{S[i, j]:>8.3f}" for j in range(len(picked))))
    same = [S[i, j] for i in range(len(picked)) for j in range(len(picked)) if i != j]
    print(f"\n跨义项（非对角）均值 {np.mean(same):+.3f}，"
          f"最大 {np.max(same):+.3f}，最小 {np.min(same):+.3f}")
    for p in (p_samp, p_mat):
        print("->", p)


if __name__ == "__main__":
    main()
