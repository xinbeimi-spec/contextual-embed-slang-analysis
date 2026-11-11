# -*- coding: utf-8 -*-
"""用最新的 CSV 重新生成 result4/README.md 的「七、5 义项热力图」一节。

为什么用脚本生成：换了样本后，句子表和余弦矩阵都会变；手抄容易出错，也让
README 与图片/Excel 不一致。这个脚本每次都从 CSV 读，保证三者同源。
"""
import os
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "beng", "result4")
README = os.path.join(RES, "README.md")
SAMP = os.path.join(RES, "semantic_heatmap_5senses_samples.csv")
ALT = SAMP.replace(".csv", ".new.csv")
MAT = os.path.join(RES, "semantic_heatmap_5senses_matrix.csv")


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def main():
    samp = ALT if (os.path.exists(ALT) and
                   (not os.path.exists(SAMP) or
                    os.path.getmtime(ALT) > os.path.getmtime(SAMP))) else SAMP
    samples = read_csv(samp)
    matrix = read_csv(MAT)

    head, rows = samples[0], samples[1:]
    # 列序：序号 义项 短标签 完整句子 原始文本 clean_id 时间 年份 簇号 词形 字数 点赞
    cats = [r[1] for r in rows]
    sentences = [r[3] for r in rows]

    # 把「非对角」的每格连同行列名一起收集，避免用 index 反查行列（原来那个写法越界了）
    pairs = []
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix)):
            if i != j:
                pairs.append((float(matrix[i][j]), matrix[i][0], matrix[0][j]))
    pairs.sort()
    mn, mi, mj = pairs[0]
    mx, xi, xj = pairs[-1]
    mean = sum(v for v, _a, _b in pairs) / len(pairs)

    L = []
    L.append("## 七、5 义项随机抽样语义热力图")
    L.append("")
    L.append("**做了什么**：从 5 个**义项**（物理义 / 忍住义 / 忍不住义 / 评价义 / 紧绷义）"
             "各自的候选库里**随机抽 1 句**，两两算余弦相似度，画 5×5 热力图；"
             "完整句子另存 Excel。")
    L.append("「非义项用法」不是词义（提及而非使用），**不参与抽样**。")
    L.append("")
    L.append("**抽样规则（可复现）**：来源为本目录同一批标注 "
             "`data/processed/beng/beng_cdpc.jsonl`；只要 `is_canonical=1` + "
             "`context_sufficiency=1` + 长度 12–80 字，并排除 `excluded_from_typology`；"
             "随机种子 **20261103**。")
    L.append("")
    L.append("**内容过滤（2026-11-03 新增）**：物理义（绷带）的评论大量来自动漫/擦边视频，"
             "随机抽容易抽到不适合公开汇报的句子。因此候选池先按固定词表剔除含性暗示/低俗词的条目"
             "（屁股 / 内衣 / 那啥 / 胸 / 涩 / 色情 / 裸 / 下体 / 搞黄 / 黄图 / 淫 / 骚 / 床 / 舔 / 揉），"
             "**再在剩余池里随机抽**——这是固定的、可复现的过滤规则，不是手工挑句子。"
             "剔除条数：物理义 10、忍住义 1、忍不住义 2、评价义 1、紧绷义 0。")
    L.append("")
    L.append("**抽到的 5 句**")
    L.append("")
    L.append("| # | 义项 | 完整句子 |")
    L.append("|---|---|---|")
    for i, (c, s) in enumerate(zip(cats, sentences), 1):
        L.append(f"| {i} | {c} | {s} |")
    L.append("")
    L.append("**余弦矩阵**（masked 表示，与主结果同一空间）")
    L.append("")
    L.append("|  | " + " | ".join(f"句{i}" for i in range(1, len(rows) + 1)) + " |")
    L.append("|---|" + "---|" * len(rows))
    for i in range(1, len(matrix)):
        vals = []
        for j in range(1, len(matrix)):
            v = float(matrix[i][j])
            s = f"{v:+.3f}"
            if i == j:
                s = "1.000"
            vals.append(s)
        L.append(f"| {matrix[i][0]} | " + " | ".join(vals) + " |")
    L.append("")
    L.append(f"非对角均值 **{mean:+.3f}**；最大 {mx:+.3f}（{xi} ~ {xj}），"
             f"最小 {mn:+.3f}（{mi} ~ {mj}）。")
    L.append("")
    L.append("**怎么读（重要）**：句向量已**去均值 + 移除第一主成分**，矩阵**绝对值不能当阈值**，"
             "只有相对高低有意义。单句抽样只代表这一句，不能代表整个义项；"
             "义项层面的证据要看 `typology_sizes.csv` 与 `cdpc_fig1_clusters.png`。")
    L.append("")
    L.append("**本节的产物**")
    L.append("")
    L.append("| 文件 | 内容 |")
    L.append("|---|---|")
    L.append("| `fig2_semantic_heatmap_5senses.png/.pdf` | 5×5 热力图（坐标轴为短标签） |")
    L.append("| `semantic_heatmap_5senses.xlsx` | **完整句子**表 + 余弦矩阵 + 说明（3 个工作表） |")
    L.append("| `semantic_heatmap_5senses_samples.csv` | 5 条样本的全部字段（Excel 的来源） |")
    L.append("| `semantic_heatmap_5senses_matrix.csv` | 余弦矩阵（Excel 的来源） |")
    L.append("| `semantic_heatmap_5senses_meta.json` | 抽样参数、候选池大小、内容过滤规则（留痕） |")
    L.append("")
    L.append("生成命令：")
    L.append("")
    L.append("```powershell")
    L.append("& $PY src\\analyze\\make_semantic_heatmap.py                          # 首次抽样")
    L.append("& $PY src\\analyze\\make_semantic_heatmap.py --resample 物理义 --seed 20261103  # 只换一个义项")
    L.append("& $DEP src\\analyze\\build_heatmap_xlsx.py                            # 重建 Excel")
    L.append("```")

    new_section = "\n".join(L) + "\n"
    s = open(README, encoding="utf-8").read().replace("\r\n", "\n")
    i = s.find("## 七、")
    s = (s[:i] if i >= 0 else s.rstrip("\n") + "\n\n") + new_section
    open(README, "w", encoding="utf-8", newline="\n").write(s)
    print("README 第七节已按最新 CSV 重写")
    print("使用的样本文件：", os.path.basename(samp))


if __name__ == "__main__":
    main()
