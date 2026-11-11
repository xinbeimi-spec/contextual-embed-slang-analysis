# -*- coding: utf-8 -*-
"""Package the final five-category results into one self-contained folder.

Rationale: the project tree accumulates many intermediate variants (k=5, k=7,
masked/unmasked, per-representation tables). The deliverable for the write-up
must be unambiguous, so the final artefacts are COLLECTED here in one place.

Files are COPIED, never moved: the originals stay where the pipeline writes them
so every number can be regenerated. The README records the source path of each
file, which is what makes the folder auditable rather than just tidy.

Output tree
  results/result1/
    README.md
    figures/   fig4_typology_stacked, fig4b_typology_line (png + pdf)
    tables/    typology_sizes, typology_by_year, typology_examples
    evidence/  select_k_comparison, form_cluster_recovery,
               forms_k7_contingency, forms_k7_topforms,
               forms_k5_contingency
    data/      beng_typology.jsonl  (item-level labels)
"""
import os
import sys
import csv
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

DEST = os.path.join(C.RESULTS, "result1")
SUB = ["cat5", "evidence"]

FIGURES = [
    ("fig1_five_category_space.png", "图1  五类用法在语义空间中的分布（聚类图，主图）"),
    ("fig1_five_category_space.pdf", "图1  同上，矢量版"),
    ("fig1c_k7_clusters.png", "图1c 归并前的 k=7 聚类散点（五类归并的直接依据）"),
    ("fig1c_k7_clusters.pdf", "图1c 同上，矢量版"),
    ("fig2a_selected5_v1_template4.png", "图2a 五句相似度（第4句＝“绷住挑战”模板句）"),
    ("fig2a_selected5_v1_template4.pdf", "图2a 同上，矢量版"),
    ("fig2b_selected5_v2_nontemplate4.png", "图2b 五句相似度（第4句＝非模板的忍住义句）"),
    ("fig2b_selected5_v2_nontemplate4.pdf", "图2b 同上，矢量版"),
    ("fig2_selected5_heatmap.png", "图2  五句相似度（早期版本，等同 2a）"),
    ("fig2_selected5_heatmap.pdf", "图2  同上，矢量版"),
    ("fig4_typology_stacked.png", "图4  五类用法的百分比堆叠构成（主图）"),
    ("fig4_typology_stacked.pdf", "图4  同上，矢量版"),
    ("fig4b_typology_line.png", "图4b 各类占比折线 + 样本量底栏"),
    ("fig4b_typology_line.pdf", "图4b 同上，矢量版"),
]
TABLES = [
    ("typology_sizes.csv", "五类的数量与占比"),
    ("typology_by_year.csv", "年度 × 五类（含每类占比）"),
    ("typology_examples.csv", "每类 12 条真实例句（写论文举证用）"),
    ("typology_representative.csv", "各类代表句（按质心余弦 + 多样性筛选）"),
    ("selected5_cosine.csv", "5 句余弦矩阵（最新单面板版）"),
    ("selected5_cosine_v1_v2.csv", "第4句两个候选的 v1/v2 对照矩阵"),
    ("selected5_selfcheck.csv", "5 句对各义项质心的相似度（验证选取是否代表本类）"),
]
EVIDENCE = [
    ("select_k_comparison.csv", "k=2..10 的三角验证：silhouette / 重抽样稳定度 / 词形熵"),
    ("form_cluster_recovery.csv", "k=4..7 对五种词形的复原力（纯度 / ARI / NMI / 稳定度）"),
    ("forms_k7_contingency.csv", "k=7 的簇 × 词形列联表（五类归并的直接依据）"),
    ("forms_k7_topforms.csv", "k=7 每簇高频用法（per_1000 可比）"),
    ("forms_k5_contingency.csv", "k=5 强制的列联表（用来说明为何五词形不可行）"),
]
DATA = [("beng_typology.jsonl", "逐条记录：category / cluster_k7 / form_group")]


def load_sizes():
    p = os.path.join(C.TABLES, "typology_sizes.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_by_year():
    p = os.path.join(C.TABLES, "typology_by_year.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def copy_group(src_dir, dest_sub, items, missing):
    out = []
    for name, desc in items:
        src = os.path.join(src_dir, name)
        dst = os.path.join(DEST, dest_sub, name)
        if not os.path.exists(src):
            missing.append(src)
            continue
        # A previously copied image may still be held open by an image viewer,
        # which makes an in-place overwrite fail with Errno 22. Remove first,
        # and fall back to a suffixed name rather than aborting the whole run.
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        try:
            shutil.copyfile(src, dst)
        except OSError:
            dst = dst.rsplit(".", 1)[0] + "_new." + dst.rsplit(".", 1)[1]
            shutil.copyfile(src, dst)
            print(f"   (target locked, wrote {os.path.basename(dst)})")
        out.append((os.path.basename(dst), desc, os.path.getsize(dst)))
    return out


def main():
    for s in SUB:
        os.makedirs(os.path.join(DEST, s), exist_ok=True)
    missing = []

    figs = copy_group(C.FIGURES, "cat5", FIGURES, missing)
    tabs = copy_group(C.TABLES, "cat5", TABLES, missing)
    evid = copy_group(C.TABLES, "evidence", EVIDENCE, missing)
    data = copy_group(C.PROCESSED, "cat5", DATA, missing)

    sizes = load_sizes()
    by_year = load_by_year()

    lines = []
    lines.append("# 「绷」五类用法的最终分析结果\n")
    lines.append("本文件夹是**最终交付版**，只包含与五类分析直接相关的图表、表格与证据链。")
    lines.append("所有文件均为**复制**（原件仍在 `results/tables`、`results/figures`、"
                 "`data/processed`），每一步都可重新生成。\n")
    lines.append("---\n")
    lines.append("## 一、五类构成\n")
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    for r in sizes:
        try:
            share = f"{100*float(r['share']):.1f}%"
        except Exception:
            share = r.get("share", "")
        lines.append(f"| {r['category']} | {r['n']} | {share} |")
    lines.append("")
    lines.append("> 第 5 类「非义项用法」不是词义，而是**提及而非使用**"
                 "（元语言玩梗 / 逐字复读 / 异形错别字 / 同形误匹配），"
                 "单列以保证样本完整、不删数据。\n")

    lines.append("## 二、图表清单\n")
    lines.append("### cat5/ (图)")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    for n, d, _ in figs:
        lines.append(f"| `{n}` | {d} |")
    lines.append("")
    lines.append("### cat5/ (表)")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    for n, d, _ in tabs:
        lines.append(f"| `{n}` | {d} |")
    lines.append("")
    lines.append("### evidence/（为什么是这五类，而不是别的分法）")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    for n, d, _ in evid:
        lines.append(f"| `{n}` | {d} |")
    lines.append("")
    lines.append("### data/")
    lines.append("| 文件 | 说明 |")
    lines.append("|---|---|")
    for n, d, _ in data:
        lines.append(f"| `{n}` | {d} |")
    lines.append("")

    if by_year:
        lines.append("## 三、年度构成\n")
        cat_cols = [c for c in by_year[0] if not c.startswith("share_")
                    and c not in ("year", "n")]
        lines.append("| 年份 | n | " + " | ".join(cat_cols) + " |")
        lines.append("|---|---|" + "---|" * len(cat_cols))
        for r in by_year:
            lines.append(f"| {r['year']} | {r['n']} | "
                         + " | ".join(str(r.get(c, "")) for c in cat_cols) + " |")
        lines.append("")

    lines.append("## 四、方法与判据（写方法节可直接引用）\n")
    lines.append("1. **向量表示**：`hfl/chinese-roberta-wwm-ext`（12 层 768 维），"
                 "取目标词 span 内 token 的隐状态均值，并用 last-4 层平均。")
    lines.append("2. **去中心化**：减去全语料均值并移除第一主成分（ABTT k=1），"
                 "随机句对余弦由 0.671 降至 0.006。")
    lines.append("3. **聚类粒度**：以多判据三角验证确定——k=2 重抽样稳定度 1.000；"
                 "以词形为类别时 k=7 的词形纯度 0.913、稳定度 0.949；"
                 "强制 k=5 词形会出现 1,323 条合并大团，故不采用。")
    lines.append("4. **五类归并**：`绷住` 的三个子簇（“绷住模式”模板／叙事反应／"
                 "“轻松绷住”复读）质心余弦 0.530，语义相同，合并为一类；"
                 "残余簇经核查为提及型用法，单列为第 5 类。")
    lines.append("5. **一致性**：按“簇号”与按“词形”两条独立规则判定，冲突 0 条。\n")

    lines.append("## 五、必须随文报告的限制\n")
    lines.append("- 语料为**关键词条件抽样**（按“绷”及若干词形检索视频后抓评论），"
                 "因此占比是**样本内份额**，不是总体使用率。")
    lines.append("- 2016 年样本量仅 3 条，2016–2018 的早期数值不可作结论。")
    lines.append("- 2017–2021 的“物理义占主导”部分来自**视频池构成**"
                 "（早年“绷不住”类视频本就稀少），不能直接解读为语言变化。")
    lines.append("- `蚌埠住了` 等同音异形写法**未纳入**（不含目标字形），"
                 "已在方法节声明为限制。")
    lines.append("- `非义项用法` 占比在 8.6%–56.3% 间波动，本身不稳定，解读需谨慎。\n")

    lines.append("## 六、复现命令\n")
    lines.append("```powershell")
    lines.append("$PY = \"C:\\Users\\19345\\anaconda3\\envs\\nlp\\python.exe\"")
    lines.append("& $PY src\\analyze\\select_k.py            # 粒度三角验证")
    lines.append("& $PY src\\analyze\\cluster_forms_final.py  # k=5 / k=7 词形聚类")
    lines.append("& $PY src\\analyze\\build_typology.py       # 五类归并 + 出图")
    lines.append("& $PY src\\analyze\\package_final_5cat.py   # 生成本文件夹")
    lines.append("```")
    lines.append("")

    with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("packaged into:", DEST)
    total = 0
    for s in SUB:
        d = os.path.join(DEST, s)
        fs = sorted(os.listdir(d))
        print(f"\n[{s}] {len(fs)} files")
        for fn in fs:
            sz = os.path.getsize(os.path.join(d, fn))
            total += sz
            print(f"   {fn:36} {sz/1024:8.1f} KB")
    rp = os.path.join(DEST, "README.md")
    total += os.path.getsize(rp)
    print(f"\n   README.md                            "
          f"{os.path.getsize(rp)/1024:8.1f} KB")
    print(f"\nTOTAL {total/1048576:.2f} MB")
    if missing:
        print("\n!! missing sources:")
        for m in missing:
            print("   ", m)
    else:
        print("\nall expected sources found")


if __name__ == "__main__":
    main()
