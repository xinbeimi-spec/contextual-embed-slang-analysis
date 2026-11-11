# -*- coding: utf-8 -*-
"""Package the final SIX-category results into one self-contained folder.

Supersedes package_final_5cat.py. The five-category folder
results/final_5cat/ is left untouched: it is the frozen earlier deliverable and
stays valid for the 2,663-item corpus. This folder is the version for the
expanded corpus with 紧绷义 added.

Files are COPIED, never moved: the originals stay where the pipeline writes them
so every number can be regenerated. The README records the source of each file.

Output tree
  results/result2/
    README.md
    figures/   fig1_six_category_space, fig4_typology_stacked,
               fig4b_typology_line (png + pdf), plus the 图2 heatmaps when the
               space has been refreshed
    tables/    typology_sizes, typology_by_year, typology_examples,
               typology_excluded, jinbeng_examples
    evidence/  select_k_comparison, form_cluster_recovery,
               forms_k7_contingency, forms_k7_topforms,
               forms_k5_contingency, clean_report_dedup
    data/      beng_typology.jsonl, jinbeng_adjudication.jsonl,
               dropped_items.jsonl
"""
import os
import sys
import csv
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

DEST = os.path.join(C.RESULTS, "result2")
SUB = ["cat6", "evidence"]

FIGURES = [
    ("fig1_six_category_space.png", "图1  六类用法在语义空间中的分布（聚类图，主图）"),
    ("fig1_six_category_space.pdf", "图1  同上，矢量版"),
    ("fig1c_k7_clusters.png", "图1c 归并前的 k=7 聚类散点（六类归并的直接依据）"),
    ("fig1c_k7_clusters.pdf", "图1c 同上，矢量版"),
    ("fig4_typology_stacked.png", "图4  六类用法的百分比堆叠构成（主图）"),
    ("fig4_typology_stacked.pdf", "图4  同上，矢量版"),
    ("fig4b_typology_line.png", "图4b 各类占比折线 + 样本量底栏"),
    ("fig4b_typology_line.pdf", "图4b 同上，矢量版"),
    ("fig2a_selected5_v1_template4.png", "图2a 五句相似度（第4句＝模板句，早期版本，见限制）"),
    ("fig2b_selected5_v2_nontemplate4.png", "图2b 五句相似度（第4句＝非模板句，早期版本）"),
]
TABLES = [
    ("typology_sizes.csv", "六类的数量与占比（不含被排除的极零散义项）"),
    ("typology_by_year.csv", "年度 × 六类（含每类占比）"),
    ("typology_examples.csv", "每类 12 条真实例句（写论文举证用）"),
    ("typology_excluded.csv", "被排除出分母的极零散词典义项（工艺义/程度义）及原因"),
    ("jinbeng_examples.csv", "「紧绷义」全部条目 + 触发的线索 + 判定理由"),
    ("typology_representative.csv", "各类代表句（按质心余弦 + 多样性筛选）"),
    ("selected5_cosine.csv", "5 句余弦矩阵（当前空间）"),
    ("selected5_cosine_v1_v2.csv", "第4句两个候选的 v1/v2 对照矩阵"),
    ("selected5_selfcheck.csv", "5 句对各义项质心的相似度"),
]
EVIDENCE = [
    ("select_k_comparison.csv", "k=2..10 的三角验证：silhouette / 重抽样稳定度 / 词形熵"),
    ("form_cluster_recovery.csv", "k=4..7 对五种词形的复原力（纯度 / ARI / NMI / 稳定度）"),
    ("forms_k7_contingency.csv", "k=7 的簇 × 词形列联表（六类归并的直接依据）"),
    ("forms_k7_topforms.csv", "k=7 每簇高频用法（per_1000 可比）"),
    ("forms_k5_contingency.csv", "k=5 强制的列联表（说明为何五词形不可行）"),
    ("clean_report_dedup.csv", "清洗漏斗：原始记录 → 保留 → 可用（含剔除与排除计数）"),
]
DATA = [
    ("beng_typology.jsonl", "逐条记录：category / cluster_k7 / form_group / 排除标记"),
    ("jinbeng_adjudication.jsonl", "「紧绷义」逐条人工判定（keep/reject + 理由）"),
    ("dropped_items.jsonl", "被剔除的不相干条目 + 原因（可审计、可撤回）"),
]


def load_csv(name):
    p = os.path.join(C.TABLES, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_dedup():
    p = os.path.join(C.TABLES, "clean_report_dedup.csv")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["metric"]: r["value"] for r in csv.DictReader(f)}


def copy_group(src_dir, dest_sub, items, missing):
    out = []
    for name, desc in items:
        src = os.path.join(src_dir, name)
        dst = os.path.join(DEST, dest_sub, name)
        if not os.path.exists(src):
            missing.append(src)
            continue
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

    figs = copy_group(C.FIGURES, "cat6", FIGURES, missing)
    tabs = copy_group(C.TABLES, "cat6", TABLES, missing)
    evid = copy_group(C.TABLES, "evidence", EVIDENCE, missing)
    data = copy_group(C.PROCESSED, "cat6", DATA, missing)

    sizes = load_csv("typology_sizes.csv")
    by_year = load_csv("typology_by_year.csv")
    dedup = load_dedup()

    n_total = sum(int(r["n"]) for r in sizes
                  if r.get("category") not in ("TOTAL",))
    lines = []
    lines.append("# 「绷」六类用法的最终分析结果\n")
    lines.append("本文件夹是**最终交付版**（v6）。上一版五类结果保存在 "
                 "`results/final_5cat/`，对应扩量前的 2,663 条语料，仍然有效；"
                 "本版对应扩量后的语料，并新增第 6 类「紧绷义」。")
    lines.append("所有文件均为**复制**（原件仍在 `results/tables`、`results/figures`、"
                 "`data/processed`），每一步都可重新生成。\n")
    lines.append("---\n")

    lines.append("## 〇、本次修订（2026-10-21）\n")
    lines.append("| 项 | 五类版 | 六类版 |")
    lines.append("|---|---|---|")
    kept = dedup.get("kept", "—")
    ready = dedup.get("analysis_ready (canonical & judgeable)", "—")
    lines.append(f"| 语料规模 | 2,663 条抽向量 / 1,654 条可用 | "
                 f"{n_total} 条入类 / {ready} 条可用（清洗后保留 {kept}） |")
    lines.append("| 类别数 | 5 | **6**（新增「紧绷义」） |")
    lines.append("| 簇 6 残差 | 全部归入「非义项用法」 | 真实物理用法归位到「物理义」，"
                 "其余仍为「非义项用法」 |")
    lines.append("| 极零散义项 | 混在「非义项用法」里 | 单列并排除出分母（`typology_excluded.csv`） |")
    lines.append("| 不相干条目 | 混在「非义项用法」里 | 剔除并留痕（`data/dropped_items.jsonl`） |")
    lines.append("")

    lines.append("## 一、六类构成\n")
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    for r in sizes:
        try:
            share = f"{100*float(r['share']):.1f}%"
        except Exception:
            share = r.get("share", "")
        label = "**合计**" if r.get("category") == "TOTAL" else r["category"]
        lines.append(f"| {label} | {r['n']} | {share} |")
    lines.append("")
    lines.append("> 第 6 类「非义项用法」不是词义，而是**提及而非使用**"
                 "（元语言玩梗 / 逐字复读 / 异形错别字 / 同形误匹配），"
                 "单列以保证样本完整、不删数据。\n")
    lines.append("> 「紧绷义」＝ 神情紧绷 / 精神紧绷（面部与心理层面的紧张）。"
                 "它由**线索规则筛候选 + 逐条人工判定**产生，判定结果见 "
                 "`tables/jinbeng_examples.csv` 与 `data/jinbeng_adjudication.jsonl`；"
                 "类别很小，只能作为**义项存在性**的证据，不能用于频次比较。\n")

    lines.append("## 二、图表清单\n")
    for title, group in (("figures/", figs), ("tables/", tabs),
                         ("evidence/（为什么是这六类，而不是别的分法）", evid),
                         ("data/", data)):
        lines.append(f"### {title}")
        lines.append("| 文件 | 说明 |")
        lines.append("|---|---|")
        for n, d, _ in group:
            lines.append(f"| `{n}` | {d} |")
        lines.append("")

    if by_year:
        lines.append("## 三、年度构成（分析可用样本）\n")
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
                 "取目标词 span 内 token 的隐状态均值，并用 last-4 层平均；"
                 "聚类使用 [MASK] 位置表示（去掉字形线索）。")
    lines.append("2. **去中心化**：减去全语料均值并移除第一主成分（ABTT k=1）。"
                 "扩量后中心化参数在新语料上重新拟合，旧参数已归档在 "
                 "`data/processed/_archive_5cat/`，两套结果不混用。")
    lines.append("3. **聚类粒度**：以多判据三角验证确定（silhouette / 重抽样 ARI / "
                 "词形熵），见 `evidence/select_k_comparison.csv`。")
    lines.append("4. **类别归并**：簇号 → 类别的映射写在 `src/analyze/build_typology.py` "
                 "顶部，运行时打印簇 × 词形列联表，映射若与数据不符会立刻暴露。")
    lines.append("5. **紧绷义**：线索规则（`src/analyze/jinbeng_cue.py`）只用于**筛候选**；"
                 "每条候选都由人工读完判定，判定表落盘，规则与判定不一致的条数会被打印。")
    lines.append("6. **一致性检查**：按“簇号”与按“词形”两条独立规则判定，冲突条数见运行日志。\n")

    lines.append("## 五、必须随文报告的限制\n")
    lines.append("- 语料为**关键词条件抽样**（按“绷”及若干词形检索视频后抓评论），"
                 "因此占比是**样本内份额**，不是总体使用率。")
    lines.append("- 扩量主要来自 2026 年（该年每条视频的命中密度约为早年 50–100 倍），"
                 "**所以 2026 在可用样本中占比很高，历时曲线的后期高点部分是采样结构，"
                 "不能直接读成语言突变**。")
    lines.append("- 2016–2018 样本量极小（早年年视频评论区多已关闭/删除，命中率仅 1–3%），"
                 "早期数值不可作结论。")
    lines.append("- 「紧绷义」样本量为数十条量级，**只能作义项存在性与共现模式的证据**；"
                 "该类的年度占比不具备统计意义。")
    lines.append("- `蚌埠住了` 等同音异形写法**未纳入**（不含目标字形）。")
    lines.append("- 「非义项用法」占比本身不稳定，解读需谨慎；它是拒绝删数据的结果，"
                 "不是一种词义。")
    lines.append("- 已剔除的条目（专名/误匹配/广告）与已排除的极零散义项（工艺义/程度义）"
                 "都留有清单与理由，可随时撤回重算。\n")

    lines.append("## 六、复现命令\n")
    lines.append("```powershell")
    lines.append("$PY = \"C:\\Users\\19345\\anaconda3\\envs\\nlp\\python.exe\"")
    lines.append("# 0) 爬取（可断点续爬）")
    lines.append("& $PY src\\crawl\\crawl_bili.py --phase crawl --select all --years 2026-2026 --max-videos 2000")
    lines.append("# 1) 清洗（含剔除规则与极零散义项标记）")
    lines.append("& $PY src\\clean\\clean_corpus.py")
    lines.append("# 2) 向量（target 与 masked 两份）")
    lines.append("& $PY src\\embed\\extract_embeddings.py")
    lines.append("& $PY src\\embed\\extract_masked_vectors.py")
    lines.append("# 3) 粒度验证与聚类")
    lines.append("& $PY src\\analyze\\select_k.py")
    lines.append("& $PY src\\analyze\\cluster_forms_final.py")
    lines.append("# 4) 紧绷义判定 + 六类归并 + 出图")
    lines.append("& $PY src\\analyze\\adjudicate_jinbeng.py")
    lines.append("& $PY src\\analyze\\build_typology.py")
    lines.append("& $PY src\\analyze\\make_five_category_figure.py")
    lines.append("# 5) 打包")
    lines.append("& $PY src\\analyze\\package_final_6cat.py")
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
            print(f"   {fn:38} {sz/1024:8.1f} KB")
    rp = os.path.join(DEST, "README.md")
    total += os.path.getsize(rp)
    print(f"\n   README.md                              "
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
