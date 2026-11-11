# -*- coding: utf-8 -*-
"""Top-5 frequency of the TARGET-WORD FORMS in every semantic category.

=====================================================================
新手导读
=====================================================================
这个文件只干一件事：**数「绷」的各个词形在每个语义类别里各出现多少次**，
取每类前 5 名，画成六宫格条形图。

为什么不能直接分词数？三个坑（都踩过）：
  1) jieba 会把「绷不住」切成 绷 + 不住。不先处理，词频榜首就会是「不住」这种残片。
     做法：分词前先把目标词形整串删掉（strip_target）。
  2) 「难绷」「紧绷」的中心字在**第二个字**。只从「绷」往后匹配会整体漏掉这两个词形。
     做法：词形表匹配允许中心字在前或在后（CANON 里的正则可以带前面的字）。
  3) 同一个词形会有很多写法（绷不住了 / 绷不住的）。不合并会被拆成好几行。
     做法：正则里用 (?:了|的|啦) 做可选后缀，归到同一个词形。

计的是「出现次数」（一条句子里出现 3 次就算 3 次），同时给 per_1000
（每千条该类句子出现多少次），因为各类大小差很多，必须归一化才能比较。

流程
  beng_cdpc.jsonl（上一步聚类结果，含 cdpc_category）
    -> 按类别分组 -> count_forms() 数词形 -> 写 CSV -> 画图

This replaces the earlier "context word" version of the figure (the user's
correction, 2026-10-21): what is wanted is how often each 绷-FORM occurs inside
each semantic cluster — 绷住, 紧绷, 绷带, 绷一下, 绷紧, 绷不住, 难绷 … — not which
ordinary words surround them.

Design
  * A canonical form list (CANON below) is matched around every 绷. Patterns
    may include the character BEFORE 绷, because 难绷 / 紧绷 put the target
    second; matching only forward from 绷 silently loses those two forms — that
    was the bug in the first attempt.
  * Inflections collapse into one form: 绷不住了 / 绷不住的 → 绷不住;
    绷住了 / 绷住的 → 绷住. This keeps the table about LEXICAL FORMS rather than
    about every string variant.
  * Any 绷 occurrence not covered by a canonical pattern is counted as 其他词形,
    so the column sums back to "how many 绷 occurrences this category has" and
    nothing is silently dropped.
  * Counted per item occurrence; per_1000 given so categories of very different
    size can be compared.

Inputs   data/processed/beng_cdpc.jsonl
Outputs  results/result4/wordfreq_top5.csv            ← 论文用表（仅规范词形）
         results/result4/wordfreq_all_forms.csv       ← 全部词形 + 其他词形
         results/result4/wordfreq_forms_by_cluster.csv ← 按原始 CDPC 簇
         results/result4/wordfreq_fig_top5.png/.pdf    ← 六面板条形图（覆盖旧图）
"""
import os
import re
import sys
import json
import csv
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join(C.PROCESSED, "beng_cdpc.jsonl")
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
CAT_COLOR = {"物理义": "#4C72B0", "忍住义": "#DD8452",
             "忍不住义": "#C44E52", "评价义": "#55A868",
             "紧绷义": "#8172B3", "非义项用法": "#9E9E9E"}
TOPN = 5
OTHER = "其他词形"

# canonical forms, longest / most specific first
# 规范词形表：每项是 (显示名, 正则)。顺序 = 优先顺序，越具体的越靠前。
# 例如句子里出现「绷不住了」，会先被「绷不住(?:了)?」匹配走，不会被后面的「绷住」抢走。
# 注意「紧绷」「难绷」的正则本身包含中心字前面的那个字，这样中心字在第二位也能命中。
CANON = [
    ("紧绷", r"紧绷"),
    ("难绷", r"难绷"),
    ("绷不住", r"绷不住(?:了|的|啦|啊|咯)?"),
    ("绷得住", r"绷得住(?:了|的)?"),
    ("绷住", r"绷住(?:了|的|着)?"),
    ("绷紧", r"绷紧"),
    ("绷带", r"绷带"),
    ("绷直", r"绷直"),
    ("绷一下", r"绷一下"),
    ("绷一绷", r"绷一绷"),
    ("绷起来", r"绷起来"),
    ("绷着脸", r"绷着(?:个?脸|表情)"),
    ("绷着", r"绷着"),
    ("绷炸弹", r"绷(?:绷)?炸弹|绷绷蛋"),
    ("绷坏", r"绷坏(?:三|3|4)?"),
    ("绷铁", r"绷铁"),
    ("绷模式", r"绷模式"),
    ("绷先生", r"绷先生"),
    ("绷似串", r"绷似串"),
    ("绷了", r"绷了"),
]
RE_CANON = [(lab, re.compile(p)) for lab, p in CANON]


def count_forms(text):
    """数一句文本里的所有目标词形 -> Counter（词形 -> 次数）。

    做法是「覆盖式计数」：
      * covered 是一个和文本等长的布尔数组，记录每个字符有没有被某个词形占用；
      * 按 CANON 的顺序（具体->笼统）逐个正则找；
      * 命中时若该区间已被占用就跳过（避免「绷不住了」又被「绷了」重复计一次）；
      * 否则整段标成已占用，并把该词形 +1。
    最后：还有没被任何词形覆盖的「绷」字符，统一记到「其他词形」，
    保证「每个词形次数之和 = 该句里目标字的出现次数」，不会有句子被悄悄丢掉。
    """
    t = text or ""
    cnt = collections.Counter()
    covered = [False] * len(t)          # 每个字符是否已被某个词形吃掉

    for lab, rx in RE_CANON:            # 按优先级从具体到笼统
        for m in rx.finditer(t):        # 找出这段文本里该正则的所有位置
            # 如果这段区间里已经有字符被更具体的词形占用，就跳过
            if any(covered[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                covered[i] = True       # 标记占用
            cnt[lab] += 1

    # 剩下的「绷」字符 = 其他词形（例如「绷技了」「绷家」这类偶发的两字组合）
    other = sum(1 for i, ch in enumerate(t)
                if ch == C.TARGET_CHAR and not covered[i])
    if other:
        cnt[OTHER] += other
    return cnt


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_table(path, groups, key_name):
    """groups: (group_id, n_items, Counter) list."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([key_name, "n_items", "rank", "form", "count", "per_1000"])
        for gid, n, cnt in groups:
            for rank, (form, k) in enumerate(
                    [(a, b) for a, b in cnt.most_common() if a != OTHER][:TOPN], 1):
                w.writerow([gid, n, rank, form, k, round(1000 * k / max(n, 1), 1)])


def main():
    rows = [r for r in rj(SRC) if not r.get("excluded_from_typology")]
    print(f"items: {len(rows)}")
    os.makedirs(C.TABLES, exist_ok=True)

    # ---- per category
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["cdpc_category"]].append(r.get("text_norm") or "")
    cat_groups, cat_counts = [], {}
    print(f"\n{'category':10}{'n':>6}  {'items_with_绷':>12}  top-5 forms")
    for c in CAT_ORDER:
        texts = by_cat.get(c) or []
        cnt = collections.Counter()
        for t in texts:
            cnt.update(count_forms(t))
        cat_counts[c] = cnt
        cat_groups.append((c, len(texts), cnt))
        top = [(a, b) for a, b in cnt.most_common() if a != OTHER][:TOPN]
        print(f"{c:10}{len(texts):>6}{sum(cnt.values()):>12}  "
              + "，".join(f"{a}×{b}" for a, b in top))

    # ---- per raw CDPC cluster (supplementary)
    by_cl = collections.defaultdict(list)
    maj = collections.defaultdict(collections.Counter)
    for r in rows:
        by_cl[r["cdpc_cluster"]].append(r.get("text_norm") or "")
        maj[r["cdpc_cluster"]][r["cdpc_category"]] += 1
    cl_groups = []
    for c in sorted(by_cl):
        cnt = collections.Counter()
        for t in by_cl[c]:
            cnt.update(count_forms(t))
        cl_groups.append((f"C{c}({maj[c].most_common(1)[0][0]})", len(by_cl[c]), cnt))

    p1 = os.path.join(C.TABLES, "wordfreq_top5.csv")
    write_table(p1, cat_groups, "category")
    p2 = os.path.join(C.TABLES, "wordfreq_all_forms.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "n_items", "form", "count", "per_1000"])
        for c, n, cnt in cat_groups:
            for form, k in cnt.most_common():
                w.writerow([c, n, form, k, round(1000 * k / max(n, 1), 1)])
    p3 = os.path.join(C.TABLES, "wordfreq_forms_by_cluster.csv")
    write_table(p3, cl_groups, "cluster")

    # ---- figure: 6 panels, target-word forms only
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.8))
    for ax, c in zip(axes.ravel(), CAT_ORDER):
        n = len(by_cat.get(c) or [])
        cnt = cat_counts[c]
        top = [(a, b) for a, b in cnt.most_common() if a != OTHER][:TOPN]
        labels_ = [a for a, _ in top][::-1]
        vals = [b for _, b in top][::-1]
        ax.barh(range(len(labels_)), vals, color=CAT_COLOR[c], alpha=0.9)
        ax.set_yticks(range(len(labels_)))
        ax.set_yticklabels(labels_, fontsize=11)
        ax.set_title(f"{c}（n={n}）", fontsize=11)
        ax.set_xlabel("出现次数", fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.22)
    fig.suptitle("各语义类别里「绷」词形的 Top-5 出现次数", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"wordfreq_fig_top5.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    for p in (p1, p2, p3):
        print("->", p)


if __name__ == "__main__":
    main()
