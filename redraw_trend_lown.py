# -*- coding: utf-8 -*-
"""重画 result4 的「历时变化」折线图，把**小样本年份**标出来（不改任何数字）。

为什么改（2026-11-03 用户指出）
  2016 年只有 3 条可用样本，折线却显示「忍不住义 66.7%」，看着像真实趋势，
  实际上 1 条 = 33 个百分点；而且这 3 条里还有 1 条是簇级多数误标。
  图本身没有算错，错在**没有告诉读者这个点的样本量**。

改法（只改画法，不动 typology_by_year.csv 的任何数值）
  1. n < LOW_N 的年份：空心点 + 虚线连接，并在点上方标出 "n=3"；
  2. 这些年份的背景加一条浅灰竖带，视觉上提示「不可靠」；
  3. 图注写明哪些年份样本过少、不得据此判断趋势；
  4. 底部样本量柱状图里，把这些年份的柱子标成深色，方便对照。

输出
  results/beng/result4/cdpc_fig4b_line.png/.pdf        ← 覆盖（新版）
  results/beng/result4/cdpc_fig4b_line_v1_plain.png/.pdf ← 旧图留档
"""
import os
import csv
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results", "beng", "result4")
SRC = os.path.join(OUTDIR, "typology_by_year.csv")
LOW_N = 30                      # 样本量低于此值的年份视为不可作趋势判断
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
CAT_COLOR = {"物理义": "#4C72B0", "忍住义": "#DD8452",
             "忍不住义": "#C44E52", "评价义": "#55A868",
             "紧绷义": "#8172B3", "非义项用法": "#BBBBBB"}


def main():
    with open(SRC, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    years = [r["year"] for r in rows]
    ns = [int(r["n"]) for r in rows]
    shares = {c: [float(r[f"share_{c}"]) for r in rows] for c in CAT_ORDER}
    low = [y for y, n in zip(years, ns) if n < LOW_N]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 6.6),
                                 height_ratios=[3, 1], sharex=True)

    # 小样本年份的灰色竖带（两幅子图都标，便于对照）
    for y in low:
        for ax in (a1, a2):
            ax.axvspan(years.index(y) - 0.5, years.index(y) + 0.5,
                       color="#EFEFEF", zorder=0)

    xs = list(range(len(years)))
    for c in CAT_ORDER:
        vals = shares[c]
        # 可靠年份：实线实心点；小样本年份：虚线空心点
        xs_ok = [x for x, y in zip(xs, years) if y not in low]
        ys_ok = [v for v, y in zip(vals, years) if y not in low]
        a1.plot(xs_ok, ys_ok, marker="o", ms=4, color=CAT_COLOR[c], label=c)
        xs_low = [x for x, y in zip(xs, years) if y in low]
        ys_low = [v for v, y in zip(vals, years) if y in low]
        if xs_low:
            a1.plot(xs_low, ys_low, ls=":", marker="o", ms=4, mfc="white",
                    color=CAT_COLOR[c], lw=1.2)

    # 在小样本年份上标出 n
    for y in low:
        i = years.index(y)
        a1.annotate(f"n={ns[i]}", (i, 1.02), ha="center", fontsize=8,
                    color="#B22222", annotation_clip=False)
    a1.set_ylabel("占比")
    a1.set_ylim(0, 1)
    a1.set_title("「绷」各用法占比的历时变化（虚线与灰底 = 样本量 < %d，不可作趋势判断）"
                 % LOW_N, fontsize=11)
    a1.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")

    a2.bar(xs, ns, color=["#B22222" if y in low else "#999999" for y in years])
    a2.set_ylabel("样本量 n")
    a2.set_xticks(xs)
    a2.set_xticklabels(years)
    a2.set_xlabel("年份")
    # 柱状图顶部把 3 条这种极小值标出来，否则在 1751 的量纲下看不见
    for i, n in enumerate(ns):
        if n < LOW_N:
            a2.annotate(str(n), (i, n), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, color="#B22222")

    note = ("注：%s 年样本量分别为 %s 条，1 条即占 20–33 个百分点，"
            "**不得据此判断趋势**（2016 的 3 条中还有 1 条为簇级多数误标）。\n"
            "其余年份为实线；灰色竖带仅提示样本不足。"
            % ("、".join(low), "、".join(str(ns[years.index(y)]) for y in low)))
    fig.text(0.5, 0.015, note, ha="center", fontsize=7.5, color="#555555")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.20, hspace=0.12)

    # 旧图留档（只在第一次运行时备份）
    for ext in ("png", "pdf"):
        old = os.path.join(OUTDIR, f"cdpc_fig4b_line.{ext}")
        bak = os.path.join(OUTDIR, f"cdpc_fig4b_line_v1_plain.{ext}")
        if os.path.exists(old) and not os.path.exists(bak):
            shutil.copyfile(old, bak)
            print("旧图备份 ->", os.path.basename(bak))
        q = os.path.join(OUTDIR, f"cdpc_fig4b_line.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)


if __name__ == "__main__":
    main()
