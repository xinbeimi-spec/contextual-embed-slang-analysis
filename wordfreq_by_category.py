# -*- coding: utf-8 -*-
"""Word-frequency profile of every semantic category (top-5 words each).

What the user asked for (2026-10-21): for every semantic cluster, summarise the
top-5 high-frequency words — e.g. under 忍住义 the words that characterise how
that sense is used, and likewise for 物理义 / 紧绷义 / 评价义 / 忍不住义 /
非义项用法.

Two decisions that must be stated in the write-up
  1. The target forms are REMOVED before counting. Not just tokens containing
     绷: 绷不住 is segmented by jieba as 绷 + 不住, so the fragment 不住 would
     otherwise top the 忍不住义 list and the table would only restate the label.
     Every known target form is therefore deleted from the text first.
  2. Only content words survive: tokens shorter than 2 characters are dropped,
     a hand-written list of Chinese function words is removed, and so is a small
     PLATFORM list (视频/弹幕/回复/三连…) because those words describe bilibili,
     not the sense of 绷. Both lists are in this file so they can be edited and
     argued about.

Rates are per 1,000 items of the category (per_1000) so categories of very
different sizes can be compared; the raw count is given too.

NOTE (2026-10-21): this script produces the CONTEXT-WORD profile
(*_context_* files). The figure the user asked for is the TARGET-WORD FORM
profile, produced by src/analyze/wordfreq_forms_by_category.py.

Inputs   data/processed/beng_cdpc.jsonl  (CDPC labels + category)
Outputs  results/result4/wordfreq_top5.csv          the table for the paper
         results/result4/wordfreq_top50_by_category.csv     longer listing
         results/result4/wordfreq_by_cluster.csv      raw CDPC cluster view
         results/result4/wordfreq_fig_top5.png/.pdf   6-panel bar chart
"""
import os
import re
import sys
import json
import csv
import collections

import jieba
import jieba.posseg as pseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")
jieba.setLogLevel(60)

SRC = os.path.join(C.PROCESSED, "beng_cdpc.jsonl")
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
CAT_COLOR = {"物理义": "#4C72B0", "忍住义": "#DD8452",
             "忍不住义": "#C44E52", "评价义": "#55A868",
             "紧绷义": "#8172B3", "非义项用法": "#9E9E9E"}
TOPN = 5

# 汉语功能词（代词/助词/连词/副词/常见填充词）。刻意保守：只删“没有内容”的词。
STOPWORDS = set("""
的 了 是 我 你 他 她 它 们 我们 你们 他们 咱们 自己 这 那 这个 那个 这些 那些
这样 那样 这种 那种 这里 那里 这么 那么 啥 什么 怎么 怎样 为什么 咋 咱
就 也 都 还 又 再 才 只 很 太 更 最 挺 特别 非常 有点 一点 一下子 一直 已经
现在 今天 明天 昨天 时候 之前 之后 以后 以前 当时 刚刚 马上 突然 然后 而且
但是 可是 所以 因为 如果 虽然 不过 只是 还是 或者 并且 而是 就是 不是 没有 没
可以 可能 应该 需要 觉得 感觉 认为 知道 看到 看着 觉得 真的 真是 其实 确实
一个 一下 一次 一样 一些 有点 有关 一边 一边儿 起来 出来 下来 上来 过来
哈 哈哈 哈哈哈 哈哈哈哈 233 2333 草 笑死 笑 笑哭 狗头 滑稽 捂脸 大哭
在 有 和 与 及 或 而 被 把 给 让 从 到 向 对 对于 关于 按 按照 通过 随着
能 会 要 想 说 做 去 来 上 下 里 中 内 外 前 后 大 小 多 少 好 不好 不错
啊 呀 吧 呢 吗 嘛 哦 喔 诶 唉 嗯 额 呃 呗 啦 咯 嘞 了啦 地 得 着 过
一 二 三 四 五 六 七 八 九 十 个 只 条 点 次 天 年 月 日 号 秒 分 小时
""".split())

RE_TOKEN_OK = re.compile(r"^[\u4e00-\u9fff]{2,}$")
RE_HAS_TARGET = re.compile(C.TARGET_CHAR)

# 只保留实词：n*=名词, v*=动词, a*=形容词, i=成语, l=习语, s=处所, f=方位,
# b=区别词, z=状态词（悠然/轻松 这类词是「绷住」梗的核心，必须保留）
KEEP_POS = set("nvailsfbz")

# 泛化修饰词：出现在任何语域里，不表征某个义项（不是虚词，但对义项画像无用）
GENERIC = set("""
不会 不能 不要 不用 一定 一直 最后 开始 直接 还有 每次 其实 然后 而且
已经 现在 这里 那里 一样 一些 有点 不少 很多 所有 应该 可能 可以 需要
""".split())

# 平台词：描述的是 B 站，不是「绷」的语义。剔除以免 top-5 被“视频/弹幕”占满。
PLATFORM = set("""
视频 up up主 up们 弹幕 投币 硬币 三连 一键三连 关注 点赞 播放 直播 评论
主页 粉丝 收藏 转发 充电 稿件 封面 简介 分区 推荐 算法 标题 回复 楼 楼主
""".split())

# 目标词形的各种写法，必须在分词前整串删除（否则 绷不住 -> 绷 + 不住）
TARGET_FORMS = ["绷不住", "绷得住", "绷不了", "绷不得", "绷住了", "绷住", "崩不住",
                "难绷", "绷带", "绷紧", "紧绷", "绷直", "绷着", "绷起来",
                "绷脆", "绷硬", "绷亮", "绷开", "绷线", "绷得", "绷不", "绷"]

RE_TARGET_ANY = re.compile("|".join(re.escape(f) for f in TARGET_FORMS))


def strip_target(text):
    """Delete every target form (and its fragments) before segmentation."""
    return RE_TARGET_ANY.sub(" ", text or "")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def words_of(text):
    """Content words only: nouns / verbs / adjectives, target forms stripped.

    A POS filter (jieba.posseg) rather than a longer stop-word list: adverbs
    (d), conjunctions (c), prepositions (p), pronouns (r) and particles (u)
    describe the sentence, not the sense of 绷. Keeping only n*/v*/a*/i/l/s
    yields an interpretable per-category vocabulary.
    """
    out = []
    for w, flag in pseg.cut(strip_target(text)):
        w = w.strip()
        if len(w) < 2:
            continue
        if RE_HAS_TARGET.search(w):        # the label itself, not context
            continue
        if w in STOPWORDS or w in PLATFORM or w in GENERIC:
            continue
        if not RE_TOKEN_OK.match(w):
            continue
        if flag[0] not in KEEP_POS:
            continue
        out.append(w)
    return out


def profile(texts):
    cnt = collections.Counter()
    for t in texts:
        cnt.update(words_of(t))
    return cnt


def main():
    rows = [r for r in rj(SRC) if not r.get("excluded_from_typology")]
    print(f"items: {len(rows)}")
    os.makedirs(C.TABLES, exist_ok=True)

    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["cdpc_category"]].append(r.get("text_norm") or "")

    # ---- top-5 / top-50 per category
    p1 = os.path.join(C.TABLES, "wordfreq_context_top5.csv")
    p2 = os.path.join(C.TABLES, "wordfreq_context_top50_by_category.csv")
    f1 = open(p1, "w", newline="", encoding="utf-8-sig")
    f2 = open(p2, "w", newline="", encoding="utf-8-sig")
    w1, w2 = csv.writer(f1), csv.writer(f2)
    w1.writerow(["category", "n_items", "rank", "word", "count", "per_1000"])
    w2.writerow(["category", "n_items", "rank", "word", "count", "per_1000"])
    print(f"\n{'category':10}{'n':>6}  top-5")
    for c in CAT_ORDER:
        texts = by_cat.get(c) or []
        n = len(texts)
        cnt = profile(texts)
        top = cnt.most_common(50)
        for rank, (word, k) in enumerate(top[:TOPN], 1):
            w1.writerow([c, n, rank, word, k, round(1000 * k / max(n, 1), 1)])
        for rank, (word, k) in enumerate(top, 1):
            w2.writerow([c, n, rank, word, k, round(1000 * k / max(n, 1), 1)])
        print(f"{c:10}{n:>6}  " + "，".join(f"{w}×{k}" for w, k in top[:TOPN]))
    f1.close()
    f2.close()

    # ---- same profile per raw CDPC cluster (supplementary)
    by_cl = collections.defaultdict(list)
    for r in rows:
        by_cl[r["cdpc_cluster"]].append(r.get("text_norm") or "")
    p3 = os.path.join(C.TABLES, "wordfreq_context_by_cluster.csv")
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n_items", "rank", "word", "count", "per_1000",
                    "majority_category"])
        maj = {}
        for r in rows:
            maj.setdefault(r["cdpc_cluster"], collections.Counter())[
                r["cdpc_category"]] += 1
        for c in sorted(by_cl):
            texts = by_cl[c]
            n = len(texts)
            cnt = profile(texts)
            cat = maj[c].most_common(1)[0][0] if c in maj else ""
            for rank, (word, k) in enumerate(cnt.most_common(TOPN), 1):
                w.writerow([c, n, rank, word, k,
                            round(1000 * k / max(n, 1), 1), cat])

    # ---- 6-panel figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6))
    for ax, c in zip(axes.ravel(), CAT_ORDER):
        texts = by_cat.get(c) or []
        n = len(texts)
        cnt = profile(texts)
        top = cnt.most_common(TOPN)
        words = [w for w, _k in top][::-1]
        vals = [k for _w, k in top][::-1]
        ax.barh(range(len(words)), vals, color=CAT_COLOR[c], alpha=0.9)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_title(f"{c}（n={n}）", fontsize=11)
        ax.set_xlabel("出现次数", fontsize=9)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.18)
    fig.suptitle("各语义类别的 Top-5 高频词（已剔除目标字词形与功能词）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"wordfreq_context_fig_top5.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    for p in (p1, p2, p3):
        print("->", p)


if __name__ == "__main__":
    main()
