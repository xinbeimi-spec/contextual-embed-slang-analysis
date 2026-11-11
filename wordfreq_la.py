# -*- coding: utf-8 -*-
"""Per-sense high-frequency vocabulary for 「拉」.

Two tables per category:
  1. 拉-forms  : the surface forms of 拉 itself (拉面 / 拉黑 / 拉胯 …), by count.
  2. collocates: content words that co-occur with 拉 in that category, ranked by
     a Monroe-style log-odds against the REST of the corpus (with a Dirichlet
     prior), and filtered so that generic comment boilerplate cannot win.

Outputs (into results/<slug>/<RESULT_SET>/):
  wordfreq_forms_by_category.csv
  wordfreq_collocates_by_category.csv
  wordfreq_top_by_category.txt      (human-readable)
  wordfreq_fig_top8.png/.pdf        (small multiples, top-8 collocates × 11 cats)

Run:  python src/analyze/wordfreq_la.py
"""
import os
import sys
import csv
import json
import math
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import cdpc_pipeline as P

sys.stdout.reconfigure(encoding="utf-8")

import jieba
jieba.setLogLevel(60)

ROWS = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_{C.RESULT_SET}.jsonl")
if not os.path.exists(ROWS):          # the pipeline always refreshes la_cdpc.jsonl
    ROWS = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc.jsonl")
OUT = C.TABLES

# 拉-forms, longest first so 拉面说 is not swallowed by 拉面
FORMS = ["拉面说", "拉布拉多", "拉面", "拉黑", "拉胯", "拉跨", "拉垮", "拉踩",
         "拉满", "拉爆", "拉票", "拉群", "拉扯", "拉链", "拉杆", "拉二胡",
         "拉小提琴", "拉肚子", "拉稀", "拉屎", "拉家常", "拉呱", "拉锯",
         "拉倒", "拉下水", "拉后腿", "拉练", "拉歌", "拉丁", "沙拉", "乌拉",
         "哥斯拉", "拉完了", "拉了", "拉开", "拉住", "拉走", "拉回", "拉起",
         "拉紧", "拉绳", "拉手", "拉车", "拉门", "拉长", "拉伸", "拉进",
         "拉出", "拉过来", "拉出来", "拉起来"]

STOP = set("""
的 了 是 我 你 他 她 它 我们 你们 他们 自己 这 那 这个 那个 这些 那些 这样 那样 这么 那么
在 有 和 与 就 都 也 很 还 又 再 才 只 把 被 给 让 对 从 到 跟 向 用 于 为 因为 所以 但是 而且 如果
不 没 没有 别 不要 不是 就是 真的 真是 什么 怎么 为什么 哪个 哪里 一个 一下 一起 一直 一些 一样
可以 可能 应该 需要 觉得 感觉 知道 喜欢 讨厌 觉得 认为 看到 看到 听过 说 讲 做 干 搞
啊 吧 呢 吗 呀 哦 噢 哈 哈哈 哈哈哈 嘿嘿 嗯 唉 诶 卧槽 我靠 笑死 绝了 好家伙
视频 up up主 弹幕 评论 三连 点赞 关注 up猪 主播 直播 楼主 前排 沙发
年 月 日 号 一 二 三 四 五 六 七 八 九 十 个 次 点 种 条 块 张 只 位 名
这期 上期 本期 下一期 前面 后面 现在 时候 已经 还是 就是 这种 那种 的话
""".split())

KEEP_ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def ok_word(w):
    if not w or w in STOP:
        return False
    if len(w) < 2:
        return False
    if any(c in KEEP_ASCII for c in w):
        return False
    if all(not ("\u4e00" <= c <= "\u9fff") for c in w):
        return False
    if "拉" in w:          # 拉-forms are counted in the other table
        return False
    return True


def main():
    rows = [json.loads(l) for l in open(ROWS, encoding="utf-8") if l.strip()]
    print(f"records: {len(rows)}")

    P.GRAIN = "fine"          # CAT_ORDER_FINE labels, matching result9
    cat = [P.item_category(r["text_norm"], r["clean_id"], set()) for r in rows]
    cats = [c for c in P.CAT_ORDER_FINE if c in set(cat)]
    N_all = len(rows)

    # ---- 1. 拉-forms
    form_rows = []
    for c in cats:
        idx = [i for i, x in enumerate(cat) if x == c]
        cnt = collections.Counter()
        for i in idx:
            t = rows[i]["text_norm"]
            for f in FORMS:
                n = t.count(f)
                if n:
                    cnt[f] += n
        for f, n in cnt.most_common(5):
            form_rows.append([c, len(idx), f, n, round(1000 * n / len(idx), 2)])
    with open(os.path.join(OUT, "wordfreq_forms_by_category.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_items", "form", "count", "per_1000_items"])
        w.writerows(form_rows)

    # ---- 2. collocates: tokenise once
    toks = []
    for r in rows:
        toks.append([x for x in jieba.lcut(r["text_norm"]) if ok_word(x)])

    total = collections.Counter()
    for t in toks:
        total.update(t)
    doc_all = N_all
    n_all = sum(total.values())

    coll_rows = []
    top_by_cat = {}
    for c in cats:
        idx = [i for i, x in enumerate(cat) if x == c]
        cnt = collections.Counter()
        vids = collections.defaultdict(set)      # word -> distinct videos
        for i in idx:
            cnt.update(toks[i])
            for wd in set(toks[i]):
                vids[wd].add(rows[i].get("thread_bvid"))
        n_c = sum(cnt.values())
        n_r = n_all - n_c
        scored = []
        for wd, k in cnt.items():
            if k < 8:
                continue
            # a word repeated 400x inside ONE thread is a catchphrase, not a
            # collocate: require it to appear in >=3 distinct videos.
            if len(vids[wd]) < 3:
                continue
            # Monroe et al. log-odds with informative Dirichlet prior (a=0.5)
            a = 0.5
            lo = (math.log((k + a) / (n_c - k + a))
                  - math.log((total[wd] - k + a) / (n_r - (total[wd] - k) + a)))
            var = 1.0 / (k + a) + 1.0 / (total[wd] - k + a)
            z = lo / math.sqrt(var)
            scored.append((wd, k, z))
        scored.sort(key=lambda x: -x[2])
        top_by_cat[c] = scored[:8]
        for rank, (wd, k, z) in enumerate(scored[:15], 1):
            coll_rows.append([c, len(idx), rank, wd, k,
                              round(1000 * k / max(n_c, 1), 2), round(z, 1)])
    with open(os.path.join(OUT, "wordfreq_collocates_by_category.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_items", "rank", "word", "count",
                    "per_1000_tokens", "logodds_z"])
        w.writerows(coll_rows)

    # ---- 3. readable dump
    L = ["「拉」各义项高频词（N=%d）" % N_all, ""]
    for c in cats:
        n_items = sum(1 for x in cat if x == c)
        L.append("== %s  (n=%d, %.1f%%) ==" % (c, n_items, 100 * n_items / N_all))
        forms = [r for r in form_rows if r[0] == c]
        L.append("  拉词形: " + " / ".join(f"{r[2]}({r[3]})" for r in forms) if forms
                 else "  拉词形: —")
        L.append("  上下文: " + " / ".join(f"{wd}({k})" for wd, k, _ in top_by_cat[c]))
        L.append("")
    open(os.path.join(OUT, "wordfreq_top_by_category.txt"), "w",
         encoding="utf-8").write("\n".join(L))

    # ---- 4. figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ncol, nrow = 4, int(math.ceil(len(cats) / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.2 * nrow))
    axes = axes.ravel()
    for ax, c in zip(axes, cats):
        items = list(reversed(top_by_cat[c]))
        if not items:
            ax.axis("off")
            continue
        ws = [w for w, _, _ in items]
        ks = [k for _, k, _ in items]
        col = P.CAT_COLOR.get(c, "#888888")
        ax.barh(range(len(ws)), ks, color=col, alpha=0.85)
        ax.set_yticks(range(len(ws)))
        ax.set_yticklabels(ws, fontsize=8)
        ax.set_title(f"{c} (n={sum(1 for x in cat if x == c)})", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[len(cats):]:
        ax.axis("off")
    fig.suptitle(f"「{C.TARGET_WORD}」各义项高频上下文词（log-odds 排序，前 8）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        q = os.path.join(OUT, f"wordfreq_fig_top8.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    print("\n".join(L))
    print("->", os.path.join(OUT, "wordfreq_forms_by_category.csv"))
    print("->", os.path.join(OUT, "wordfreq_collocates_by_category.csv"))


if __name__ == "__main__":
    main()
