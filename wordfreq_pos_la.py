# -*- coding: utf-8 -*-
"""词性视角：每个义项里「拉」的词性 + 带词性标注的高频词.

做两件事（用户 2026-11-01 要求：要词性，不是词表）：

  A. 「拉」本身在每个义项里的词性分布
     - 先把 拉X 固定搭配加进 jieba 词典并指定词性，避免 拉胯/拉黑/拉满 被切碎
     - 再对每条语料找出「拉」所在的那个 token，记录它的词性
     - 汇总：义项 × 词性（动词 v / 形容词 a / 名词 n / 其他）

  B. 每个义项的高频词，**每行带词性**
     - 仍用 log-odds（Monroe 式）排序，避免「就是/真的」这类套话占榜
     - 过滤：同一词至少出现在 3 个不同视频（防单视频刷屏）
     - 输出 word, POS, count, per1000, logodds_z

输出（results/<slug>/<RESULT_SET>/）：
  pos_distribution_by_category.csv
  pos_examples_by_category.txt
  wordfreq_pos_by_category.csv
  wordfreq_pos_top_by_category.txt
"""
import os
import sys
import csv
import json
import math
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import cdpc_pipeline as P

sys.stdout.reconfigure(encoding="utf-8")

import jieba
import jieba.posseg as pseg
jieba.setLogLevel(60)

# --- 固定搭配入典，词性按其在现代汉语中的典型功能标注 -------------------
# a = 形容词（可受「很/太」修饰，作谓语）；v = 动词；n = 名词性构词；d = 程度/副词
CUSTOM = {
    "拉胯": "a", "拉跨": "a", "拉垮": "a",          # 很拉胯 / 太拉胯
    "拉满": "v", "拉爆": "v", "拉踩": "v",          # 把 X 拉满 / 拉踩对手
    "拉黑": "v", "拉票": "v", "拉群": "v",          # 拉黑某人 / 拉票
    "拉面": "n", "拉链": "n", "拉杆": "n",          # 名词性构词
    "拉二胡": "v", "拉小提琴": "v", "拉肚子": "v", "拉稀": "v",
    "拉屎": "v", "拉家常": "v", "拉呱": "v", "拉锯": "v",
    "拉扯": "v", "拉倒": "v", "拉伸": "v", "拉练": "v", "拉琴": "v",
}
for w, t in CUSTOM.items():
    jieba.add_word(w, freq=200000, tag=t)

POS_NAME = {"v": "动词", "a": "形容词", "n": "名词", "d": "副词", "ad": "副形词",
            "ng": "名语素", "vn": "名动词", "an": "形名词", "nr": "人名",
            "ns": "地名", "nz": "专名", "x": "非语素", "un": "未知", "t": "时间词"}

# jieba 的词性对少数常用词不稳定（特效→a、好听→v、小提琴→nr、英雄→ns…）。
# 这里只校正高频词里出现的那几个，避免表格里出现明显错误。
CORRECT = {
    "特效": "n", "好吃": "a", "小提琴": "n", "英雄": "n", "有点": "d",
    "确实": "d", "直接": "d", "快递": "n", "努力": "a", "曾经": "d",
    "极致": "d", "好听": "a", "难受": "a", "二胡": "n", "全员": "n",
    "对面": "n", "视角": "n", "剧情": "n", "剧本": "n", "演技": "n",
    "效果": "n", "节目": "n", "细节": "n", "血压": "n", "队友": "n",
    "对比": "v", "事实": "n", "玩家": "n", "对方": "n", "好友": "n",
    "朋友": "n", "删除": "v", "举报": "v", "大家": "n", "偶像": "n",
    "表演": "v", "加油": "v", "公演": "n", "舞蹈": "n", "成员": "n",
    "曲子": "n", "老师": "n", "厕所": "n", "肠胃": "n", "肚子": "n",
    "便秘": "a", "严重": "a", "姨妈": "n", "裤子": "n", "乳糖": "n",
    "木下": "nr", "奥特曼": "nr", "塔露拉": "nr", "奥拉夫": "nr",
    "固拉多": "nz", "多拉": "nz", "达拉": "nz", "巴拉": "nz", "闪暖": "nz",
    "轩轩": "nr", "亚轩": "nr", "宋亚轩": "nr",
}

STOP = set("""
的 了 是 我 你 他 她 它 我们 你们 他们 自己 这 那 这个 那个 这些 那些 这样 那样 这么 那么
在 有 和 与 就 都 也 很 还 又 再 才 只 把 被 给 让 对 从 到 跟 向 用 于 为 因为 所以 但是 而且 如果
不 没 没有 别 不要 不是 就是 真的 真是 什么 怎么 为什么 哪个 哪里 一个 一下 一起 一直 一些 一样
可以 可能 应该 需要 觉得 感觉 知道 喜欢 讨厌 认为 看到 说 讲 做 干 搞
啊 吧 呢 吗 呀 哦 噢 哈 哈哈 哈哈哈 嘿嘿 嗯 唉 诶 卧槽 我靠 笑死 绝了 好家伙
视频 up 弹幕 评论 三连 点赞 关注 主播 直播 楼主 前排 沙发 回复
年 月 日 号 一 二 三 四 五 六 七 八 九 十 个 次 点 种 条 块 张 只 位 名
这期 上期 本期 下一期 前面 后面 现在 时候 已经 还是 这种 那种 的话
""".split())

ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def ok_word(w, pos):
    if not w or w in STOP or len(w) < 2:
        return False
    if any(c in ASCII for c in w):
        return False
    if w.startswith("拉"):
        return False                      # 拉词形单独统计
    if pos in ("x", "w", "m", "q", "p", "c", "u", "uj", "ul", "uz", "y", "e", "o", "r"):
        return False                      # 虚词/数词/量词/代词：不是搭配内容
    return True


def locate_target_token(text, idx):
    """Return (word, pos) of the token that COVERS character position idx.

    Taking the first 拉-token in the sentence is wrong: in
    「达拉崩吧…他一把拉住了」 the first 拉 belongs to a name. We therefore walk
    the tokens with running offsets and keep the one that covers the record's
    target_start_norm (the same position the embeddings use).
    """
    pos = 0
    for w, p in pseg.lcut(text):
        s, e = pos, pos + len(w)
        pos = e
        if s <= idx < e:
            return w, p
    return None


def main():
    rows = [json.loads(l) for l in open(
        os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc.jsonl"), encoding="utf-8")
        if l.strip()]
    print(f"records: {len(rows)}")

    P.GRAIN = "fine"
    cat = [P.item_category(r["text_norm"], r["clean_id"], set()) for r in rows]
    cats = [c for c in P.CAT_ORDER_FINE if c in set(cat)]
    N = len(rows)

    # ---------- A. 「拉」的词性分布 ----------
    la_pos = []            # 每条语料：拉所在 token 的词性
    la_word = []           # 该 token 本身
    examples = collections.defaultdict(list)
    for i, r in enumerate(rows):
        t = r["text_norm"]
        idx = r.get("target_start_norm", -1)
        hit = locate_target_token(t, idx) if 0 <= idx < len(t) else None
        if hit is None:
            la_pos.append("?"); la_word.append("")
            continue
        # jieba tags bare 拉 as v even in 「太拉了 / 这么拉」. Those are the
        # ADJECTIVAL use (the sense 拉胯 without 胯), so re-tag them.
        if hit[0] == "拉" and P.NET_EVAL_CUE.search(t):
            hit = (hit[0], "a")
        la_pos.append(hit[1]); la_word.append(hit[0])
        if len(examples[cat[i]]) < 8:
            examples[cat[i]].append((hit[0], hit[1], t[:42]))

    L = ["「拉」各义项的词性分布（N=%d）" % N, ""]
    pos_rows = []
    for c in cats:
        idx = [i for i, x in enumerate(cat) if x == c]
        cnt = collections.Counter(la_pos[i] for i in idx)
        tot = sum(cnt.values())
        L.append("== %s (n=%d) ==" % (c, len(idx)))
        L.append("  词性: " + " / ".join(
            f"{POS_NAME.get(p, p)}({p}) {v} ({100*v/tot:.0f}%)"
            for p, v in cnt.most_common()))
        ws = collections.Counter(la_word[i] for i in idx if la_word[i]).most_common(6)
        L.append("  承载形式: " + " / ".join(f"{w}({n})" for w, n in ws))
        for w, p, s in examples[c][:4]:
            L.append(f"    · [{w}/{POS_NAME.get(p,p)}] {s}")
        L.append("")
        for p, v in cnt.most_common():
            pos_rows.append([c, len(idx), p, POS_NAME.get(p, p), v, round(100 * v / tot, 1)])
    open(os.path.join(C.TABLES, "pos_examples_by_category.txt"), "w",
         encoding="utf-8").write("\n".join(L))
    with open(os.path.join(C.TABLES, "pos_distribution_by_category.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_items", "pos", "pos_zh", "count", "share_pct"])
        w.writerows(pos_rows)

    # ---------- B. 带词性的高频词 ----------
    toks = []      # [(word, pos)]
    for r in rows:
        toks.append([(x.word, x.flag) for x in pseg.lcut(r["text_norm"])
                     if ok_word(x.word, x.flag)])
    total = collections.Counter()
    for t in toks:
        total.update(w for w, _ in t)
    n_all = sum(total.values())

    out_rows, top = [], {}
    for c in cats:
        idx = [i for i, x in enumerate(cat) if x == c]
        cnt = collections.Counter()
        vids = collections.defaultdict(set)
        pos_of = collections.defaultdict(collections.Counter)
        for i in idx:
            for wd, p in toks[i]:
                cnt[wd] += 1
                vids[wd].add(rows[i].get("thread_bvid"))
                pos_of[wd][p] += 1
            # 同一句里重复的词也只算一次视频出现
        n_c = sum(cnt.values()); n_r = n_all - n_c
        scored = []
        for wd, k in cnt.items():
            if k < 8 or len(vids[wd]) < 3:
                continue
            a = 0.5
            lo = (math.log((k + a) / (n_c - k + a))
                  - math.log((total[wd] - k + a) / (n_r - (total[wd] - k) + a)))
            z = lo / math.sqrt(1.0 / (k + a) + 1.0 / (total[wd] - k + a))
            p = pos_of[wd].most_common(1)[0][0]
            p = CORRECT.get(wd, p)
            scored.append((wd, p, k, z))
        scored.sort(key=lambda x: -x[3])
        top[c] = scored[:10]
        for rk, (wd, p, k, z) in enumerate(scored[:15], 1):
            out_rows.append([c, len(idx), rk, wd, p, POS_NAME.get(p, p), k,
                             round(1000 * k / max(n_c, 1), 2), round(z, 1)])
    with open(os.path.join(C.TABLES, "wordfreq_pos_by_category.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_items", "rank", "word", "pos", "pos_zh",
                    "count", "per_1000", "logodds_z"])
        w.writerows(out_rows)

    L2 = ["「拉」各义项高频词（带词性，log-odds 排序）", ""]
    for c in cats:
        n_items = sum(1 for x in cat if x == c)
        L2.append("== %s (n=%d) ==" % (c, n_items))
        L2.append("  " + " / ".join(f"{w}[{POS_NAME.get(p,p)}]{k}" for w, p, k, _ in top[c]))
        L2.append("")
    open(os.path.join(C.TABLES, "wordfreq_pos_top_by_category.txt"), "w",
         encoding="utf-8").write("\n".join(L2))

    print("\n".join(L))
    print("\n".join(L2))


if __name__ == "__main__":
    main()
