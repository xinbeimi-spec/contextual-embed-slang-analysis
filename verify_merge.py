# -*- coding: utf-8 -*-
"""Integrity + composition report for the CURRENT word's cleaned corpus.

Rewritten 2026-10-27 so that it works for every word, not only for 绷's
old-vs-new batch merge.  It answers the same questions the user asked to be
warned about, in a word-agnostic way:

  1. Does the clean table cover exactly the raw hits (minus stated drops)?
  2. Is the schema complete and type-stable?
  3. What is the year composition, and is any single year over-represented
     (the 2026-concentration warning)?
  4. Are the traditional / physical senses present, or did keyword sampling
     collapse the corpus onto the network usage?

Outputs: logs/<slug>/merge_report.txt (also printed).
Run:  python src/verify_merge.py
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
RAW_C = os.path.join(C.RAW, "bilibili", "hits_comment.jsonl")
RAW_D = os.path.join(C.RAW, "bilibili", "hits_danmaku.jsonl")
DROPPED_F = os.path.join(C.PROCESSED, "dropped_items.jsonl")
OUT = os.path.join(C.LOGS, "merge_report.txt")

REQUIRED = ["clean_id", "text_norm", "target_start_norm", "target_end_norm",
            "content_type", "ctime_iso", "is_canonical", "context_sufficiency",
            "dup_group", "platform", "sense_hint", "excluded_from_typology"]

EARLY = ("2016", "2017", "2018", "2019", "2020")


def rj(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def year_of(r):
    return (r.get("ctime_iso") or "")[:4]


def main():
    L = []
    clean = rj(CLEAN)
    raw_c, raw_d = rj(RAW_C), rj(RAW_D)
    raw = raw_c + raw_d
    dropped = rj(DROPPED_F)

    L.append(f"=== 目标词 {C.TARGET_WORD}（slug={C.WORD_SLUG}）语料完整性 ===")
    L.append("")

    # ---- 1. coverage
    L.append("=== 1. 原始命中 vs 清洗主表 ===")
    L.append(f"raw 命中：评论 {len(raw_c)} + 弹幕 {len(raw_d)} = {len(raw)}")
    L.append(f"清洗主表 {os.path.basename(CLEAN)}：{len(clean)} 行")
    L.append(f"drop_rules 剔除：{len(dropped)} 行（见 dropped_items.jsonl，可撤回）")
    raw_ids = {r.get("item_id_hash") or r.get("id") for r in raw}
    clean_ids = {r.get("item_id_hash") or r.get("id") for r in clean}
    L.append(f"清洗表覆盖的 raw id：{len(clean_ids)} / {len(raw_ids)}"
             f"（差值＝目标字清洗后消失、过短、drop_rules 剔除）")
    L.append(f"清洗表里出现但 raw 没有的 id：{len(clean_ids - raw_ids)}（应为 0）")
    L.append("")

    # ---- 2. drops by reason
    L.append("=== 2. 剔除理由分布 ===")
    if not dropped:
        L.append("  （本次没有走 drop_rules 的条目）")
    else:
        c = collections.Counter((r.get("drop_reason") or "").split(":")[0]
                                for r in dropped)
        for k, v in c.most_common():
            L.append(f"  {k:24} {v}")
    n_excl = sum(1 for r in clean if r.get("excluded_from_typology"))
    L.append(f"  极零散词典义（保留在语料、仅排除出分母）：{n_excl}")
    L.append("")

    # ---- 3. schema
    L.append("=== 3. schema 与必需字段 ===")
    keys = set()
    for r in clean:
        keys |= set(r)
    L.append(f"  字段数 {len(keys)}：{', '.join(sorted(keys))}")
    gaps = collections.Counter()
    for r in clean:
        for f in REQUIRED:
            if (f not in r or r[f] is None) and f not in ("dup_group",
                                                           "excluded_from_typology"):
                gaps[f] += 1
    if gaps:
        L.append(f"  !! 缺失/空值：{dict(gaps)}")
    else:
        L.append("  必需字段齐备（dup_group / excluded_from_typology 的 None 是正常值）")
    # type stability on the shared fields
    drift = []
    for f in sorted(keys):
        types = {type(r.get(f)).__name__ for r in clean}
        if len(types) > 1:
            drift.append(f"{f}={sorted(types)}")
    L.append(f"  单一类型字段之外（混合类型属预期）：{', '.join(drift) if drift else '无'}")
    L.append("")

    # ---- 4. year composition
    L.append("=== 4. 年份构成（发送年 = 真实时间轴）===")
    n = len(clean) or 1
    by_year = collections.Counter(year_of(r) for r in clean)
    canon = collections.Counter(year_of(r) for r in clean if r.get("is_canonical"))
    judge = collections.Counter(year_of(r) for r in clean
                                if r["is_canonical"] and r["context_sufficiency"])
    L.append(f"  {'年':>6}{'total':>8}{'占比':>8}{'canon':>8}{'可用':>8}")
    for y in sorted(k for k in by_year if k):
        L.append(f"  {y:>6}{by_year[y]:>8}{100*by_year[y]/n:>7.1f}%"
                 f"{canon[y]:>8}{judge[y]:>8}")
    L.append("")

    # ---- 5. hint distribution (provisional)
    L.append("=== 5. 表层线索分布（provisional sense_hint，非标签）===")
    hc = collections.Counter(r.get("sense_hint") for r in clean if r.get("is_canonical"))
    tot = sum(hc.values()) or 1
    for h, v in hc.most_common():
        L.append(f"  {str(h):16} {v:6d}  {100*v/tot:5.1f}%")
    L.append("")

    # ---- 6. verdicts
    L.append("=== 6. 必须反馈的判断 ===")
    d26 = 100 * by_year["2026"] / n
    early = 100 * sum(by_year[y] for y in EARLY) / n
    L.append(f"  * 2026 占比 {d26:.1f}% → "
             f"{'过于集中，必须在图注/正文声明' if d26 > 60 else '尚可'}")
    L.append(f"  * 2016–2020 合计 {early:.1f}% → "
             f"{'早年样本太薄，早年结论要谨慎' if early < 10 else '早年样本可用'}")
    n_dup = sum(1 for r in clean if r.get("dup_group"))
    L.append(f"  * 近重复组成员的条目 {n_dup}（{100*n_dup/n:.1f}%），"
             f"canonical {sum(1 for r in clean if r.get('is_canonical'))}")
    L.append(f"  * 分析可用（canonical & judgeable）："
             f"{sum(1 for r in clean if r['is_canonical'] and r['context_sufficiency'])}")
    hint_top = hc.most_common(1)[0][0] if hc else "n/a"
    L.append(f"  * 主导线索 {hint_top} 占 {100*(hc[hint_top]/tot if hc else 0):.1f}% → "
             f"关键词条件抽样的偏倚要写进方法节")

    txt = "\n".join(L)
    os.makedirs(C.LOGS, exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
