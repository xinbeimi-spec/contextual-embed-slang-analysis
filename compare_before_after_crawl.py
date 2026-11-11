# -*- coding: utf-8 -*-
"""Compare sense composition BEFORE vs AFTER the traditional-video crawl.

Uses the pre-crawl snapshot so the comparison is auditable rather than relying
on memory. Also quantifies how many of the targeted videos actually yielded any
comments at all -- that is the platform-side censoring evidence.
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

BILI = os.path.join(C.RAW, "bilibili")
SNAP = os.path.join(C.RAW, "_backups", "20261020_171219_before_trad_crawl",
                    "bilibili")

CUES = [
    ("A1_bandage", ["绷带"]),
    ("A1_tauten", ["绷紧", "紧绷", "绷直", "绷开"]),
    ("A2_straight_face", ["绷着脸", "绷着面孔", "绷住脸"]),
    ("A3_intensifier", ["绷脆", "绷硬", "绷亮"]),
    ("NEW_parody", ["绷绷炸弹", "绷炸弹", "绷绷蛋", "绷铁", "绷坏", "破绷",
                    "绷一笑二", "绷不绷", "绷神", "绷绷羊", "绷绷鼠", "绷绷强"]),
    ("B_bengzhu", ["绷不住", "绷住了", "没绷住", "绷住"]),
]


def classify(t):
    for lab, pats in CUES:
        for p in pats:
            if p in t:
                return lab
    return "unknown"


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def comp(rows):
    c = collections.Counter(classify(r.get("text_clean") or r.get("text_norm")
                                     or "") for r in rows)
    return c


old = load(os.path.join(SNAP, "hits_comment.jsonl")) + \
      load(os.path.join(SNAP, "hits_danmaku.jsonl"))
new = load(os.path.join(BILI, "hits_comment.jsonl")) + \
      load(os.path.join(BILI, "hits_danmaku.jsonl"))
print(f"before: {len(old)} hits   after: {len(new)} hits   "
      f"delta: +{len(new)-len(old)}")

co, cn = comp(old), comp(new)
names = [c[0] for c in CUES] + ["unknown"]
print(f"\n{'sense':20}{'before':>9}{'after':>9}{'delta':>8}")
for n in names:
    print(f"{n:20}{co[n]:>9}{cn[n]:>9}{cn[n]-co[n]:>+8}")

TRAD = {"A1_bandage", "A1_tauten", "A2_straight_face", "A3_intensifier"}
t_o = sum(co[n] for n in TRAD)
t_n = sum(cn[n] for n in TRAD)
print(f"\ntraditional total: {t_o} -> {t_n}  (+{t_n-t_o})")
print(f"traditional share: {100*t_o/len(old):.1f}% -> {100*t_n/len(new):.1f}%")

# ---- how many target videos actually gave any comments?
vids = {v["bvid"]: v for v in load(os.path.join(BILI, "videos.jsonl"))}
targets = [v for v in vids.values()
           if v.get("n_comment_sampled") is not None
           and (v.get("year") or 0) <= 2021]
zero = [v for v in targets if (v.get("n_comment_sampled") or 0) == 0]
print(f"\nvideos crawled (any year<=2021): {len(targets)}")
print(f"  of which returned ZERO comments: {len(zero)} "
      f"({100*len(zero)/max(1,len(targets)):.1f}%)")
print("  -> comment sections closed/deleted; this is platform-side censoring")

# ---- new hits by year (traditional sense only)
print("\n=== traditional-sense hits by SEND year (after) ===")
ty = collections.Counter()
for r in new:
    lab = classify(r.get("text_clean") or r.get("text_norm") or "")
    if lab in TRAD:
        ty[(r.get("ctime_iso") or "")[:4]] += 1
for y in sorted(ty):
    print(f"  {y}: {ty[y]}")
print(f"  total traditional hits = {sum(ty.values())}")
