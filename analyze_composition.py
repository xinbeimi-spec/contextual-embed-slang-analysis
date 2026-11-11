# -*- coding: utf-8 -*-
"""Quick composition check of the harvested hits, by surface cue.

This is NOT annotation -- it is only a sanity check on whether the corpus
actually contains the traditional senses the study needs.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

D = C.BILI_DIR


def rj(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


rows = rj(os.path.join(D, "hits_comment.jsonl")) + rj(os.path.join(D, "hits_danmaku.jsonl"))
print("total hits:", len(rows))

# surface cues, ordered so that the more specific pattern wins
CUES = [
    ("A1 绷带", ["绷带"]),
    ("A1 绷紧/紧绷", ["绷紧", "紧绷"]),
    ("A2 绷着脸/绷着", ["绷着脸", "绷着面孔", "绷着不笑"]),
    ("A3 绷脆/绷硬", ["绷脆", "绷硬", "绷亮"]),
    ("B 绷不住", ["绷不住", "绷不住了"]),
    ("B 绷住", ["绷住"]),
    ("B 蚌埠住了", ["蚌埠住"]),
    ("B 难绷", ["难绷"]),
    ("其他 裸绷/未知", []),
]


def classify(t):
    for label, pats in CUES:
        for p in pats:
            if p in t:
                return label
    return "其他 裸绷/未知"


cnt = Counter()
by_year = defaultdict(Counter)
for r in rows:
    t = r.get("text_clean") or ""
    lab = classify(t)
    cnt[lab] += 1
    y = r.get("ctime_iso", "")[:4]
    by_year[y][lab] += 1

print("\n=== composition by surface cue ===")
for label, _ in CUES:
    if cnt[label]:
        print(f"  {label:16} {cnt[label]:5d}  ({100*cnt[label]/len(rows):.1f}%)")

print("\n=== cue x year (top rows) ===")
years = sorted(by_year)
head = f"{'cue':16}" + "".join(f"{y[2:]:>7}" for y in years)
print(head)
for label, _ in CUES:
    line = f"{label:16}"
    for y in years:
        line += f"{by_year[y][label]:>7}"
    print(line)

print("\n=== length distribution of hits ===")
L = sorted(len(r.get("text_clean") or "") for r in rows)
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"  p{int(q*100):>2} = {L[int(q*(len(L)-1))]} 字")
print(f"  max = {L[-1]} 字")
print(f"  <=10 字 (很难标注) = {sum(1 for x in L if x <= 10)} "
      f"({100*sum(1 for x in L if x <= 10)/len(L):.1f}%)")
print(f"  >=20 字 (上下文较充分) = {sum(1 for x in L if x >= 20)} "
      f"({100*sum(1 for x in L if x >= 20)/len(L):.1f}%)")

print("\n=== examples of each traditional cue ===")
seen = defaultdict(list)
for r in rows:
    lab = classify(r.get("text_clean") or "")
    if lab.startswith(("A1", "A2", "A3")) and len(seen[lab]) < 4:
        seen[lab].append(r)
for lab in sorted(seen):
    print(f"\n[{lab}]")
    for r in seen[lab]:
        print("   ", r.get("ctime_iso", "")[:10], "|",
              (r.get("text_clean") or "")[:80].replace("\n", " "))
