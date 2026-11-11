# -*- coding: utf-8 -*-
"""Diagnose whether the scarcity of traditional-sense data is a LANGUAGE fact
or a SAMPLING artifact.

Three separate questions:
  1. What is the sense composition, by year?
  2. How much did the keyword set tilt the corpus? (each item's video carries
     the query that surfaced it)
  3. Inside A1_bandage, how much is the dictionary 'bandage' sense vs the
     video-game item sense (PUBG/王者/吃鸡 回血道具)? Those are different.
"""
import os
import re
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
rows = [json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip()]
print("cleaned rows:", len(rows))

TRAD_HINTS = {"A1_bandage", "A1_tauten", "A2_straight_face", "A3_intensifier"}
NEW_HINTS = {"B_bengzhu", "NEW_parody"}

print("\n=== 1. sense composition overall ===")
cnt = collections.Counter(r["sense_hint"] for r in rows)
for k, v in cnt.most_common():
    print(f"  {k:18} {v:5d}  {100*v/len(rows):5.1f}%")
n_trad = sum(cnt[h] for h in TRAD_HINTS)
n_new = sum(cnt[h] for h in NEW_HINTS)
n_unk = cnt["unknown"]
print(f"\n  traditional (A1+A2+A3) = {n_trad} ({100*n_trad/len(rows):.1f}%)")
print(f"  new (B+parody)         = {n_new} ({100*n_new/len(rows):.1f}%)")
print(f"  unknown                = {n_unk} ({100*n_unk/len(rows):.1f}%)")

print("\n=== 2. sense x year (analysis-ready only) ===")
ready = [r for r in rows if r.get("is_canonical") == 1
         and r.get("context_sufficiency") == 1]
tab = collections.defaultdict(collections.Counter)
for r in ready:
    tab[r["ctime_iso"][:4]][r["sense_hint"]] += 1
years = sorted(tab)
print(f"{'year':6}{'total':>7}" + "".join(f"{h[:9]:>10}" for h in
                                          ["A1_bandage", "A1_tauten",
                                           "A2", "B_bengzhu", "NEW", "unknown"]))
for y in years:
    c = tab[y]
    tot = sum(c.values())
    print(f"{y:6}{tot:>7}"
          f"{c['A1_bandage']:>10}{c['A1_tauten']:>10}"
          f"{c['A2_straight_face']:>10}{c['B_bengzhu']:>10}"
          f"{c['NEW_parody']:>10}{c['unknown']:>10}")

print("\n=== 3. which search query surfaced the item's video ===")
q = collections.Counter(r.get("source_query") for r in rows)
TRAD_KW = set(C.SEARCH_KEYWORDS_TRAD)
meme_kw = [k for k in q if k not in TRAD_KW]
n_from_trad_kw = sum(v for k, v in q.items() if k in TRAD_KW)
print(f"  items whose video was surfaced by a TRADITIONAL keyword: "
      f"{n_from_trad_kw} ({100*n_from_trad_kw/len(rows):.1f}%)")
print(f"  items from meme keywords: {len(rows)-n_from_trad_kw} "
      f"({100*(len(rows)-n_from_trad_kw)/len(rows):.1f}%)")
print("  top queries:")
for k, v in q.most_common(12):
    tag = "  <-- traditional" if k in TRAD_KW else ""
    print(f"    {str(k):14} {v:5d}{tag}")

print("\n=== 4. A1_bandage contamination: medical vs video-game item ===")
GAME = re.compile(r"游戏|吃鸡|血|回血|道具|装备|药|debuff|buff|CS|PUBG|王者|"
                  r"永劫|APEX|使命|捡|箱子|队友|复活|治疗量|奶")
band = [r for r in rows if r["sense_hint"] == "A1_bandage"]
g = [r for r in band if GAME.search(r["text_norm"])]
print(f"  A1_bandage total          : {len(band)}")
print(f"  with game vocabulary      : {len(g)} ({100*len(g)/max(1,len(band)):.1f}%)")
print(f"  without game vocabulary   : {len(band)-len(g)}")
print("\n  game examples:")
for r in g[:5]:
    print("   ", r["text_norm"][:70])
print("\n  non-game examples:")
for r in [r for r in band if r not in g][:5]:
    print("   ", r["text_norm"][:70])

print("\n=== 5. where could traditional data come from? pool check ===")
vids = [json.loads(l) for l in
        open(os.path.join(C.RAW, "bilibili", "videos.jsonl"), encoding="utf-8")
        if l.strip()]
uniq = {v["bvid"]: v for v in vids}
for pat in ("绷带", "绷紧", "紧绷", "绷直", "绷着脸"):
    n = sum(1 for v in uniq.values() if pat in (v.get("title") or ""))
    print(f"  videos whose TITLE contains {pat:6}: {n}")
