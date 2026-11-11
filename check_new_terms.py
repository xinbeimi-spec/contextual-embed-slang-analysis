# -*- coding: utf-8 -*-
"""Check which of the newly discovered videos actually carry the 2024-2026
meme compounds, and how their titles read."""
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

videos_path = os.path.join(C.BILI_DIR, "videos.jsonl")
rows = [json.loads(l) for l in open(videos_path, encoding="utf-8") if l.strip()]
new = [r for r in rows if r.get("year") in (2024, 2025, 2026)]
print("2024-2026 videos:", len(new))

forms = ["绷绷炸弹", "绷绷蛋", "绷炸弹", "绷铁", "绷坏", "破绷", "绷一笑二",
         "绷不绷", "绷神", "绷住文明", "绷绷强", "绷绷羊", "绷绷鼠", "绷炸蛋"]
print("\n=== title matches per form ===")
for f in forms:
    hit = [r for r in new if f in (r.get("title") or "")]
    if hit:
        print(f"  {f:8} {len(hit):4d}  e.g. {hit[0]['title'][:44]}")

print("\n=== which query surfaced them ===")
print(dict(sorted(Counter(r.get("source_query") for r in new).items(),
                  key=lambda kv: -kv[1])[:10]))

print("\n=== sample titles carrying the new forms ===")
pat = re.compile("|".join(re.escape(f) for f in forms))
hits = [r for r in new if pat.search(r.get("title") or "")]
hits.sort(key=lambda x: -(x.get("video_review") or 0))
for r in hits[:20]:
    print(f"  {r['year']} rev={r.get('video_review'):<6} | {r['title'][:56]}")
print(f"\ntotal videos whose TITLE carries a new form: {len(hits)}")
