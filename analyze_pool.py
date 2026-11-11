# -*- coding: utf-8 -*-
"""Analyse the discovered video pool: how many are 'about' the meme?"""
import sys
import json
import collections
import os

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

P = os.path.join(C.BILI_DIR, "videos.jsonl")
rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
print("total videos:", len(rows))

pats = ["绷不住", "难绷", "绷住", "蚌埠住", "没绷住", "绷带", "紧绷", "绷"]
print()
print("%-8s %7s %9s %11s" % ("pattern", "count", "med_review", "sum_review"))
for p in pats:
    sub = [r for r in rows if p in (r.get("title") or "")]
    rev = sorted(r.get("video_review") or 0 for r in sub)
    print("%-8s %7d %9d %11d" % (
        p, len(sub), rev[len(rev) // 2] if rev else 0, sum(rev)))

for label, cond in [
    ("难绷 OR 绷不住", lambda t: ("难绷" in t) or ("绷不住" in t)),
    ("难绷 only", lambda t: "难绷" in t),
    ("绷不住 only", lambda t: "绷不住" in t),
]:
    c = collections.Counter()
    for r in rows:
        if cond(r.get("title") or ""):
            c[r["year"]] += 1
    print()
    print("== by year:", label, "==")
    print(dict(sorted(c.items())))

print()
print("== top videos by comment count among meme-titled ==")
sub = [r for r in rows
       if ("难绷" in (r.get("title") or "")) or ("绷不住" in (r.get("title") or ""))]
sub.sort(key=lambda x: -(x.get("video_review") or 0))
for r in sub[:12]:
    print("  %s rev=%-6s | %s" % (r["year"], r.get("video_review"), r["title"][:46]))

print()
print("== source_query distribution ==")
print(dict(sorted(collections.Counter(r.get("source_query") for r in rows).items(),
                  key=lambda kv: -kv[1])))
