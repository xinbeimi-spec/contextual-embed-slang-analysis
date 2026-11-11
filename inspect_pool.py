# -*- coding: utf-8 -*-
"""Inspect videos.jsonl: duplicates, source_query coverage, A2 availability."""
import json
import sys
import collections
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

P = os.path.join(C.BILI_DIR, "videos.jsonl")
rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]
uniq = {r["bvid"]: r for r in rows}
print("file rows:", len(rows), " unique bvid:", len(uniq))

c = collections.Counter(r.get("source_query") for r in rows if r["bvid"] in uniq
                        and uniq[r["bvid"]] is r)
cu = collections.Counter(v.get("source_query") for v in uniq.values())
print("distinct source_query (deduped):", len(cu))
for k, v in sorted(cu.items(), key=lambda kv: -kv[1]):
    print("   %r: %d" % (k, v))

print()
for pat in ("绷带", "绷紧", "紧绷", "绷着脸", "绷直", "难绷", "绷绷炸弹", "绷铁", "破绷"):
    n = sum(1 for v in uniq.values() if pat in (v.get("title") or ""))
    print(f"title contains {pat:8}: {n}")

print("\nyear distribution (deduped):")
print(dict(sorted(collections.Counter(v.get("year") for v in uniq.values()).items())))
