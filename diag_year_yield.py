# -*- coding: utf-8 -*-
"""Diagnose per-year discovery yield: how many DISTINCT videos does the
pubtime-filtered search actually return, and how many are already known?

Also dumps which keywords contribute, so we can see whether the filter is
being honoured by the API.
"""
import os
import sys
import json
import time
import datetime
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import Bili, read_jsonl, F_VIDEOS

sys.stdout.reconfigure(encoding="utf-8")

known = {r["bvid"]: r for r in read_jsonl(F_VIDEOS)}
print("known pool:", len(known), "by year:",
      dict(sorted(collections.Counter(v.get("year") for v in known.values()).items())))

b = Bili()

KWS = ["绷", "绷带", "绷不住"]
for y in (2019, 2021, 2023):
    beg = int(datetime.datetime(y, 1, 1).timestamp())
    end = int(datetime.datetime(y, 12, 31, 23, 59, 59).timestamp())
    allb = set()
    fresh = set()
    per_kw = {}
    years_seen = collections.Counter()
    for kw in KWS:
        got = 0
        for page in range(1, 4):
            j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
                      {"search_type": "video", "keyword": kw, "page": page,
                       "page_size": 30, "order": "totalrank",
                       "pubtime_begin_s": beg, "pubtime_end_s": end})
            if not j or j.get("code") != 0:
                print(f"    {kw} p{page}: code={j.get('code') if j else None}")
                b.sleep()
                continue
            res = (j.get("data") or {}).get("result") or []
            got += len(res)
            for it in res:
                bv = it.get("bvid")
                if not bv:
                    continue
                allb.add(bv)
                py = datetime.datetime.fromtimestamp(int(it.get("pubdate") or 0)).year
                years_seen[py] += 1
                if bv not in known:
                    fresh.add(bv)
            if not res:
                break
            b.sleep()
        per_kw[kw] = got
    print(f"\nyear {y}: requests={len(KWS)*3} results={sum(per_kw.values())} "
          f"distinct_bvids={len(allb)} new_vs_pool={len(fresh)}")
    print(f"  per keyword hits: {per_kw}")
    print(f"  actual pubdate years returned: {dict(sorted(years_seen.items()))}")
