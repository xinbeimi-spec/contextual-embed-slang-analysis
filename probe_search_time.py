# -*- coding: utf-8 -*-
"""Probe bilibili search: deep paging order=pubdate + pubtime_begin_s filter."""
import os
import sys
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import Bili

b = Bili()


def ts(t):
    return datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d")


def search(params):
    return b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type", params)


print("=== A. deep paging, order=pubdate, keyword=绷 ===")
for page in (1, 5, 10, 15, 20, 25, 30, 34):
    j = search({"search_type": "video", "keyword": "绷", "page": page,
                "page_size": 30, "order": "pubdate"})
    if not j or j.get("code") != 0:
        print(f"  p{page}: code={j.get('code') if j else None}")
        b.sleep()
        continue
    res = (j.get("data") or {}).get("result") or []
    ds = [int(x.get("pubdate") or 0) for x in res]
    if ds:
        print(f"  p{page}: n={len(res)} min={ts(min(ds))} max={ts(max(ds))} "
              f"numResults={(j.get('data') or {}).get('numResults')}")
    else:
        print(f"  p{page}: n=0")
    b.sleep()

print("\n=== B. pubtime filter test (keyword=绷, order=pubdate) ===")
for y in (2016, 2018, 2020, 2022, 2024):
    beg = int(datetime.datetime(y, 1, 1).timestamp())
    end = int(datetime.datetime(y, 12, 31, 23, 59, 59).timestamp())
    j = search({"search_type": "video", "keyword": "绷", "page": 1,
                "page_size": 30, "order": "pubdate",
                "pubtime_begin_s": beg, "pubtime_end_s": end})
    if not j or j.get("code") != 0:
        print(f"  {y}: code={j.get('code') if j else None} msg={j.get('message') if j else ''}")
        b.sleep()
        continue
    res = (j.get("data") or {}).get("result") or []
    ds = [int(x.get("pubdate") or 0) for x in res]
    print(f"  {y}: n={len(res)} numResults={(j.get('data') or {}).get('numResults')} "
          f"日期范围={ts(min(ds)) if ds else '-'}~{ts(max(ds)) if ds else '-'}")
    if res:
        for x in res[:3]:
            print(f"      {ts(x.get('pubdate'))} review={x.get('video_review')} "
                  f"| {str(x.get('title'))[:34]}")
    b.sleep()

print("\n=== C. pubtime filter + totalrank ===")
for y in (2016, 2017, 2018, 2019, 2020):
    beg = int(datetime.datetime(y, 1, 1).timestamp())
    end = int(datetime.datetime(y, 12, 31, 23, 59, 59).timestamp())
    j = search({"search_type": "video", "keyword": "绷", "page": 1,
                "page_size": 30, "order": "totalrank",
                "pubtime_begin_s": beg, "pubtime_end_s": end})
    if j and j.get("code") == 0:
        res = (j.get("data") or {}).get("result") or []
        print(f"  {y}: n={len(res)} numResults={(j.get('data') or {}).get('numResults')}")
    else:
        print(f"  {y}: code={j.get('code') if j else None}")
    b.sleep()
