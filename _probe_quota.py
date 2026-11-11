# -*- coding: utf-8 -*-
"""Diagnostic (2026-11-09): replicate discover_by_year's exact call pattern for
one year (13 keywords x 3 orders, page 1 only) and print the result count of
every single call, to see whether the "empty page 1" is systematic (quota /
throttle) or query-specific.
"""
import os
import sys
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import Bili

sys.stdout.reconfigure(encoding="utf-8")

KW = C.SEARCH_KEYWORDS + C.SEARCH_KEYWORDS_TRAD
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else C.REQUEST_DELAY
beg = int(datetime.datetime(YEAR, 1, 1).timestamp())
end = int(datetime.datetime(YEAR, 12, 31, 23, 59, 59).timestamp())

b = Bili()
n_ok = n_empty = n_err = 0
for order in C.SEARCH_ORDERS:
    for kw in KW:
        j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
                  {"search_type": "video", "keyword": kw, "page": 1,
                   "page_size": 30, "order": order,
                   "pubtime_begin_s": beg, "pubtime_end_s": end})
        if not j or j.get("code") != 0:
            n_err += 1
            print(f"  ERR  {order:10} {kw:6} code={j.get('code') if j else None}")
        else:
            d = j.get("data") or {}
            res = d.get("result") or []
            n_ok += 1 if res else 0
            n_empty += 0 if res else 1
            print(f"  {order:10} {kw:6} len={len(res):2d} numResults={d.get('numResults')}")
        time.sleep(DELAY)
print(f"\nYEAR {YEAR}: ok={n_ok} empty={n_empty} err={n_err}")
