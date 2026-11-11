# -*- coding: utf-8 -*-
"""Is search pagination degraded per-identity? Compare a fresh visitor session
(new buvid3/buvid4) against the current one."""
import os
import sys
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import Bili

sys.stdout.reconfigure(encoding="utf-8")


def depth_probe(tag, b, pages=4, keyword="绷"):
    print(f"\n=== {tag} ===")
    print("  cookies:", sorted(b.s.cookies.get_dict().keys()))
    seen = set()
    total = 0
    for order in ("pubdate", "click", "totalrank"):
        got_pages = 0
        for page in range(1, pages + 1):
            j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
                      {"search_type": "video", "keyword": keyword, "page": page,
                       "page_size": 30, "order": order})
            if not j or j.get("code") != 0:
                print(f"    {order} p{page}: code={j.get('code') if j else None}")
                break
            res = (j.get("data") or {}).get("result") or []
            if not res:
                print(f"    {order} p{page}: EMPTY -> stop")
                break
            seen |= {r.get("bvid") for r in res}
            total += len(res)
            got_pages += 1
            b.sleep()
        print(f"  {order:10} pages={got_pages} unique_so_far={len(seen)}")
    print(f"  {tag}: total_results={total} unique={len(seen)}")


print("NOW =", datetime.datetime.now().isoformat(timespec="seconds"))
b1 = Bili()
depth_probe("existing session (reused cookie chain)", b1)

print("\n--- waiting 20s, then fresh session ---")
time.sleep(20)
b2 = Bili()                      # brand-new buvid3/buvid4 from finger/spi
depth_probe("fresh session (new buvid)", b2)
