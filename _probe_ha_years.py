# -*- coding: utf-8 -*-
"""One-off diagnostic (2026-11-09): why did year-filtered discovery return
almost nothing for 2017-2020 when 2016 gave 856 videos?

Prints the RAW response shape (code, numResults, numPages, len(result)) for a
few keyword x year combinations, and checks whether page 2 is really empty.
Does not write any data.
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import Bili

sys.stdout.reconfigure(encoding="utf-8")


def probe(b, kw, y, order="pubdate", pages=(1, 2)):
    beg = int(datetime.datetime(y, 1, 1).timestamp())
    end = int(datetime.datetime(y, 12, 31, 23, 59, 59).timestamp())
    out = []
    for page in pages:
        j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
                  {"search_type": "video", "keyword": kw, "page": page,
                   "page_size": 30, "order": order,
                   "pubtime_begin_s": beg, "pubtime_end_s": end})
        if not j:
            out.append(f"{kw}/{y}/{order}/p{page}: NO RESPONSE")
            b.sleep()
            continue
        d = j.get("data") or {}
        res = d.get("result") or []
        first = (res[0].get("title") if res else "")
        first = str(first)[:40]
        out.append(f"{kw}/{y}/{order}/p{page}: code={j.get('code')} "
                   f"numResults={d.get('numResults')} numPages={d.get('numPages')} "
                   f"len={len(res)} first={first}")
        b.sleep()
    return out


b = Bili()
for kw, y in [("哈", 2019), ("哈", 2017), ("哈欠", 2019), ("哈哈哈", 2019)]:
    for line in probe(b, kw, y):
        print(line)
    print("-" * 60)

print("\n=== no time filter, same keyword, page 1-3 ===")
for page in (1, 2, 3):
    j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
              {"search_type": "video", "keyword": "哈", "page": page,
               "page_size": 30, "order": "pubdate"})
    d = (j or {}).get("data") or {}
    res = d.get("result") or []
    print(f"no-filter p{page}: code={(j or {}).get('code')} "
          f"numResults={d.get('numResults')} numPages={d.get('numPages')} "
          f"len={len(res)}")
    b.sleep()
