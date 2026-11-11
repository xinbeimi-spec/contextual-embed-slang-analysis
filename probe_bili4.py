# -*- coding: utf-8 -*-
"""Battery test for Bilibili reply variants and danmaku."""
import sys
import time
import json
import re
import hashlib
import urllib.parse
import datetime

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

UA_PC = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIXIN_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43,
             5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16,
             24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59,
             6, 63, 57, 62, 11, 36, 20, 34, 44, 52]

S = requests.Session()
S.headers.update({"User-Agent": UA_PC, "Referer": "https://www.bilibili.com/",
                  "Origin": "https://www.bilibili.com",
                  "Accept": "application/json, text/plain, */*"})


def ts2s(t):
    try:
        return datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(t)


def boot():
    S.get("https://www.bilibili.com/", timeout=20)
    j = S.get("https://api.bilibili.com/x/frontend/finger/spi", timeout=20).json()
    d = j.get("data") or {}
    if d.get("b_3"):
        S.cookies.set("buvid3", d["b_3"], domain=".bilibili.com")
    if d.get("b_4"):
        S.cookies.set("buvid4", d["b_4"], domain=".bilibili.com")


def wbi():
    j = S.get("https://api.bilibili.com/x/web-interface/nav", timeout=20).json()
    img = j["data"]["wbi_img"]["img_url"].rsplit("/", 1)[-1].split(".")[0]
    sub = j["data"]["wbi_img"]["sub_url"].rsplit("/", 1)[-1].split(".")[0]
    o = img + sub
    return "".join(o[i] for i in MIXIN_TAB)[:32]


def sign(params, mk):
    pr = dict(params)
    pr["wts"] = int(time.time())
    clean = [(k, "".join(c for c in str(v) if c not in "!'()*"))
             for k, v in sorted(pr.items())]
    q = urllib.parse.urlencode(clean)
    pr["w_rid"] = hashlib.md5((q + mk).encode()).hexdigest()
    return pr


def try_get(tag, url, params, headers=None):
    try:
        r = S.get(url, params=params, timeout=20, headers=headers)
        code = r.status_code
        try:
            j = r.json()
        except Exception:
            print(f"[{tag}] HTTP {code} non-json len={len(r.text)} head={r.text[:100]!r}")
            return None
        c = j.get("code")
        print(f"[{tag}] HTTP {code} code={c} msg={str(j.get('message'))[:80]}")
        return j
    except Exception as e:
        print(f"[{tag}] EXC {type(e).__name__}: {e}")
        return None


def count_replies(j):
    if not j or j.get("code") != 0:
        return -1
    d = j.get("data") or {}
    reps = d.get("replies") or []
    return len(reps)


def bili_part():
    boot()
    mk = wbi()

    # find a mid-size video
    sp = sign({"search_type": "video", "keyword": "绷不住", "page": 1,
               "page_size": 30, "order": "click"}, mk)
    j = S.get("https://api.bilibili.com/x/web-interface/wbi/search/type",
              params=sp, timeout=20).json()
    res = (j.get("data") or {}).get("result") or []
    aid = None
    cid = None
    for it in res:
        rv = int(it.get("video_review") or 0)
        if 200 < rv < 5000:
            aid = it["aid"]
            print("picked", it.get("bvid"), "review=", rv,
                  "pub=", ts2s(it.get("pubdate")))
            break
    if aid is None and res:
        aid = res[0]["aid"]
        print("fallback first", res[0].get("bvid"))
    time.sleep(1.0)

    vp = sign({"aid": aid}, mk)
    vj = S.get("https://api.bilibili.com/x/web-interface/wbi/view",
               params=vp, timeout=20).json()
    if vj.get("code") == 0:
        cid = vj["data"].get("cid") or (vj["data"].get("pages") or [{}])[0].get("cid")
        print("cid =", cid, "reply=", vj["data"]["stat"]["reply"])

    print("\n---- reply variants ----")
    pag = json.dumps({"offset": ""}, separators=(",", ":"))
    variants = [
        ("wbi/main mode=2", sign({"oid": aid, "type": 1, "mode": 2, "plat": 1,
                                  "web_location": "1315875",
                                  "pagination_str": pag}, mk),
         "https://api.bilibili.com/x/v2/reply/wbi/main"),
        ("wbi/main mode=3 ps=49", sign({"oid": aid, "type": 1, "mode": 3, "plat": 1,
                                        "web_location": "1315875", "ps": 49,
                                        "pagination_str": pag}, mk),
         "https://api.bilibili.com/x/v2/reply/wbi/main"),
        ("main mode=3 no-wbi", {"oid": aid, "type": 1, "mode": 3, "next": 0,
                                "ps": 20, "plat": 1},
         "https://api.bilibili.com/x/v2/reply/main"),
        ("reply sort=2 ps=49", {"type": 1, "oid": aid, "pn": 1, "ps": 49, "sort": 2},
         "https://api.bilibili.com/x/v2/reply"),
        ("reply sort=0 ps=49", {"type": 1, "oid": aid, "pn": 1, "ps": 49, "sort": 0},
         "https://api.bilibili.com/x/v2/reply"),
        ("reply web mode=3", {"oid": aid, "type": 1, "mode": 3, "pagination_str": pag,
                              "plat": 1, "web_location": "333.999"},
         "https://api.bilibili.com/x/v2/reply/web"),
    ]
    for tag, params, url in variants:
        j = try_get(tag, url, params)
        n = count_replies(j)
        if n >= 0:
            print(f"    -> replies={n}")
            if n:
                d = j["data"]
                reps = d.get("replies") or []
                print("    first ctime", ts2s(reps[0].get("ctime")))
        time.sleep(1.0)

    print("\n---- danmaku ----")
    if cid:
        for tag, url in [
            ("dm list.so", f"https://api.bilibili.com/x/v1/dm/list.so?oid={cid}"),
            ("dm seg.so", f"https://api.bilibili.com/x/v2/dm/web/seg.so?type=1&oid={cid}&pid={aid}&segment_index=1"),
        ]:
            try:
                r = S.get(url, timeout=20)
                raw = r.content
                print(f"[{tag}] HTTP {r.status_code} bytes={len(raw)} "
                      f"ctype={r.headers.get('Content-Type')}")
                txt = raw.decode("utf-8", errors="ignore")
                ds = re.findall(r'<d p="([^"]*)">([^<]*)</d>', txt)
                print(f"    danmaku_parsed={len(ds)}")
                for p_, t_ in ds[:3]:
                    parts = p_.split(",")
                    print(f"    ctime={ts2s(parts[4]) if len(parts)>4 else '?'} | {t_[:40]}")
                if ds:
                    hits = [t_ for _, t_ in ds if "绷" in t_]
                    print(f"    含'绷'= {len(hits)}")
            except Exception as e:
                print(f"[{tag}] EXC {type(e).__name__}: {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    bili_part()
