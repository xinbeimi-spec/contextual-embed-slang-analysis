# -*- coding: utf-8 -*-
"""Probe bilibili API availability: wbi signing, search, comment pagination."""
import sys
import time
import json
import hashlib
import urllib.parse

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

MIXIN_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]

S = requests.Session()
S.headers.update({
    "User-Agent": UA,
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
})


def get_wbi():
    r = S.get("https://api.bilibili.com/x/web-interface/nav", timeout=20)
    j = r.json()
    img = j["data"]["wbi_img"]["img_url"].rsplit("/", 1)[-1].split(".")[0]
    sub = j["data"]["wbi_img"]["sub_url"].rsplit("/", 1)[-1].split(".")[0]
    orig = img + sub
    mk = "".join(orig[i] for i in MIXIN_TAB)[:32]
    return img, sub, mk, j.get("code")


def sign(params, mk):
    p = dict(params)
    p["wts"] = int(time.time())
    items = sorted(p.items())
    clean = [(k, "".join(c for c in str(v) if c not in "!'()*")) for k, v in items]
    q = urllib.parse.urlencode(clean)
    p["w_rid"] = hashlib.md5((q + mk).encode()).hexdigest()
    return p


def show(tag, r):
    try:
        j = r.json()
    except Exception:
        print(f"[{tag}] HTTP {r.status_code} non-json, len={len(r.text)}")
        return None
    code = j.get("code")
    msg = str(j.get("message", ""))[:120]
    print(f"[{tag}] HTTP {r.status_code} code={code} msg={msg}")
    return j


def main():
    print("== step0: prime cookies ==")
    r = S.get("https://www.bilibili.com/", timeout=20)
    print("homepage", r.status_code, "cookies:", sorted(S.cookies.get_dict().keys()))

    print("\n== step1: nav / wbi keys ==")
    t0 = time.time()
    img, sub, mk, code = get_wbi()
    print(f"nav code={code} took={time.time()-t0:.2f}s")
    print("img_key len", len(img), "sub_key len", len(sub), "mixin len", len(mk))
    print("cookies now:", sorted(S.cookies.get_dict().keys()))

    print("\n== step2: search (wbi/type) ==")
    for kw in ["绷不住", "绷"]:
        params = sign({
            "search_type": "video",
            "keyword": kw,
            "page": 1,
            "page_size": 30,
            "order": "totalrank",
        }, mk)
        r = S.get("https://api.bilibili.com/x/web-interface/wbi/search/type",
                  params=params, timeout=20)
        j = show(f"search:{kw}", r)
        if j and j.get("code") == 0:
            res = j["data"].get("result") or []
            print(f"  numResults={j['data'].get('numResults')} got={len(res)}")
            for it in res[:3]:
                print(f"  - {it.get('bvid')} pubdate={it.get('pubdate')} "
                      f"play={it.get('play')} title={str(it.get('title'))[:40]}")
        time.sleep(1.2)

    print("\n== step3: resolve aid from bvid ==")
    bvid = None
    params = sign({"search_type": "video", "keyword": "绷不住", "page": 1,
                   "page_size": 30, "order": "totalrank"}, mk)
    r = S.get("https://api.bilibili.com/x/web-interface/wbi/search/type",
              params=params, timeout=20)
    j = r.json()
    if j.get("code") == 0:
        res = j["data"].get("result") or []
        if res:
            bvid = res[0].get("bvid")
    print("bvid =", bvid)
    aid = None
    if bvid:
        r = S.get("https://api.bilibili.com/x/web-interface/view",
                  params={"bvid": bvid}, timeout=20)
        j = show("view", r)
        if j and j.get("code") == 0:
            aid = j["data"]["aid"]
            st = j["data"]["stat"]
            print(f"  aid={aid} reply={st['reply']} view={st['view']} "
                  f"pubdate={j['data']['pubdate']}")

    print("\n== step4: comment api, old /x/v2/reply ==")
    if aid:
        r = S.get("https://api.bilibili.com/x/v2/reply",
                  params={"type": 1, "oid": aid, "pn": 1, "ps": 20, "sort": 2},
                  timeout=20)
        j = show("reply-old", r)
        if j and j.get("code") == 0:
            reps = (j.get("data") or {}).get("replies") or []
            print(f"  got={len(reps)}")
            for c in reps[:2]:
                print(f"  - ctime={c.get('ctime')} like={c.get('like')} "
                      f"msg={str(c.get('content',{}).get('message',''))[:40]}")
        time.sleep(1.2)

    print("\n== step5: comment api, new /x/v2/reply/wbi/main ==")
    if aid:
        pag = json.dumps({"offset": ""}, separators=(",", ":"))
        params = sign({
            "oid": aid, "type": 1, "mode": 3, "plat": 1,
            "web_location": "1315875", "pagination_str": pag,
        }, mk)
        r = S.get("https://api.bilibili.com/x/v2/reply/wbi/main",
                  params=params, timeout=20)
        j = show("reply-main", r)
        if j and j.get("code") == 0:
            d = j.get("data") or {}
            reps = d.get("replies") or []
            print(f"  got={len(reps)} cursor_next={str(d.get('cursor',{}).get('pagination_reply',{}))[:80]}")
            for c in reps[:2]:
                print(f"  - ctime={c.get('ctime')} like={c.get('like')} "
                      f"msg={str(c.get('content',{}).get('message',''))[:40]}")


if __name__ == "__main__":
    main()
