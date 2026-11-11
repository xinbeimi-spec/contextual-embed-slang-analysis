# -*- coding: utf-8 -*-
"""Probe pagination ordering on a high-comment video."""
import sys
import time
import json
import hashlib
import urllib.parse
import datetime

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIXIN_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43,
             5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16,
             24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59,
             6, 63, 57, 62, 11, 36, 20, 34, 44, 52]

S = requests.Session()
S.headers.update({"User-Agent": UA, "Referer": "https://www.bilibili.com/",
                  "Origin": "https://www.bilibili.com",
                  "Accept": "application/json, text/plain, */*"})


def ts2s(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")


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


def main():
    boot()
    mk = wbi()

    print("== search order=click, keyword=绷 ==")
    sp = sign({"search_type": "video", "keyword": "绷", "page": 1,
               "page_size": 30, "order": "click"}, mk)
    j = S.get("https://api.bilibili.com/x/web-interface/wbi/search/type",
              params=sp, timeout=20).json()
    res = (j.get("data") or {}).get("result") or []
    print("got", len(res))
    for it in res[:8]:
        print(f"  {it.get('bvid')} pub={ts2s(int(it.get('pubdate',0)))} "
              f"play={it.get('play')} review={it.get('video_review')} "
              f"aid={it.get('aid')} | {str(it.get('title'))[:30]}")

    # pick video with most video_review
    cand = max(res, key=lambda x: int(x.get("video_review") or 0))
    aid = cand["aid"]
    print(f"\npicked aid={aid} review={cand['video_review']} "
          f"pub={ts2s(int(cand['pubdate']))}")
    time.sleep(1.0)

    for mode in (2, 3):
        print(f"\n===== mode={mode} (2=time, 3=hot) =====")
        offset = ""
        for page in range(4):
            pag = json.dumps({"offset": offset}, separators=(",", ":"))
            rp = sign({"oid": aid, "type": 1, "mode": mode, "plat": 1,
                       "web_location": "1315875", "pagination_str": pag}, mk)
            j = S.get("https://api.bilibili.com/x/v2/reply/wbi/main",
                      params=rp, timeout=20).json()
            if j.get("code") != 0:
                print(f"  page{page} code={j.get('code')} msg={j.get('message')}")
                break
            d = j.get("data") or {}
            reps = d.get("replies") or []
            cur = d.get("cursor") or {}
            print(f"  page{page} n={len(reps)} all_count={cur.get('all_count')} "
                  f"is_end={cur.get('is_end')}")
            if reps:
                print(f"    first ctime={ts2s(reps[0]['ctime'])} "
                      f"last ctime={ts2s(reps[-1]['ctime'])} "
                      f"like={reps[0].get('like')}")
                # nested replies present?
                nsub = sum(len(c.get("replies") or []) for c in reps)
                print(f"    nested_replies_on_page={nsub}")
            pr = cur.get("pagination_reply") or {}
            nxt = pr.get("next") or cur.get("next")
            if not nxt or cur.get("is_end"):
                print("    stop: no next / is_end")
                break
            offset = str(nxt)
            time.sleep(1.0)

    print("\n== old /x/v2/reply sort=0 pn=1..3 ==")
    for pn in (1, 2, 3):
        j = S.get("https://api.bilibili.com/x/v2/reply",
                  params={"type": 1, "oid": aid, "pn": pn, "ps": 20, "sort": 0},
                  timeout=20).json()
        if j.get("code") != 0:
            print(f"  pn={pn} code={j.get('code')}")
            break
        reps = (j.get("data") or {}).get("replies") or []
        if reps:
            print(f"  pn={pn} n={len(reps)} first={ts2s(reps[0]['ctime'])} "
                  f"last={ts2s(reps[-1]['ctime'])}")
        else:
            print(f"  pn={pn} n=0")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
