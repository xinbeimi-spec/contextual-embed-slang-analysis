# -*- coding: utf-8 -*-
"""Probe bilibili anti-crawl chain: finger/spi + GenWebTicket + wbi/view."""
import sys
import time
import json
import hmac
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
S.headers.update({
    "User-Agent": UA,
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
})


def p(tag, obj, n=200):
    print(f"[{tag}] {str(obj)[:n]}")


def get_json(url, params=None):
    r = S.get(url, params=params, timeout=20)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"_raw": r.text[:200]}


def boot_cookies():
    print("-- homepage --")
    r = S.get("https://www.bilibili.com/", timeout=20)
    print("homepage", r.status_code, sorted(S.cookies.get_dict().keys()))

    print("-- finger/spi --")
    st, j = get_json("https://api.bilibili.com/x/frontend/finger/spi")
    p("spi", j)
    if j.get("code") == 0:
        d = j.get("data") or {}
        if d.get("b_3"):
            S.cookies.set("buvid3", d["b_3"], domain=".bilibili.com")
        if d.get("b_4"):
            S.cookies.set("buvid4", d["b_4"], domain=".bilibili.com")
    print("cookies:", sorted(S.cookies.get_dict().keys()))

    print("-- GenWebTicket --")
    ts = int(time.time())
    msg = f"ts{ts}".encode()
    sign = hmac.new(b"XgwSnGZ1p", msg, hashlib.sha256).hexdigest()
    st, j = get_json(
        "https://api.bilibili.com/bapis/bilibili.api.ticket.v1.Ticket/GenWebTicket",
        {"key_id": "ec02", "hexsign": sign, "context[ts]": ts, "csrf": ""},
    )
    p("ticket", j)
    if j.get("code") == 0:
        tk = (j.get("data") or {}).get("ticket")
        if tk:
            S.cookies.set("bili_ticket", tk, domain=".bilibili.com")
    print("cookies:", sorted(S.cookies.get_dict().keys()))


def get_wbi():
    st, j = get_json("https://api.bilibili.com/x/web-interface/nav")
    img = j["data"]["wbi_img"]["img_url"].rsplit("/", 1)[-1].split(".")[0]
    sub = j["data"]["wbi_img"]["sub_url"].rsplit("/", 1)[-1].split(".")[0]
    orig = img + sub
    return "".join(orig[i] for i in MIXIN_TAB)[:32]


def sign(params, mk):
    pr = dict(params)
    pr["wts"] = int(time.time())
    clean = [(k, "".join(c for c in str(v) if c not in "!'()*"))
             for k, v in sorted(pr.items())]
    q = urllib.parse.urlencode(clean)
    pr["w_rid"] = hashlib.md5((q + mk).encode()).hexdigest()
    return pr


def main():
    now = datetime.datetime.now()
    print("NOW =", now.isoformat(), " ts =", int(now.timestamp()))
    print("now-2y =", (now - datetime.timedelta(days=730)).date(),
          " ts=", int((now - datetime.timedelta(days=730)).timestamp()))

    boot_cookies()
    mk = get_wbi()

    print("\n-- search, dump item keys --")
    sp = sign({"search_type": "video", "keyword": "绷不住", "page": 1,
               "page_size": 30, "order": "totalrank"}, mk)
    st, j = get_json("https://api.bilibili.com/x/web-interface/wbi/search/type", sp)
    p("search", f"code={j.get('code')}")
    item = None
    if j.get("code") == 0:
        res = j["data"].get("result") or []
        if res:
            item = res[0]
            print("ITEM KEYS:", sorted(item.keys()))
            for k in ("aid", "bvid", "pubdate", "play", "author", "description",
                      "tag", "typename", "video_review", "favorites", "duration"):
                if k in item:
                    p("  " + k, item[k], 120)
    time.sleep(1.0)

    bvid = item.get("bvid") if item else None
    aid = item.get("aid") if item else None

    print("\n-- wbi/view --")
    vp = sign({"bvid": bvid}, mk)
    st, j = get_json("https://api.bilibili.com/x/web-interface/wbi/view", vp)
    p("wbi/view", f"code={j.get('code')} msg={j.get('message')}")
    if j.get("code") == 0:
        d = j["data"]
        aid = d["aid"]
        print("  aid", aid, "stat", d["stat"])
    time.sleep(1.0)

    print("\n-- plain view --")
    st, j = get_json("https://api.bilibili.com/x/web-interface/view",
                     {"bvid": bvid})
    p("view", f"code={j.get('code')} msg={j.get('message')}")
    time.sleep(1.0)

    print("\n-- reply/wbi/main --")
    if aid:
        pag = json.dumps({"offset": ""}, separators=(",", ":"))
        rp = sign({"oid": aid, "type": 1, "mode": 3, "plat": 1,
                   "web_location": "1315875", "pagination_str": pag}, mk)
        st, j = get_json("https://api.bilibili.com/x/v2/reply/wbi/main", rp)
        p("reply/main", f"code={j.get('code')} msg={j.get('message')}")
        if j.get("code") == 0:
            d = j.get("data") or {}
            reps = d.get("replies") or []
            print("  got", len(reps))
            for c in reps[:3]:
                print("   ctime", c.get("ctime"), "like", c.get("like"),
                      "|", str((c.get("content") or {}).get("message", ""))[:50])
            print("  cursor keys:", sorted((d.get("cursor") or {}).keys()))
            print("  pagination_reply:",
                  str((d.get("cursor") or {}).get("pagination_reply"))[:120])
        time.sleep(1.0)

    print("\n-- reply old --")
    if aid:
        st, j = get_json("https://api.bilibili.com/x/v2/reply",
                         {"type": 1, "oid": aid, "pn": 1, "ps": 20, "sort": 2})
        p("reply/old", f"code={j.get('code')} msg={j.get('message')}")
        if j.get("code") == 0:
            print("  got", len((j.get("data") or {}).get("replies") or []))


if __name__ == "__main__":
    main()
