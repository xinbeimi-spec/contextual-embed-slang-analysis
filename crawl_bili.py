# -*- coding: utf-8 -*-
"""bilibili crawler: video discovery -> danmaku (works anonymously)
                        -> comments (guest cap = 3/video unless SESSDATA given).

Storage under data/raw/bilibili/:
  videos.jsonl          one row per discovered video (metadata + counts)
  danmaku_all.jsonl     ALL danmaku text of crawled videos  (denominator / reuse)
  hits_danmaku.jsonl    danmaku containing the target char  (full record schema)
  hits_comment.jsonl    comments containing the target char
  _state.json           resume state (done bvids)

Usage:
  python crawl_bili.py --phase discover
  python crawl_bili.py --phase crawl --per-year 22 --target-hits 2500
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse
import datetime
import urllib.parse
import re

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

BILI_DIR = os.path.join(C.RAW, "bilibili")
F_VIDEOS = os.path.join(BILI_DIR, "videos.jsonl")
F_DM_ALL = os.path.join(BILI_DIR, "danmaku_all.jsonl")
F_DM_HITS = os.path.join(BILI_DIR, "hits_danmaku.jsonl")
F_CM_HITS = os.path.join(BILI_DIR, "hits_comment.jsonl")
F_STATE = os.path.join(BILI_DIR, "_state.json")

VERSION = "bili_dm_v1"
P_MIXIN = C.BILI_MIXIN_TAB

# traditional-only characters, used for a lightweight 繁简 heuristic.
# Proper conversion needs opencc, which is deliberately not installed.
TRAD_HINTS = set("這個們說時麼來對開間關為實現學國會後發體產經濟電腦網絡讓還見"
                 "過點無錯這樣樣處麼幾當親愛鄉東車馬長門問題")


def script_guess(text):
    if not text:
        return "unknown"
    return "traditional_or_mixed" if (set(text) & TRAD_HINTS) else "simplified"


# --------------------------------------------------------------------- helpers
def sha1(s, n=16):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:n]


def iso(t):
    try:
        return datetime.datetime.fromtimestamp(int(t)).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def year_of(t):
    try:
        return datetime.datetime.fromtimestamp(int(t)).year
    except Exception:
        return None


def jappend(path, rows):
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


# ---------------------------------------------------------------- http client
class Bili:
    def __init__(self, sessdata=None):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": C.UA_PC,
            "Referer": "https://www.bilibili.com/",
            "Origin": "https://www.bilibili.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
        self.sessdata = sessdata
        self.mk = None
        self.boot()

    def sleep(self):
        time.sleep(C.REQUEST_DELAY * (0.75 + random.random() * 0.6))

    def boot(self):
        try:
            self.s.get("https://www.bilibili.com/", timeout=20)
            j = self.s.get(f"{C.BILI_API}/x/frontend/finger/spi", timeout=20).json()
            d = j.get("data") or {}
            if d.get("b_3"):
                self.s.cookies.set("buvid3", d["b_3"], domain=".bilibili.com")
            if d.get("b_4"):
                self.s.cookies.set("buvid4", d["b_4"], domain=".bilibili.com")
            if self.sessdata:
                self.s.cookies.set("SESSDATA", self.sessdata, domain=".bilibili.com")
        except Exception as e:
            print("  ! boot warn:", e, file=sys.stderr)
        self.refresh_wbi()

    def refresh_wbi(self):
        j = self.s.get(f"{C.BILI_API}/x/web-interface/nav", timeout=20).json()
        img = j["data"]["wbi_img"]["img_url"].rsplit("/", 1)[-1].split(".")[0]
        sub = j["data"]["wbi_img"]["sub_url"].rsplit("/", 1)[-1].split(".")[0]
        o = img + sub
        self.mk = "".join(o[i] for i in P_MIXIN)[:32]

    def sign(self, params):
        pr = dict(params)
        pr["wts"] = int(time.time())
        clean = [(k, "".join(c for c in str(v) if c not in "!'()*"))
                 for k, v in sorted(pr.items())]
        q = urllib.parse.urlencode(clean)
        pr["w_rid"] = hashlib.md5((q + self.mk).encode()).hexdigest()
        return pr

    def get(self, url, params=None, signed=True, allow_nonjson=False):
        last = None
        for attempt in range(C.MAX_RETRY):
            try:
                pp = self.sign(params) if (signed and self.mk) else params
                r = self.s.get(url, params=pp, timeout=25)
                if r.status_code == 412:
                    raise RuntimeError("412 banned")
                if allow_nonjson:
                    return r.content if r.status_code == 200 else None
                j = r.json()
                code = j.get("code")
                if code == 0:
                    return j
                if code in (-412, -352, -509, 103):
                    raise RuntimeError(f"risk control code={code}")
                return j  # legitimate non-zero (e.g. -404 deleted)
            except Exception as e:
                last = e
                time.sleep(3 + attempt * 4 + random.random() * 2)
                if attempt == 1:
                    try:
                        self.boot()
                    except Exception:
                        pass
        print(f"  ! give up after {C.MAX_RETRY}: {url.split('/')[-1]} {last}", file=sys.stderr)
        return None


# ------------------------------------------------------------------ discovery
def discover(b, pages=5, keywords=None, stop_at=None):
    seen = {}
    if os.path.exists(F_VIDEOS):
        for r in read_jsonl(F_VIDEOS):
            seen[r["bvid"]] = r
        print(f"  resume: {len(seen)} videos already known")

    n_req = 0
    n_fail = 0
    for kw in (keywords if keywords else C.SEARCH_KEYWORDS):
        for order in C.SEARCH_ORDERS:
            for page in range(1, pages + 1):
                j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
                          {"search_type": "video", "keyword": kw, "page": page,
                           "page_size": 30, "order": order})
                n_req += 1
                if not j or j.get("code") != 0:
                    n_fail += 1
                    print(f"    ! {kw} {order} p{page}: "
                          f"code={j.get('code') if j else 'None'}")
                    b.sleep()
                    continue
                res = (j.get("data") or {}).get("result") or []
                if not res:
                    b.sleep()
                    break
                new = 0
                for it in res:
                    bvid = it.get("bvid")
                    if not bvid or bvid in seen:
                        continue
                    pub = it.get("pubdate")
                    seen[bvid] = {
                        "bvid": bvid,
                        "aid": it.get("aid"),
                        "title": re.sub(r"<[^>]+>", "", str(it.get("title") or "")),
                        "pubdate_epoch": pub,
                        "pubdate_iso": iso(pub or 0),
                        "year": year_of(pub or 0),
                        "play": it.get("play"),
                        "video_review": it.get("video_review"),
                        "author": it.get("author"),
                        "typename": it.get("typename"),
                        "tags": it.get("tag"),
                        "source_query": kw,
                        "source_order": order,
                        "crawl_date": C.CRAWL_DATE,
                    }
                    new += 1
                print(f"  [{kw}/{order}/p{page}] +{new} (total {len(seen)})")
                b.sleep()
                if stop_at and len(seen) >= stop_at:
                    save_videos(seen)
                    print(f"stop_at={stop_at} reached -> {len(seen)} videos")
                    return seen
    save_videos(seen)
    print(f"discover done: {len(seen)} videos, {n_req} requests, {n_fail} failed")
    return seen


def discover_by_year(b, years, pages_new=3, pages_old=3, keywords=None):
    """Year-stratified discovery using B站's pubtime_begin_s / pubtime_end_s filter.

    This is the ONLY way to get pre-2021 videos: plain relevance search is
    dominated by recent uploads, so the default pool is ~100% post-2023.
    """
    import datetime as _dt

    kws_all = list(keywords) if keywords else list(C.SEARCH_KEYWORDS)
    seen = {}
    if os.path.exists(F_VIDEOS):
        for r in read_jsonl(F_VIDEOS):
            seen[r["bvid"]] = r
    print(f"  resume: {len(seen)} videos already known")

    for y in years:
        beg = int(_dt.datetime(y, 1, 1).timestamp())
        end = int(_dt.datetime(y, 12, 31, 23, 59, 59).timestamp())
        # Order matters a lot: with a pubtime filter, order=totalrank returns
        # exactly ONE page (page 2 is empty), while order=pubdate and order=click
        # paginate up to their real numResults (~60 for a single keyword-year).
        # Sweeping all three orders is what makes per-year recall acceptable.
        kws = kws_all
        pages = pages_old if y <= 2020 else pages_new
        y_new = 0
        y_req = 0
        y_fail = 0
        for order in C.SEARCH_ORDERS:
            for kw in kws:
                for page in range(1, pages + 1):
                    j = b.get(f"{C.BILI_API}/x/web-interface/wbi/search/type",
                              {"search_type": "video", "keyword": kw, "page": page,
                               "page_size": 30, "order": order,
                               "pubtime_begin_s": beg, "pubtime_end_s": end})
                    y_req += 1
                    if not j or j.get("code") != 0:
                        y_fail += 1
                        print(f"    ! {y} {order} {kw} p{page}: "
                              f"code={j.get('code') if j else 'None'}")
                        b.sleep()
                        continue
                    res = (j.get("data") or {}).get("result") or []
                    if not res:
                        b.sleep()
                        break
                    for it in res:
                        bvid = it.get("bvid")
                        if not bvid or bvid in seen:
                            continue
                        pub = it.get("pubdate")
                        seen[bvid] = {
                            "bvid": bvid,
                            "aid": it.get("aid"),
                            "title": re.sub(r"<[^>]+>", "", str(it.get("title") or "")),
                            "pubdate_epoch": pub,
                            "pubdate_iso": iso(pub or 0),
                            "year": year_of(pub or 0),
                            "play": it.get("play"),
                            "video_review": it.get("video_review"),
                            "author": it.get("author"),
                            "typename": it.get("typename"),
                            "tags": it.get("tag"),
                            "source_query": kw,
                            "source_order": f"{order}+pubtime",
                            "search_year_window": y,
                            "crawl_date": C.CRAWL_DATE,
                        }
                        y_new += 1
                    b.sleep()
        flag = "  <<< PARTIAL (requests failed, year likely under-collected)" if y_fail else ""
        print(f"  year {y}: +{y_new} new | requests={y_req} failed={y_fail}"
              f" (total {len(seen)}){flag}")

    save_videos(seen)
    print(f"year-stratified discover done: {len(seen)} videos")
    return seen


def coverage_table(seen):
    from collections import Counter
    by_year = Counter()
    for v in seen.values():
        by_year[v["year"]] += 1
    print("\n=== discovered videos by pubdate year ===")
    for y in sorted(k for k in by_year if k):
        print(f"  {y}: {by_year[y]}")
    rows = [{"year": y, "n_videos": by_year[y]} for y in sorted(k for k in by_year if k)]
    import csv
    p = os.path.join(C.TABLES, "coverage_bili_videos.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["year", "n_videos"])
        w.writeheader()
        w.writerows(rows)
    print("  ->", p)


# ---------------------------------------------------------------------- crawl
def pick_videos(seen, per_year, max_videos):
    """Select videos to crawl.

    Ranking rules (learned the hard way):
      1. title containing the target char is by far the strongest predictor of
         danmaku containing it (measured 0.05% vs ~3% hit rate);
      2. within a year, prefer high comment volume;
      3. order the final list ROUND-ROBIN ACROSS YEARS so that any early stop
         (target hit count) still leaves every year represented.
    """
    from collections import defaultdict
    by_year = defaultdict(list)
    for v in seen.values():
        if v["year"] and C.YEAR_MIN <= v["year"] <= C.YEAR_MAX:
            v["_title_hit"] = 1 if C.TARGET_CHAR in (v.get("title") or "") else 0
            by_year[v["year"]].append(v)

    per = {}
    for y in sorted(by_year):
        pool = [x for x in by_year[y] if x["_title_hit"]]
        if not pool:                      # keep a trickle from non-title matches
            pool = list(by_year[y])
            pool.sort(key=lambda x: -(x.get("video_review") or 0))
            pool = pool[:max(3, per_year // 4)]
        else:
            pool.sort(key=lambda x: -(x.get("video_review") or 0))
            keep = per_year * 2 if y <= 2020 else per_year
            pool = pool[:keep]
        per[y] = pool
        print(f"  year {y}: candidates={len(by_year[y])} "
              f"title_hit={sum(1 for x in by_year[y] if x['_title_hit'])} take={len(pool)}")

    out, i = [], 0
    while True:
        added = False
        for y in sorted(per):
            if i < len(per[y]):
                out.append(per[y][i])
                added = True
        if not added:
            break
        i += 1
    return out[:max_videos]


def parse_danmaku(xml_bytes):
    txt = xml_bytes.decode("utf-8", errors="ignore")
    out = []
    for m in re.finditer(r'<d p="([^"]*)">([^<]*)</d>', txt):
        p, t = m.group(1), m.group(2)
        parts = p.split(",")
        if len(parts) < 8:
            continue
        try:
            ctime = int(float(parts[4]))
        except Exception:
            continue
        out.append({"ctime": ctime, "text": t, "dmid": parts[7],
                    "progress": parts[0], "uid_crc": parts[6]})
    return out


def make_hit_text(text, content_type, base):
    """Return a full record dict if text contains target char, else None."""
    idx = text.find(C.TARGET_CHAR)
    if idx < 0:
        return None
    rec = dict(base)
    rec.update({
        "id": None,                                   # filled with item_id_hash
        "content_type": content_type,
        "text_raw": text,
        "text_clean": text,
        "target_word": C.TARGET_WORD,
        "target_start": idx,
        "target_end": idx + len(C.TARGET_CHAR),
        "target_count": text.count(C.TARGET_CHAR),
        "sentence": text,
        "ctx_before": "",
        "ctx_after": "",
        "context_sufficiency": None,
        "sense_label": None,
        "n_chars": len(text),
        "script": script_guess(text),
        "dup_group": None,
        "crawl_time": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return rec


# Title filters for the two "tail" selection modes. 2026-11-09: rewritten for
# 哈 (hā). A title carrying a network item (哈基米/哈人/哈哈哈…) is by far the
# densest source of hits in comment threads; a title carrying a concrete
# dictionary form (哈欠/哈腰/马大哈…) is the only source of pre-2020 material.
MEME_RE = re.compile(r"哈基米|哈人|哈哈哈|哈喽|哈皮|哈韩|哈日")


TRAD_TITLE_RE = re.compile(r"哈欠|哈腰|哈气|马大哈|哈巴狗|打哈哈|哈哈镜")


def pick_trad_videos(seen, per_year, limit):
    """Videos whose TITLE carries a TRADITIONAL 绷-form.

    These are the only realistic source of A1/A2 data: the medical/skincare
    videos dominate 2016-2021, which is exactly where the corpus is thin.
    """
    from collections import defaultdict
    cand = [v for v in seen.values()
            if v.get("year") and C.YEAR_MIN <= v["year"] <= C.YEAR_MAX
            and TRAD_TITLE_RE.search(v.get("title") or "")]
    by_year = defaultdict(list)
    for v in cand:
        by_year[v["year"]].append(v)
    per = {}
    for y in sorted(by_year):
        vs = sorted(by_year[y], key=lambda x: -(x.get("video_review") or 0))
        per[y] = vs[:per_year]
        print(f"  year {y}: trad_titled={len(by_year[y])} take={len(per[y])}")
    out, i = [], 0
    while True:
        added = False
        for y in sorted(per):
            if i < len(per[y]):
                out.append(per[y][i]); added = True
        if not added:
            break
        i += 1
    return out[:limit]


def pick_all_videos(seen, per_year, limit):
    """Take EVERY discovered video, not only title matches.

    The title-hit pool is exhausted after pass 2; the remaining ~3.6k videos
    matched on tag/description. Their comment density is lower, but they are
    the only remaining source of 绷 contexts on bilibili -- and they carry the
    traditional senses (medical bandage videos, skincare "紧绷" videos).
    Titles sort first so the highest-yield items are still processed earliest.
    """
    from collections import defaultdict
    by_year = defaultdict(list)
    for v in seen.values():
        if v["year"] and C.YEAR_MIN <= v["year"] <= C.YEAR_MAX:
            by_year[v["year"]].append(v)
    per = {}
    for y in sorted(by_year):
        vs = by_year[y]
        vs.sort(key=lambda x: (0 if C.TARGET_CHAR in (x.get("title") or "") else 1,
                               -(x.get("video_review") or 0)))
        per[y] = vs[:per_year]
        print(f"  year {y}: all={len(vs)} take={len(per[y])}")
    out, i = [], 0
    while True:
        added = False
        for y in sorted(per):
            if i < len(per[y]):
                out.append(per[y][i])
                added = True
        if not added:
            break
        i += 1
    return out[:limit]


def pick_query_videos(seen, queries, limit):
    """Videos that were DISCOVERED BY a specific search string.

    Motivation (2026-10-21): the pool was built by 30+ queries, but the crawl
    passes so far were selected by year/title, so entire query strata have
    never been touched. The 紧绷 / 绷紧 / 绷着脸 strata are exactly where the
    traditional senses live (skincare "紧绷感", fitness "脚踝紧绷", 板脸
    "绷着脸"), and they are the only place the proposed 紧绷 sense can grow.
    Selecting by `source_query` is the cheapest way to spend requests where the
    new category needs evidence.

    Highest comment volume first, no per-year quota (the strata are small).
    """
    qs = {q.strip() for q in (queries or []) if q.strip()}
    cand = [v for v in seen.values()
            if (v.get("source_query") or "").strip() in qs]
    # a video can be discovered by several queries; `source_query` keeps the
    # first, so also accept title matches for the same strings
    have = {v["bvid"] for v in cand}
    titled = [v for v in seen.values()
              if v["bvid"] not in have
              and any(q in (v.get("title") or "") for q in qs)]
    pool = cand + titled
    pool.sort(key=lambda x: -(x.get("video_review") or 0))
    print(f"  query strata {sorted(qs)}: {len(cand)} by query, "
          f"+{len(titled)} by title = {len(pool)} candidates")
    return pool[:limit]


def pick_by_comment_volume(seen, limit):
    """Every remaining video, ordered by COMMENT VOLUME across all years.

    Motivation (2026-10-21, measured): after the high-comment 2026 videos were
    exhausted, the year-ordered crawl fell to ~6 hits/minute, because the rest
    of the pool is low-engagement video (5-10 sampled comments each). But hits
    are proportional to how many comments are READ, and each video's comment
    thread was previously truncated at 5 pages (~100 items). Two levers, both
    applied here:
      * order by `video_review` (the API's comment count) so the deepest
        threads are read first;
      * read them deeper (raise --max-comment-pages).
    This changes the per-video comment cap, i.e. it is a sampling design
    decision, not just a speed-up; it must be reported as such.
    """
    cand = [v for v in seen.values()
            if v.get("year") and C.YEAR_MIN <= v["year"] <= C.YEAR_MAX]
    cand.sort(key=lambda x: -(x.get("video_review") or 0))
    print(f"  by comment volume: {len(cand)} candidates, top has "
          f"{cand[0].get('video_review') if cand else 0} comments")
    return cand[:limit]


def pick_meme_videos(seen, per_year, limit):
    """Prefer videos whose TITLE carries the new-usage phrase (meme videos).

    Rationale: the base rate of '绷' in comments is ~0.4% overall, but a
    '绷不住了' compilation has comments that reproduce the phrase constantly.
    Selection, not volume, is what makes this tractable.
    """
    from collections import defaultdict
    cand = [v for v in seen.values()
            if v["year"] and C.YEAR_MIN <= v["year"] <= C.YEAR_MAX
            and MEME_RE.search(v.get("title") or "")]
    by_year = defaultdict(list)
    for v in cand:
        by_year[v["year"]].append(v)
    per = {}
    for y in sorted(by_year):
        vs = sorted(by_year[y], key=lambda x: -(x.get("video_review") or 0))
        per[y] = vs[:per_year]
        print(f"  year {y}: meme_titled={len(by_year[y])} take={len(per[y])}")
    out, i = [], 0
    while True:
        added = False
        for y in sorted(per):
            if i < len(per[y]):
                out.append(per[y][i])
                added = True
        if not added:
            break
        i += 1
    return out[:limit]


def save_videos(seen_all):
    """Persist the FULL video pool, deduped, with a one-generation backup.

    Bug history: an earlier version rewrote videos.jsonl from the *year-filtered*
    working set, which silently deleted every video outside the filter window.
    Always write from the unfiltered pool.
    """
    if os.path.exists(F_VIDEOS):
        try:
            import shutil
            shutil.copyfile(F_VIDEOS, F_VIDEOS + ".bak")
        except Exception:
            pass
    tmp = F_VIDEOS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for vv in seen_all.values():
            f.write(json.dumps(vv, ensure_ascii=False) + "\n")
    os.replace(tmp, F_VIDEOS)


def crawl(b, per_year=22, max_videos=260, target_hits=2500, max_comment_pages=5,
          skip_danmaku=False, select="roundrobin", year_range=None, min_pages=4,
          queries=None):
    seen = {r["bvid"]: r for r in read_jsonl(F_VIDEOS)}
    if not seen:
        print("no videos discovered yet; run --phase discover first")
        return
    seen_all = seen          # unfiltered pool -- the only thing we may persist
    if year_range:
        lo, hi = year_range
        seen = {k: v for k, v in seen.items()
                if v.get("year") and lo <= v["year"] <= hi}
        print(f"year filter {lo}-{hi}: {len(seen)} of {len(seen_all)} videos "
              f"(persisting the full {len(seen_all)})")
    done = set()
    if os.path.exists(F_STATE):
        try:
            done = set(json.load(open(F_STATE, encoding="utf-8")).get("done", []))
        except Exception:
            pass
    print(f"resume: done={len(done)}")

    # NOTE: filter the already-done set BEFORE truncating to max_videos,
    # otherwise a resume run can come back with an empty work list.
    if select == "meme":
        full = pick_meme_videos(seen, per_year, 10 ** 9)
    elif select == "trad":
        full = pick_trad_videos(seen, per_year, 10 ** 9)
    elif select == "all":
        full = pick_all_videos(seen, per_year, 10 ** 9)
    elif select == "query":
        full = pick_query_videos(seen, queries, 10 ** 9)
    elif select == "comments":
        full = pick_by_comment_volume(seen, 10 ** 9)
    else:
        full = pick_videos(seen, per_year, 10 ** 9)
    pool = [v for v in full if v["bvid"] not in done]
    todo = pool[:max_videos]
    print(f"to crawl: {len(todo)} videos (skipped {len(done)} already done, "
          f"select={select})")

    n_hit = len(read_jsonl(F_DM_HITS)) + len(read_jsonl(F_CM_HITS))
    print(f"existing hits (danmaku+comment): {n_hit}")
    n_start = n_hit
    t_start = time.time()

    for i, v in enumerate(todo, 1):
        aid, bvid = v["aid"], v["bvid"]
        cid = None
        dm = []
        if skip_danmaku:
            # danmaku turned out to be a dead end (0.28% hit rate, 412s),
            # so skip both the view call and the danmaku fetch entirely.
            v.setdefault("danmaku_stat", None)
        else:
            j = b.get(f"{C.BILI_API}/x/web-interface/wbi/view", {"aid": aid})
            if not j or j.get("code") != 0:
                print(f"[{i}/{len(todo)}] {bvid} view fail")
                done.add(bvid)
                continue
            d = j["data"]
            cid = d.get("cid") or (d.get("pages") or [{}])[0].get("cid")
            stat = d.get("stat") or {}
            v["cid"] = cid
            v["danmaku_stat"] = stat.get("danmaku")
            v["reply_stat"] = stat.get("reply")
            v["view_stat"] = stat.get("view")
            b.sleep()

            # ---- danmaku (anonymous OK)
            if cid:
                raw = b.get(f"{C.BILI_API}/x/v1/dm/list.so?oid={cid}",
                            signed=False, allow_nonjson=True)
                if raw:
                    dm = parse_danmaku(raw)
                b.sleep()

        dm_rows, dm_hits = [], []
        base = {
            "platform": "bilibili",
            "thread_id": sha1("bili|" + bvid),
            "thread_bvid": bvid,
            "thread_oid": aid,
            "thread_title": v["title"],
            "thread_pubdate_epoch": v["pubdate_epoch"],
            "thread_pubdate_iso": v["pubdate_iso"],
            "thread_year": v["year"],
            "thread_play": v.get("play"),
            "source_query": v.get("source_query"),
            "crawl_date": C.CRAWL_DATE,
            "crawler_version": VERSION,
        }
        for x in dm:
            ctime = x["ctime"]
            dm_rows.append({"bvid": bvid, "dmid": x["dmid"], "ctime": ctime,
                            "year": year_of(ctime), "text": x["text"]})
            r = make_hit_text(x["text"], "danmaku", {})
            if r is None:
                continue
            r.update(base)
            r["item_id_hash"] = sha1("dm|" + bvid + "|" + x["dmid"])
            r["id"] = r["item_id_hash"]
            r["parent_id_hash"] = None
            r["author_hash"] = sha1("uidcrc|" + (x["uid_crc"] or "")) if x["uid_crc"] else None
            r["ctime_epoch"] = ctime
            r["ctime_iso"] = iso(ctime)
            r["send_year"] = year_of(ctime)
            r["time_precision"] = "second"
            r["like_count"] = None
            r["reply_count"] = None
            r["dm_progress_sec"] = x["progress"]
            dm_hits.append(r)

        # ---- comments (guest cap 3/video; full pagination needs SESSDATA)
        cm_hits = []
        cm_seen = 0
        offset = ""
        for cp in range(max_comment_pages):
            pag = json.dumps({"offset": offset}, separators=(",", ":"))
            cj = b.get(f"{C.BILI_API}/x/v2/reply/wbi/main",
                       {"oid": aid, "type": 1, "mode": 3, "plat": 1,
                        "web_location": "1315875", "pagination_str": pag})
            if not cj or cj.get("code") != 0:
                break
            cd = cj.get("data") or {}
            reps = cd.get("replies") or []
            if not reps:
                break
            flat = []
            for c in reps:
                flat.append((c, None))
                for sub in (c.get("replies") or []):
                    flat.append((sub, c.get("rpid")))
            for c, parent in flat:
                msg = ((c.get("content") or {}).get("message") or "")
                cm_seen += 1
                r = make_hit_text(msg, "comment", {})
                if r is None:
                    continue
                r.update(base)
                r["item_id_hash"] = sha1("cm|" + bvid + "|" + str(c.get("rpid")))
                r["id"] = r["item_id_hash"]
                r["parent_id_hash"] = sha1("rpid|" + str(parent)) if parent else None
                r["author_hash"] = sha1("mid|" + str(c.get("mid"))) if c.get("mid") else None
                r["ctime_epoch"] = c.get("ctime")
                r["ctime_iso"] = iso(c.get("ctime") or 0)
                r["time_precision"] = "second"
                r["like_count"] = c.get("like")
                r["reply_count"] = c.get("rcount")
                cm_hits.append(r)
            cur = cd.get("cursor") or {}
            # 2026-11-09 fix: the API returns the cursor as
            # {"pagination_reply": {"next_offset": ...}} ("next" was the old
            # shape). Reading only "next" silently stopped every thread at page
            # 1 (~20 top comments), which is why the corpus had ~50 comments
            # per video. Accept both keys.
            pr = cur.get("pagination_reply") or {}
            nxt = pr.get("next") or pr.get("next_offset")
            if not nxt or cur.get("is_end"):
                break
            # adaptive early stop: the base rate of a rare char is ~0.3%, so a
            # video that produced nothing in its first pages almost never
            # produces anything later. Saves ~2-3x requests.
            if (cp + 1) >= min_pages and len(cm_hits) == 0:
                break
            offset = nxt
            b.sleep()

        # Politeness: the comment loop only slept between *pages*, so videos
        # whose first page came back with is_end=True got ZERO delay. Force one
        # sleep per video so a long run cannot trip risk control.
        b.sleep()

        v["n_danmaku"] = len(dm)
        v["n_comment_sampled"] = cm_seen
        v["n_danmaku_hits"] = len(dm_hits)
        v["n_comment_hits"] = len(cm_hits)

        jappend(F_DM_ALL, dm_rows)
        jappend(F_DM_HITS, dm_hits)
        jappend(F_CM_HITS, cm_hits)
        done.add(bvid)
        n_hit += len(dm_hits) + len(cm_hits)

        el = time.time() - t_start
        gained = n_hit - n_start
        rate = gained / el if el > 0 else 0
        eta = (target_hits - n_hit) / rate / 60 if rate > 0 and n_hit < target_hits else 0
        print(f"[{i}/{len(todo)}] {bvid} y={v['year']} dm={len(dm)} "
              f"dm_hits={len(dm_hits)} cm={cm_seen} cm_hits={len(cm_hits)} "
              f"| total_hits={n_hit} ETA={eta:.0f}min")

        if i % 10 == 0:
            json.dump({"done": sorted(done)}, open(F_STATE, "w", encoding="utf-8"))
            # persist the FULL pool (never the year-filtered subset)
            save_videos(seen_all)
        if n_hit >= target_hits:
            print(f"reached target {target_hits}, stopping")
            break

    json.dump({"done": sorted(done)}, open(F_STATE, "w", encoding="utf-8"))
    save_videos(seen_all)
    print(f"\nDONE. danmaku hits total={n_hit}, elapsed={(time.time()-t_start)/60:.1f}min")


# -------------------------------------------------------------------- summary
def summary():
    """Write the coverage tables. The time axis is the SEND year of each
    comment, not the video's upload year -- a 2016 video keeps collecting
    comments for a decade, so video year is not the language-time axis."""
    import csv
    from collections import Counter

    cm = read_jsonl(F_CM_HITS)
    dm = read_jsonl(F_DM_HITS)
    all_dm = read_jsonl(F_DM_ALL)
    print(f"comment hits = {len(cm)}  danmaku hits = {len(dm)}  "
          f"all danmaku stored = {len(all_dm)}")

    def bucket(rows):
        send = Counter()
        vid = Counter()
        for r in rows:
            send[year_of(r.get("ctime_epoch"))] += 1
            vid[r.get("thread_year")] += 1
        return send, vid

    cm_send, cm_vid = bucket(cm)
    dm_send, dm_vid = bucket(dm)

    print("\n=== HITS BY SEND YEAR (the real time axis) ===")
    print(f"{'year':6}{'comment':>9}{'danmaku':>9}{'total':>8}")
    for y in sorted(k for k in set(cm_send) | set(dm_send) if k):
        print(f"{y:6}{cm_send[y]:9d}{dm_send[y]:9d}{cm_send[y]+dm_send[y]:8d}")

    print("\n=== HITS BY VIDEO UPLOAD YEAR (secondary) ===")
    for y in sorted(k for k in set(cm_vid) | set(dm_vid) if k):
        print(f"  {y}: comment={cm_vid[y]:5d} danmaku={dm_vid[y]:5d}")

    rows = []
    for y in sorted(k for k in set(cm_send) | set(dm_send) | set(cm_vid) if k):
        rows.append({
            "year": y,
            "comment_hits": cm_send[y],
            "danmaku_hits": dm_send[y],
            "total_hits": cm_send[y] + dm_send[y],
            "comment_hits_by_video_year": cm_vid[y],
        })
    p = os.path.join(C.TABLES, f"coverage_{C.WORD_SLUG}_bili.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["year"])
        w.writeheader()
        w.writerows(rows)
    print("\n->", p)


def check_cookie(b, n=3):
    """Verify whether the supplied cookie lifts the 3-comment guest cap."""
    seen = {r["bvid"]: r for r in read_jsonl(F_VIDEOS)}
    cand = [v for v in seen.values() if C.TARGET_CHAR in (v.get("title") or "")]
    cand.sort(key=lambda x: -(x.get("video_review") or 0))
    print(f"cookie check over {n} videos (SESSDATA present: {bool(b.sessdata)})")
    ok_all = True
    for v in cand[:n]:
        pag = json.dumps({"offset": ""}, separators=(",", ":"))
        j = b.get(f"{C.BILI_API}/x/v2/reply/wbi/main",
                  {"oid": v["aid"], "type": 1, "mode": 3, "plat": 1,
                   "web_location": "1315875", "pagination_str": pag})
        if j and j.get("code") == 0:
            d = j.get("data") or {}
            cur = d.get("cursor") or {}
            nrep = len(d.get("replies") or [])
            print(f"  {v['bvid']} replies={nrep} all_count={cur.get('all_count')} "
                  f"is_end={cur.get('is_end')}")
            if nrep <= 3:
                ok_all = False
        else:
            print(f"  {v['bvid']} code={j.get('code') if j else None}")
            ok_all = False
        b.sleep()
    if ok_all:
        print("\nRESULT: full comment access OK -> ready to crawl")
    else:
        print("\nRESULT: still capped at 3 -> cookie insufficient or not applied")
    return ok_all


def load_sessdata(path):
    """Accept any of these paste formats:
         SESSDATA=xxxxx
         Cookie: SESSDATA=xxx; buvid3=yyy; bili_ticket=zzz
         xxxxx                      (raw value only)
    Never prints the value.
    """
    if not os.path.exists(path):
        return None
    raw = open(path, encoding="utf-8").read().strip()
    if not raw:
        return None
    m = re.search(r"SESSDATA\s*=\s*([^;,\s]+)", raw, re.I)
    if m:
        return m.group(1).strip()
    for line in raw.splitlines():
        line = line.strip()
        if not line or "=" in line:
            continue
        return line
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="crawl",
                    choices=["discover", "discover-year", "crawl", "summary",
                             "all", "check"])
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--per-year", type=int, default=22)
    ap.add_argument("--max-videos", type=int, default=260)
    ap.add_argument("--target-hits", type=int, default=2500)
    ap.add_argument("--max-comment-pages", type=int, default=5)
    ap.add_argument("--skip-danmaku", action="store_true", default=True)
    ap.add_argument("--with-danmaku", dest="skip_danmaku", action="store_false")
    ap.add_argument("--select", default="meme",
                    choices=["meme", "roundrobin", "all", "trad", "query", "comments"])
    ap.add_argument("--queries", default="",
                    help="comma-separated source_query strata, e.g. 紧绷,绷紧")
    ap.add_argument("--kwset", default="meme",
                    choices=["meme", "trad", "all", "new", "hapi"])
    ap.add_argument("--years", default="", help="e.g. 2024-2026")
    ap.add_argument("--stop-at", type=int, default=None, help="stop discovery at N videos")
    ap.add_argument("--min-pages", type=int, default=4, help="early-stop after N pages with 0 hits")
    args = ap.parse_args()

    # 2026-11-09 finding (哈): the search endpoint returns near-empty page-1
    # results for this SESSDATA account while ANONYMOUS search is fine --
    # measured 2018: 60 new videos / 41 requests with the cookie vs 1,571 new
    # / 101 requests without it, same code path. So discovery runs WITHOUT the
    # login; only the comment endpoints need it (guest cap = 3/video).
    # Do not "fix" this by putting the cookie back into discovery.
    need_login = args.phase in ("crawl", "check", "all")
    sessdata = load_sessdata(C.BILI_COOKIE_FILE) if need_login else None
    print("SESSDATA provided:", bool(sessdata))

    if args.phase == "summary":
        summary()
        return

    b = Bili(sessdata)
    print("wbi mixin ok:", bool(b.mk), "cookies:", sorted(b.s.cookies.get_dict().keys()))

    if args.phase == "check":
        check_cookie(b, n=3)
        return

    yr_lo, yr_hi = C.YEAR_MIN, C.YEAR_MAX
    if args.years:
        a, b_ = args.years.split("-")
        yr_lo, yr_hi = int(a), int(b_)
        print(f"year window: {yr_lo}-{yr_hi}")

    if args.phase in ("discover", "all"):
        if args.kwset == "all":
            kws = C.SEARCH_KEYWORDS + C.SEARCH_KEYWORDS_TRAD
        elif args.kwset == "hapi":
            kws = C.SEARCH_KEYWORDS_DIALECT
        else:
            kws = None
        print("keyword set:", args.kwset, "n =", len(kws) if kws else len(C.SEARCH_KEYWORDS))
        seen = discover(b, pages=args.pages, keywords=kws, stop_at=args.stop_at)
        coverage_table(seen)
    if args.phase == "discover-year":
        if args.kwset == "trad":
            kws = C.SEARCH_KEYWORDS_TRAD
        elif args.kwset == "all":
            kws = C.SEARCH_KEYWORDS + C.SEARCH_KEYWORDS_TRAD
        elif args.kwset == "new":
            kws = C.SEARCH_KEYWORDS_NEW
        elif args.kwset == "hapi":
            kws = C.SEARCH_KEYWORDS_DIALECT
        else:
            kws = None
        print("keyword set:", args.kwset, "n =", len(kws) if kws else len(C.SEARCH_KEYWORDS))
        seen = discover_by_year(b, years=range(yr_lo, yr_hi + 1), keywords=kws)
        coverage_table(seen)
    if args.phase in ("crawl", "all"):
        crawl(b, per_year=args.per_year, max_videos=args.max_videos,
              target_hits=args.target_hits,
              max_comment_pages=args.max_comment_pages,
              skip_danmaku=args.skip_danmaku, select=args.select,
              year_range=(yr_lo, yr_hi), min_pages=args.min_pages,
              queries=[q for q in args.queries.split(",") if q.strip()])
        summary()


if __name__ == "__main__":
    main()
