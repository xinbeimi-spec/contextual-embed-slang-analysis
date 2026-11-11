# -*- coding: utf-8 -*-
"""Supplementary crawl for the 哈皮/哈批 stratum (= 傻/憨, Sichuan dialect).

Why a separate script: the main harvest is dominated by the streamer 凉哈皮 and
by ordinary laughter, so a normal `--select comments` run would add hundreds of
"哈哈哈" records for every dialect item. This crawl keeps ONLY comments that
contain 哈皮 or 哈批, and writes them to a separate file:

    data/raw/ha/bilibili/hits_hapi.jsonl      (stratum = "hapi_dialect")

clean_corpus.py reads that file as a third source and tags the rows, so the
main corpus (hits_comment.jsonl) is not altered by this supplement.

Usage:
  python src/crawl/crawl_hapi.py --max-videos 10 --max-pages 6     # probe
  python src/crawl/crawl_hapi.py --max-videos 150 --max-pages 10   # real run
"""
import os
import sys
import json
import time
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import (Bili, load_sessdata, sha1, iso, year_of, make_hit_text,
                        read_jsonl, jappend, VERSION)

sys.stdout.reconfigure(encoding="utf-8")

BILI = os.path.join(C.RAW, "bilibili")
F_VIDEOS = os.path.join(BILI, "videos.jsonl")
F_MAIN = os.path.join(BILI, "hits_comment.jsonl")
F_OUT = os.path.join(BILI, "hits_hapi.jsonl")
DIALECT_Q = set(C.SEARCH_KEYWORDS_DIALECT)
KEEP = ("哈皮", "哈批")


def pick(seen, limit):
    cand = []
    for v in seen.values():
        title = v.get("title") or ""
        if "凉哈皮" in title:          # the streamer's channel: not dialect
            continue
        if (v.get("source_query") in DIALECT_Q
                or any(k in title for k in KEEP)):
            cand.append(v)
    # title mentions of the word first (the word is the topic), then volume
    cand.sort(key=lambda x: (0 if any(k in (x.get("title") or "") for k in KEEP)
                             else 1, -(x.get("video_review") or 0)))
    print(f"  dialect candidates: {len(cand)}  top has "
          f"{cand[0].get('video_review') if cand else 0} comments")
    return cand[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=150)
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--min-pages", type=int, default=2)
    a = ap.parse_args()

    seen = {r["bvid"]: r for r in read_jsonl(F_VIDEOS)}
    print(f"pool: {len(seen)} videos")

    done = set()
    if os.path.exists(F_OUT):
        done = {r["thread_bvid"] for r in read_jsonl(F_OUT)}
        print(f"resume: {len(done)} videos already harvested in this stratum")

    main_ids = set()
    if os.path.exists(F_MAIN):
        main_ids = {r["id"] for r in read_jsonl(F_MAIN)}

    todo = [v for v in pick(seen, 10 ** 9) if v["bvid"] not in done][:a.max_videos]
    print(f"to crawl: {len(todo)} videos (max-pages={a.max_pages})")

    b = Bili(load_sessdata(C.BILI_COOKIE_FILE))
    print("SESSDATA provided:", bool(b.sessdata))

    n_new = 0
    t0 = time.time()
    for i, v in enumerate(todo, 1):
        aid, bvid = v["aid"], v["bvid"]
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
            "stratum": "hapi_dialect",
        }
        rows = []
        offset = ""
        seen_comments = 0
        for cp in range(a.max_pages):
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
                seen_comments += 1
                if not any(k in msg for k in KEEP):
                    continue
                r = make_hit_text(msg, "comment", base)
                if r is None:
                    continue
                r.update(base)
                r["item_id_hash"] = sha1("cm|" + bvid + "|" + str(c.get("rpid")))
                r["id"] = r["item_id_hash"]
                if r["id"] in main_ids:
                    continue
                r["parent_id_hash"] = sha1("rpid|" + str(parent)) if parent else None
                r["author_hash"] = sha1("mid|" + str(c.get("mid"))) if c.get("mid") else None
                r["ctime_epoch"] = c.get("ctime")
                r["ctime_iso"] = iso(c.get("ctime") or 0)
                r["time_precision"] = "second"
                r["like_count"] = c.get("like")
                r["reply_count"] = c.get("rcount")
                rows.append(r)
            cur = cd.get("cursor") or {}
            pr = cur.get("pagination_reply") or {}
            nxt = pr.get("next") or pr.get("next_offset")
            if not nxt or cur.get("is_end"):
                break
            if (cp + 1) >= a.min_pages and not rows:
                break
            offset = nxt
            b.sleep()
        b.sleep()
        jappend(F_OUT, rows)
        n_new += len(rows)
        print(f"[{i}/{len(todo)}] {bvid} y={v['year']} cm={seen_comments} "
              f"hapi={len(rows)} | total={n_new}")

    print(f"\nDONE. new 哈皮/哈批 records={n_new}, "
          f"elapsed={(time.time()-t0)/60:.1f}min -> {F_OUT}")


if __name__ == "__main__":
    main()
