# -*- coding: utf-8 -*-
"""Stage 2: cleaning.

Turns the raw harvest into an analysis-ready table without destroying anything:
the raw files are only ever READ, and every derived column is additive.

Pipeline per record
  text_raw  -> (strip URLs / @mentions / #topics# / [emoticon] tokens /
               collapse repeated chars) -> text_norm
  recompute target offsets on text_norm
  drop if the target character vanished, or if text_norm is shorter than
  MIN_CHARS -- EXCEPT pure 哈-strings of length >= 2 (哈？ is a genuine
  interrogative use, 哈哈/哈哈哈 are the laughter use; 2026-11-09)
  drop if the item is not a use of the character at all (drop_rules.IRRELEVANT:
      homograph / proper name / ad copy) -> data/processed/dropped_items.jsonl
  flag excluded_from_typology when the item is genuine but instantiates a
      dictionary sense too rare to carry a category (drop_rules.TINY_SENSE);
      such rows stay in the corpus but are kept out of typology denominators
  dedupe near-identical items per (platform, send_year) with 2-gram shingles,
      Jaccard >= DUP_THRESH  -> dup_group + is_canonical
  flag context_sufficiency  (can a human actually judge the sense?)
  attach a provisional sense_hint from surface cues (for PILOT STRATIFICATION
      ONLY -- never used as a label)

Outputs
  data/processed/<slug>/<slug>_clean.jsonl  every surviving record, annotated
  data/processed/<slug>/dropped_items.jsonl every removed record + its reason
  results/<slug>/result4/clean_report_by_year.csv
  results/<slug>/result4/clean_report_dedup.csv
"""
import os
import re
import sys
import json
import html
import hashlib
import collections
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from drop_rules import (classify as classify_exclusion, NOISE_TOKENS,
                        hapi_name_videos, hapi_dialect_evidence,
                        meta_about_word)

sys.stdout.reconfigure(encoding="utf-8")

BILI = os.path.join(C.RAW, "bilibili")
OUT = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
DROPPED = os.path.join(C.PROCESSED, "dropped_items.jsonl")

MIN_CHARS = 2
# 2026-11-09 (哈): lowered from 4 to 2 for this word, on purpose.
# 哈's two core uses are SHORT: "哈？" (huh?, 2 chars) and "哈哈/哈哈哈"
# (laughter). A 4-char floor plus the 3-repeat collapse would delete both --
# measured: 47 genuine items, including the bare new-usage words 哈基米/哈皮/
# 哈欠/哈人. Only a lone "哈" (1 char, ambiguous between laugh and sigh) is
# still dropped. Pure-哈 items shorter than 4 chars are flagged
# `short_pure_ha` so the analysis can decide what to do with them.
MIN_CHARS_PURE_HA = 2
DUP_THRESH = 0.85
SHINGLE_N = 2

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_AT = re.compile(r"@[^\s:：,，。]{1,20}")
RE_TOPIC = re.compile(r"#[^#]{1,30}#")
RE_EMOJI = re.compile(r"\[[^\[\]]{1,8}\]")
RE_SPACE = re.compile(r"\s+")
RE_PUNCT = re.compile(r"[\s，。！？、,.!?~～…·:：;；\"'“”‘’()（）\[\]【】"
                      r"<>《》/\\\-—_+=*#@%&|^$`]+")
HA_FORMS = sorted(["哈基米", "马大哈", "打哈哈", "哈哈镜", "哈士奇", "哈喽",
                   "哈啰", "哈罗", "哈皮", "哈批", "哈欠", "哈人", "哈气",
                   "哈腰", "二哈", "哈"], key=len, reverse=True)


def is_ha_only(t):
    """True when the item is made of 哈/哈-forms + punctuation and nothing else."""
    s = RE_PUNCT.sub("", t)
    for f in HA_FORMS:
        s = s.replace(f, "")
    return s == ""

# surface cues -> provisional hint. Order matters: most specific first.
# 2026-11-09: rewritten for 哈 (hā). The generic laughter cue 哈哈哈 must come
# LAST, otherwise every 打哈哈 / 哈哈镜 item would be hinted as laughter.
# 哈？ goes first because it is the one interrogative use that must not be
# swallowed by the laughter default.
CUES = [
    ("N_hajimi", ["哈基米", "hjm"]),
    ("N_haren", ["哈人"]),
    ("D_hello", ["哈喽", "哈啰", "哈罗"]),
    ("D_perfunctory", ["打哈哈"]),
    ("D_mirror", ["哈哈镜"]),
    ("D_yawn", ["哈欠"]),
    ("D_breathe", ["哈气", "哈手"]),
    ("D_bend", ["哈腰"]),
    ("D_careless", ["马大哈"]),
    ("D_erha", ["二哈", "哈士奇"]),
    ("H_laugh", ["哈哈哈", "哈哈"]),
    # 2026-11-09: 哈皮 LAST on purpose. In this corpus it is mostly the
    # streamer 凉哈皮's name (see drop_rules docstring); when the same comment
    # also contains laughter, the laughter reading is the safer hint.
    ("D_hapi", ["哈皮", "哈批"]),
]

# 哈？ as an interrogative: 哈 followed by a question mark that is NOT the tail
# of a 哈-run ("哈哈哈???" is laughter, not "huh?").
RE_Q = re.compile(r"(?<!哈)哈[？?]")

HINT_LABELS = ["M_meta", "Q_question"] + [c[0] for c in CUES]


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sha1(s, n=16):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def normalize(text):
    """Return (text_norm, stats). Never mutates the input."""
    stats = {"url": 0, "at": 0, "topic": 0, "emoji": 0, "repeat": 0}
    t = html.unescape(text or "")
    for pat, key in ((RE_URL, "url"), (RE_AT, "at"), (RE_TOPIC, "topic"),
                     (RE_EMOJI, "emoji")):
        t, n = pat.subn("", t)
        stats[key] = n
    # collapse 3+ repeats of the same char, e.g. 哈哈哈哈哈 -> 哈哈哈
    t2, n = re.subn(r"(.)\1{3,}", r"\1\1\1", t)
    stats["repeat"] = n
    t2 = RE_SPACE.sub(" ", t2).strip()
    return t2, stats


def shingles(s):
    s = s.replace(" ", "")
    if len(s) < SHINGLE_N:
        return {s}
    return {s[i:i + SHINGLE_N] for i in range(len(s) - SHINGLE_N + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def cue_hint(t):
    # 2026-11-09: talking ABOUT the word outranks any usage cue.
    if meta_about_word(t):
        return "M_meta"
    if RE_Q.search(t):
        return "Q_question"
    for label, pats in CUES:
        for p in pats:
            if p in t:
                return label
    return "unknown"


def first_meaningful_index(t):
    """Position of the first 哈 that is NOT inside a 音译/专名 token.

    For 哈 the first occurrence is very often 哈尔滨 / 哈利波特 / 哈士奇;
    masking that position would give the model a vector for the NAME, not for
    the word. The replacement is length-preserving, so every offset computed on
    text_norm stays valid.
    """
    masked = t
    for tok in NOISE_TOKENS:
        if tok in masked:
            masked = masked.replace(tok, "\u0000" * len(tok))
    i = masked.find(C.TARGET_CHAR)
    return i if i >= 0 else t.find(C.TARGET_CHAR)


def sufficiency(t, hint):
    """Can a human judge the sense from this item alone?"""
    n = len(t.replace(" ", ""))
    if n >= 12:
        return 1
    # short but carries an explicit disambiguating collocation
    if hint in HINT_LABELS:
        return 1
    return 0


def export_record(record):
    """Remove account/name metadata and redundant unnormalised text on export.

    The raw harvest remains untouched for auditability. The cleaned tables keep
    the normalized text plus analysis fields, not author hashes or source titles.
    """
    out = dict(record)
    normalized = out.get("text_norm")
    for key in ("author_hash", "user_name", "username", "uname", "thread_title"):
        out.pop(key, None)
    if normalized is not None:
        for key in ("text_raw", "text_clean", "sentence"):
            if key in out:
                out[key] = normalized
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dropped", default=DROPPED)
    # 2026-10-31: cap items per (video, send year). Danmaku-heavy videos can
    # otherwise contribute hundreds of near-duplicate hits and dominate a
    # category. 0 disables the cap.
    ap.add_argument("--max-per-video", type=int, default=25)
    a = ap.parse_args()
    out_path, dropped_path = a.out, a.dropped
    cap_per_video = a.max_per_video

    rows = []
    # 2026-11-09: third source = the supplementary 哈皮/哈批 dialect stratum
    # (crawl_hapi.py). Tagged so the analysis can tell it apart and so the
    # dialect filter below only applies to it.
    for fn, ctype in (("hits_comment.jsonl", "comment"),
                      ("hits_danmaku.jsonl", "danmaku"),
                      ("hits_hapi.jsonl", "comment")):
        for r in read_jsonl(os.path.join(BILI, fn)):
            r.setdefault("content_type", ctype)
            r.setdefault("stratum", "hapi_dialect" if fn == "hits_hapi.jsonl"
                         else "main")
            rows.append(r)
    print(f"raw records: {len(rows)}")

    # 2026-11-09: 哈皮 is a Sichuan-dialect word BUT most of the corpus uses it
    # as the streamer 凉哈皮's name. Decide per video, once, before the loop.
    name_bvids = hapi_name_videos(rows)
    print(f"videos where 哈皮 is the streamer's name: {len(name_bvids)}")

    kept, removed = [], []
    dropped = collections.Counter()
    n_excluded = 0
    seen_ids = set()
    for r in rows:
        raw = r.get("text_raw") or r.get("text_clean") or ""
        t, stats = normalize(raw)
        if C.TARGET_CHAR not in t:
            dropped["target_char_lost"] += 1
            continue
        core = t.replace(" ", "")
        # 2026-11-09: keep 哈？ (interrogative) and 哈哈/哈哈哈 (laughter), and
        # the bare new-usage words 哈基米/哈皮/哈欠, even though they are short.
        ha_only = is_ha_only(core) and len(core) >= MIN_CHARS_PURE_HA
        if len(core) < MIN_CHARS and not ha_only:
            dropped["too_short"] += 1
            continue
        # 2026-11-09: the supplementary dialect batch must show dialect
        # evidence, otherwise it is just more 哈皮-as-a-name noise.
        if r.get("stratum") == "hapi_dialect" and not hapi_dialect_evidence(t):
            removed.append({**r, "text_norm": t,
                            "clean_id": sha1(str(r.get("platform", "")) + "|" +
                                             str(r.get("id") or
                                                 r.get("item_id_hash"))),
                            "drop_stage": "dialect_filter",
                            "drop_reason": "nickname_only: 方言补充批次中无方言证据"
                                           "（哈皮=主播/卡通名，非四川话 哈皮/哈批）"})
            dropped["dialect_noise"] += 1
            continue
        # (1) not a use of the target character at all -> out of the analysis
        # table, but logged and recoverable
        reason_drop, reason_excl = classify_exclusion(
            t, bvid=r.get("thread_bvid"), name_bvids=name_bvids)
        if reason_drop:
            removed.append({**r, "text_norm": t,
                            "clean_id": sha1(str(r.get("platform", "")) + "|" +
                                             str(r.get("id") or
                                                 r.get("item_id_hash"))),
                            "drop_stage": "drop_rules", "drop_reason": reason_drop})
            dropped["irrelevant"] += 1
            continue
        idx = first_meaningful_index(t)
        hint = cue_hint(t)
        rec = dict(r)
        rec.update({
            "text_norm": t,
            "target_start_norm": idx,
            "target_end_norm": idx + len(C.TARGET_CHAR),
            "n_chars_norm": len(t.replace(" ", "")),
            "clean_stats": stats,
            "short_pure_ha": 1 if (ha_only and len(core) < 4) else 0,
            "sense_hint": hint,
            "context_sufficiency": sufficiency(t, hint),
            "dup_group": None,
            "is_canonical": 1,
            # (2) genuine but too rare to carry a category: stays in the
            # corpus, excluded from typology denominators, reported as a
            # footnote instead of being deleted
            "excluded_from_typology": reason_excl,
        })
        if reason_excl:
            n_excluded += 1
        rec["clean_id"] = sha1(rec.get("platform", "") + "|" +
                               str(rec.get("id") or rec.get("item_id_hash")))
        # A crawl that was interrupted between "hits appended" and "state saved"
        # re-fetches the same comment, so the raw file can hold the same item
        # twice. Drop the exact duplicate record here (it is not a new item).
        if rec["clean_id"] in seen_ids:
            dropped["duplicate_record"] += 1
            continue
        seen_ids.add(rec["clean_id"])
        kept.append(rec)
    print(f"kept {len(kept)}  dropped: {dict(dropped)}  "
          f"excluded_from_typology {n_excluded}")

    # ---- dedupe within (platform, send year)
    buckets = collections.defaultdict(list)
    for i, r in enumerate(kept):
        buckets[(r.get("platform"), (r.get("ctime_iso") or "")[:4])].append(i)

    n_groups = 0
    for key, idxs in buckets.items():
        sh = {i: shingles(kept[i]["text_norm"]) for i in idxs}
        parent = {i: i for i in idxs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if abs(sh[i].__len__() - sh[j].__len__()) > 6:
                    continue
                if jaccard(sh[i], sh[j]) >= DUP_THRESH:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        groups = collections.defaultdict(list)
        for i in idxs:
            groups[find(i)].append(i)
        for root, members in groups.items():
            if len(members) > 1:
                n_groups += 1
                gid = sha1(f"{key[0]}|{key[1]}|{min(members)}", 12)
                # canonical = the most informative member
                members.sort(key=lambda k: (-kept[k]["n_chars_norm"],
                                            -(kept[k].get("like_count") or 0)))
                for rank, k in enumerate(members):
                    kept[k]["dup_group"] = gid
                    kept[k]["dup_group_size"] = len(members)
                    kept[k]["is_canonical"] = 1 if rank == 0 else 0

    print(f"duplicate groups formed: {n_groups}")

    # ---- per-video cap (2026-10-31)
    # A single danmaku-heavy video can otherwise decide a category by itself:
    # BV1rv411h7VV contributed 577 hits (556 of them 拉黑), i.e. 6.3% of the
    # whole corpus from ONE thread. Group by (platform, thread, send year) and
    # keep at most --max-per-video items per group, preferring canonical then
    # longer text (deterministic, no RNG). Everything removed is counted and
    # reported, and the raw files keep every row.
    n_capped = 0
    if cap_per_video and cap_per_video > 0:
        byv = collections.defaultdict(list)
        for i, r in enumerate(kept):
            byv[(r.get("platform"), r.get("thread_bvid"),
                 (r.get("ctime_iso") or "")[:4])].append(i)
        keep_mask = [True] * len(kept)
        for _key, idxs in byv.items():
            if len(idxs) <= cap_per_video:
                continue
            ranked = sorted(idxs, key=lambda i: (-kept[i]["is_canonical"],
                                                 -kept[i]["n_chars_norm"], i))
            for i in ranked[cap_per_video:]:
                keep_mask[i] = False
                n_capped += 1
                removed.append({**kept[i], "drop_stage": "per_video_cap",
                                "drop_reason": f"per_video_cap: 同视频同年超过 "
                                               f"{cap_per_video} 条，只保留前 "
                                               f"{cap_per_video} 条（防单视频主导）"})
        kept = [r for i, r in enumerate(kept) if keep_mask[i]]
        print(f"per-video cap={cap_per_video}: removed {n_capped} items")

    os.makedirs(C.PROCESSED, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(export_record(r), ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    # removed items are preserved here (never deleted from the raw harvest)
    tmp = dropped_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in removed:
            f.write(json.dumps(export_record(r), ensure_ascii=False) + "\n")
    os.replace(tmp, dropped_path)

    # ---- reports
    import csv
    by_year = collections.defaultdict(lambda: collections.Counter())
    for r in kept:
        y = (r.get("ctime_iso") or "")[:4]
        by_year[y]["total"] += 1
        if r["is_canonical"]:
            by_year[y]["canonical"] += 1
        if r["context_sufficiency"]:
            by_year[y]["judgeable"] += 1
        if r["is_canonical"] and r["context_sufficiency"]:
            by_year[y]["analysis_ready"] += 1
        by_year[y]["_" + r["sense_hint"]] += 1

    hints = HINT_LABELS + ["unknown"]
    p1 = os.path.join(C.TABLES, "clean_report_by_year.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["year", "total", "canonical", "judgeable", "analysis_ready",
                    "dupe_groups_total"] + [f"hint_{h}" for h in hints])
        for y in sorted(by_year):
            if not y:
                continue
            c = by_year[y]
            w.writerow([y, c["total"], c["canonical"], c["judgeable"],
                        c["analysis_ready"],
                        sum(1 for r in kept if (r.get("ctime_iso") or "")[:4] == y
                            and r.get("dup_group"))]
                       + [c["_" + h] for h in hints])

    p2 = os.path.join(C.TABLES, "clean_report_dedup.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["raw_records", len(rows)])
        w.writerow(["kept", len(kept)])
        w.writerow(["dropped_target_lost", dropped["target_char_lost"]])
        w.writerow(["dropped_too_short", dropped["too_short"]])
        w.writerow(["dropped_irrelevant", dropped["irrelevant"]])
        w.writerow(["excluded_from_typology", n_excluded])
        w.writerow(["dup_groups", n_groups])
        w.writerow(["non_canonical", sum(1 for r in kept if not r["is_canonical"])])
        w.writerow(["judgeable", sum(1 for r in kept if r["context_sufficiency"])])
        w.writerow(["analysis_ready (canonical & judgeable)",
                    sum(1 for r in kept
                        if r["is_canonical"] and r["context_sufficiency"])])

    print(f"\n-> {out_path}")
    print(f"-> {dropped_path}")
    print(f"-> {p1}")
    print(f"-> {p2}")

    print("\n=== by year ===")
    print(f"{'year':6}{'total':>7}{'canon':>7}{'judge':>7}{'ready':>7}")
    for y in sorted(k for k in by_year if k):
        c = by_year[y]
        print(f"{y:6}{c['total']:>7}{c['canonical']:>7}{c['judgeable']:>7}"
              f"{c['analysis_ready']:>7}")

    print("\n=== sense hints (provisional, canonical only) ===")
    ch = collections.Counter(r["sense_hint"] for r in kept if r["is_canonical"])
    for h, n in ch.most_common():
        print(f"  {h:18} {n:5d}")


if __name__ == "__main__":
    main()
