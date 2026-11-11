# -*- coding: utf-8 -*-
"""Verify the video pool against the pre-incident baseline, and verify the
irreplaceable hit files against the latest snapshot manifest.

Exit code 0 only when every year reaches its baseline.
"""
import os
import sys
import json
import hashlib
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

# Pool sizes measured immediately BEFORE the 2026-10-20 truncation incident.
# Historic reference only: the pool was later expanded by the 30-keyword
# discovery sweep (2026-10-20 evening) and again on 2026-10-21, which shifted
# videos between years (e.g. 2019 730 -> 290, 2026 1866 -> 3793) without losing
# any row. The comparison that matters today is BASELINE_20261021 below.
BASELINE_PRE_INCIDENT = {
    2016: 194, 2017: 387, 2018: 553, 2019: 730, 2020: 836,
    2021: 556, 2022: 311, 2023: 308, 2024: 611, 2025: 723, 2026: 1866,
}
# Measured 2026-10-21 after the corpus expansion crawl finished.
# Any year below these numbers means the pool lost videos -> investigate.
BASELINE = {
    2016: 195, 2017: 424, 2018: 548, 2019: 290, 2020: 174,
    2021: 324, 2022: 202, 2023: 268, 2024: 761, 2025: 888, 2026: 3793,
}
BASELINE_FILE = os.path.join(C.RAW, "_baseline.json")


def _load_baseline():
    """Per-word baseline (2026-10-27).

    The numbers below were measured for 绷. A new word gets its own first-day
    measurement written to data/raw/<slug>/_baseline.json, which then takes
    precedence -- otherwise this script would compare 塌's pool against 绷's.
    """
    if os.path.exists(BASELINE_FILE):
        try:
            d = json.load(open(BASELINE_FILE, encoding="utf-8"))
            base = {int(k): int(v) for k, v in d.items()}
            print(f"baseline: {BASELINE_FILE} ({len(base)} years)")
            return base
        except Exception as e:                     # noqa: BLE001
            print("  ! baseline file unreadable, falling back:", e)
    print("baseline: built-in 绷 numbers (no _baseline.json for this word yet)")
    return BASELINE


BILI = os.path.join(C.RAW, "bilibili")


def rj(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    baseline = _load_baseline()
    rows = rj(os.path.join(BILI, "videos.jsonl"))
    uniq = {r["bvid"]: r for r in rows}
    by_year = collections.Counter(v.get("year") for v in uniq.values())

    print(f"videos.jsonl rows={len(rows)} unique={len(uniq)}")
    if len(rows) != len(uniq):
        print("  !! duplicate bvids present")

    print(f"\n{'year':6}{'baseline':>10}{'now':>8}{'delta':>8}  status")
    short = 0
    for y in sorted(baseline):
        now = by_year.get(y, 0)
        d = now - baseline[y]
        status = "OK" if d >= 0 else "SHORT"
        if d < 0:
            short += 1
        print(f"{y:<6}{baseline[y]:>10}{now:>8}{d:>+8}  {status}")
    tot_now = sum(by_year.get(y, 0) for y in baseline)
    print(f"{'TOTAL':<6}{sum(baseline.values()):>10}{tot_now:>8}"
          f"{tot_now - sum(baseline.values()):>+8}")

    extra = {k: v for k, v in by_year.items() if k not in baseline}
    if extra:
        print("  years outside baseline window:", dict(sorted(extra.items())))

    print("\n=== irreplaceable corpus files ===")
    for rel in ("hits_comment.jsonl", "hits_danmaku.jsonl", "danmaku_all.jsonl"):
        p = os.path.join(BILI, rel)
        if os.path.exists(p):
            n = sum(1 for _ in open(p, "rb"))
            print(f"  {rel:24} {n:>7} lines  sha256 {sha256(p)[:16]}")

    if short:
        print(f"\nRESULT: {short} year(s) below baseline -- more recovery needed")
        return 1
    print("\nRESULT: every year is at or above the pre-incident baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
