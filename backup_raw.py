# -*- coding: utf-8 -*-
"""Snapshot every raw data file with a timestamp + sha256 manifest.

Rationale: the harvested comment/danmaku hits cannot be cheaply re-crawled, so
they must never be mutated in place without a recoverable copy. Run this before
and after any harvesting step.

Usage:
    python src/backup_raw.py            # snapshot with timestamp
    python src/backup_raw.py --verify   # show manifest summary only
"""
import os
import sys
import json
import shutil
import hashlib
import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

BACKUP_ROOT = os.path.join(C.RAW, "_backups")
MANIFEST = os.path.join(C.RAW, "_manifest.jsonl")

TRACKED = [
    ("bilibili/hits_comment.jsonl", "irreplaceable"),
    ("bilibili/hits_danmaku.jsonl", "irreplaceable"),
    ("bilibili/danmaku_all.jsonl", "irreplaceable"),
    ("bilibili/videos.jsonl", "recoverable-but-expensive"),
    ("bilibili/_state.json", "irreplaceable (crawl progress)"),
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if a.tag:
        stamp += "_" + a.tag
    dest_dir = os.path.join(BACKUP_ROOT, stamp)

    rows = []
    for rel, kind in TRACKED:
        src = os.path.join(C.RAW, rel)
        if not os.path.exists(src):
            continue
        size = os.path.getsize(src)
        n = count_lines(src)
        h = sha256(src)
        rows.append({"file": rel, "kind": kind, "bytes": size,
                     "lines": n, "sha256": h})
        print(f"  {rel:34} {kind:26} {n:>7} lines  {size/1048576:8.2f} MB  {h[:12]}")
        if not a.verify:
            dst = os.path.join(dest_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

    # append an audit entry
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "snapshot": stamp,
            "kind": "verify" if a.verify else "snapshot",
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
            "files": rows,
        }, ensure_ascii=False) + "\n")

    if not a.verify:
        print(f"\nsnapshot -> {dest_dir}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
