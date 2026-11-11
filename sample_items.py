# -*- coding: utf-8 -*-
"""Random sample N items from a cluster (or a form) of a labeling file.

Usage:
  python src/analyze/sample_items.py --labels data/processed/ha/ha_cdpc_result2.jsonl \
      --key cluster --value 3 --n 6 --seed 42
  python src/analyze/sample_items.py --key form --value 哈气 --n 6 --seed 7

--key  cluster | form      (what to filter on)
--value 3 | 哈气            (the value to match)
--seed  fixed so the draw is reproducible; change it for another draw
--text-only  print only the text (no year prefix)

The label files are the *_cdpc*.jsonl under data/processed/<slug>/; each row has
clean_id / cluster / form / text_norm / ctime_iso.
"""
import os
import re
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(
        C.PROCESSED, f"{C.WORD_SLUG}_cdpc_result2.jsonl"))
    ap.add_argument("--key", choices=["cluster", "form"], default="cluster")
    ap.add_argument("--value", default="3")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-only", action="store_true")
    # 2026-11-09: optional regex on text_norm, e.g. --match "哈气|哈着气"
    # so a cluster can be sampled from its CLEAN core instead of its noisy tail
    ap.add_argument("--match", default="", help="只抽正文匹配此正则的条目")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.labels, encoding="utf-8") if l.strip()]
    val = int(a.value) if a.key == "cluster" else a.value
    pool = [r for r in rows if r.get(a.key) == val]
    if a.match:
        rx = re.compile(a.match)
        pool = [r for r in pool if rx.search(r.get("text_norm") or "")]
    print(f"# {a.labels}")
    print(f"# {a.key}={val}"
          + (f" + match/{a.match}" if a.match else "")
          + f": {len(pool)} items, "
          f"sampling {min(a.n, len(pool))} with seed={a.seed}")
    if not pool:
        sys.exit("nothing matched")
    random.seed(a.seed)
    for r in random.sample(pool, min(a.n, len(pool))):
        if a.text_only:
            print(r["text_norm"])
        else:
            print(f"[{r['ctime_iso'][:10]}] {r['text_norm']}")


if __name__ == "__main__":
    main()
