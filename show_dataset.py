# -*- coding: utf-8 -*-
"""Print the analysis-ready dataset location, schema and a real sample row.

Purpose: the user needs to know exactly which folder to point at and which
fields exist before doing any plotting.
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
EMB = C.EMB_DIR

rows = [json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip()]
print("=" * 78)
print("DATASET FOLDER :", C.PROCESSED)
print("MAIN TABLE     :", CLEAN)
print("ROWS           :", len(rows))
print("=" * 78)

# field inventory
keys = collections.Counter()
for r in rows:
    for k in r:
        keys[k] += 1
print("\n--- FIELDS (name : present in N rows) ---")
for k, n in sorted(keys.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {k:26} {n:6d}")

print("\n--- ONE FULL SAMPLE ROW ---")
# pick a mid-length, judgeable, canonical, comment record for readability
cand = [r for r in rows
        if r.get("is_canonical") == 1
        and r.get("context_sufficiency") == 1
        and 20 <= (r.get("n_chars_norm") or 0) <= 60]
sample = cand[1234] if len(cand) > 1234 else cand[0]
print(json.dumps(sample, ensure_ascii=False, indent=2))

print("\n--- 3 more short examples (id / date / text / hint) ---")
for r in rows[:3]:
    print(f"  {r.get('clean_id')}  {r.get('ctime_iso','')[:10]}  "
          f"[{r.get('sense_hint')}]  {r.get('text_norm','')[:50]}")
