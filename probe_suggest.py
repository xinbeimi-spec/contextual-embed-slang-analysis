# -*- coding: utf-8 -*-
"""Harvest real current 绷-compounds from bilibili's search-suggestion API.

Guessing neologisms is unreliable; the suggest endpoint surfaces what people
actually type right now, which is exactly what a 2024-2026 sweep needs.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from crawl_bili import Bili

sys.stdout.reconfigure(encoding="utf-8")

b = Bili()

SEEDS = ["绷", "绷绷", "绷不", "绷不住", "绷住", "绷了", "难绷", "绷一", "绷死",
         "绷坏", "绷人", "绷味", "绷系", "绷弹", "绷炸", "绷哥", "绷铁"]

found = {}

print("=== suggest API (s.search.bilibili.com/main/suggest) ===")
for seed in SEEDS:
    try:
        r = b.s.get("https://s.search.bilibili.com/main/suggest",
                    params={"term": seed, "main_ver": "v1", "highlight": ""},
                    timeout=20)
        j = r.json()
        tags = ((j.get("result") or {}).get("tag") or [])
        names = [t.get("name") for t in tags if t.get("name")]
        vals = [t.get("value") for t in tags if t.get("value")]
        for n in names + vals:
            if n:
                found[n] = found.get(n, 0) + 1
        print(f"  {seed:8} -> {len(names)} terms: {names[:8]}")
    except Exception as e:
        print(f"  {seed:8} EXC {type(e).__name__}: {e}")
    time.sleep(0.6)

print("\n=== suggest API (api.bilibili.com variant) ===")
for seed in ["绷", "绷绷", "绷不住", "难绷"]:
    try:
        r = b.s.get("https://api.bilibili.com/x/web-interface/search/suggest",
                    params={"term": seed, "type": "video"}, timeout=20)
        j = r.json()
        ts = [t.get("value") for t in (j.get("data", {}).get("tag") or [])]
        print(f"  {seed:8} -> {ts[:8]}")
        for n in ts:
            if n:
                found[n] = found.get(n, 0) + 1
    except Exception as e:
        print(f"  {seed:8} EXC {type(e).__name__}: {e}")
    time.sleep(0.6)

print("\n=== unique terms harvested ===")
beng = sorted([t for t in found if "绷" in t], key=lambda x: -len(x))
print(f"total with 绷: {len(beng)}")
for t in beng:
    print("   ", t)
