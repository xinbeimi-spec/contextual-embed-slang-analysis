# -*- coding: utf-8 -*-
"""Read-only probe: what does 紧绷 actually look like in the current corpus?

Motivation: the user wants a new sense category 「紧绷」 = 神情紧绷 / 精神紧绷
(facial and psychological tension). Before adding it we must know:
  1. how many items contain the 紧绷 string at all;
  2. how those items are currently categorised (they now land in 物理义,
     because form_label() maps 紧绷 -> 绷紧 -> 物理义);
  3. which collocations separate "tense face / tense nerves" from the
     physical "taut bandage/string" reading.

Nothing is written. Run:
    python src/analyze/probe_jinbeng.py
"""
import os
import sys
import json
import collections
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

TYPOLOGY = os.path.join(C.PROCESSED, "beng_typology.jsonl")

# collocations that point at the psychological / facial reading
HINTS_PSY = ["神情", "表情", "神色", "神经", "精神", "心情", "情绪", "氛围",
             "气氛", "焦", "紧张", "脸", "眉", "肌肉", "面", "声", "语气",
             "关系", "局势", "局势", "局面", "空气", "心", "身", "僵"]
# collocations that point at the physical / textile reading
HINTS_PHYS = ["皮肤", "绷带", "布", "绢", "弦", "线", "绳", "膜", "弦", "鼓",
              "皮", "衣服", "裤", "丝", "网", "棉", "钢", "橡皮", "弹簧"]


def main():
    rows = [json.loads(l) for l in open(TYPOLOGY, encoding="utf-8") if l.strip()]
    print(f"total items: {len(rows)}")

    hit = [r for r in rows if "紧绷" in (r.get("text_norm") or "")]
    hit2 = [r for r in rows if "绷紧" in (r.get("text_norm") or "")]
    print(f"contain 紧绷: {len(hit)}   contain 绷紧: {len(hit2)}")

    print("\n--- 紧绷 x category (all) ---")
    for k, v in collections.Counter(r["category"] for r in hit).most_common():
        print(f"  {k:10} {v:5d}")
    print("--- 紧绷 x form_group (all) ---")
    for k, v in collections.Counter(r.get("form_group") for r in hit).most_common():
        print(f"  {str(k):10} {v:5d}")
    print("--- 紧绷 x cluster_k7 (all) ---")
    for k, v in collections.Counter(r.get("cluster_k7") for r in hit).most_common():
        print(f"  {str(k):10} {v:5d}")

    ready = [r for r in hit
             if r.get("is_canonical") == 1 and r.get("context_sufficiency") == 1]
    print(f"\nanalysis-ready (canonical & sufficient) with 紧绷: {len(ready)}")

    # left/right context of 紧绷
    left = collections.Counter()
    right = collections.Counter()
    for r in hit:
        t = r.get("text_norm") or ""
        for m in re.finditer("紧绷", t):
            i = m.start()
            left[t[max(0, i - 1):i]] += 1
            right[t[m.end():m.end() + 1]] += 1
    print("\n--- char immediately LEFT of 紧绷 (top 25) ---")
    for k, v in left.most_common(25):
        print(f"  {k!r:6} {v:5d}")
    print("--- char immediately RIGHT of 紧绷 (top 25) ---")
    for k, v in right.most_common(25):
        print(f"  {k!r:6} {v:5d}")

    # 2-char window before 紧绷 (i.e. the noun it modifies)
    pre2 = collections.Counter()
    for r in hit:
        t = r.get("text_norm") or ""
        for m in re.finditer("紧绷", t):
            i = m.start()
            pre2[t[max(0, i - 2):i]] += 1
    print("\n--- 2 chars before 紧绷 (top 30) ---")
    for k, v in pre2.most_common(30):
        print(f"  {k!r:8} {v:5d}")

    print("\n--- all 紧绷 texts (analysis-ready, sorted by length) ---")
    for r in sorted(ready, key=lambda r: len(r.get("text_norm") or "")):
        print(f"[{r.get('cluster_k7')}|{r.get('form_group')}|{r.get('category')}] "
              f"{(r.get('ctime_iso') or '')[:10]}  {r.get('text_norm')}")

    print("\n--- 紧绷 命中但非 analysis-ready 的前 40 条 ---")
    n = 0
    for r in hit:
        if r in ready:
            continue
        print(f"[canonical={r.get('is_canonical')}|ctx={r.get('context_sufficiency')}] "
              f"{(r.get('text_norm') or '')[:120]}")
        n += 1
        if n >= 40:
            break


if __name__ == "__main__":
    main()
