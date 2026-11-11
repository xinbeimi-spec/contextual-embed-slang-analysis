# -*- coding: utf-8 -*-
"""Pick REPRESENTATIVE sentences for each of the five categories.

"Representative" is operationalised, not hand-picked:
  * restrict to judgeable, canonically-kept items with a readable length
    (10-70 chars) so the quote is self-contained;
  * rank by cosine to the category centroid (closest = most typical);
  * enforce diversity: a candidate is skipped if it is near-duplicate of an
    already-selected one (pairwise cosine >= 0.85), so the list shows the range
    of the category instead of five paraphrases.

For 非义项用法 the category is heterogeneous by definition, so it is broken into
subtypes (mentions / verbatim repetition / spelling variants / homograph noise)
and examples are drawn from each.

Outputs
  results/final_5cat/tables/typology_representative.csv
"""
import os
import re
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

ROOT = C.EMB_MASK_DIR
TYPOLOGY = os.path.join(C.PROCESSED, "beng_typology.jsonl")
LAYERS = (9, 10, 11, 12)
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
N_PER = 8
DIVERSITY = 0.85
MIN_LEN, MAX_LEN = 10, 70


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_repr():
    index = rj(os.path.join(ROOT, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(ROOT, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    Z = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    Z = Z - (Z @ Vt[:1].T) @ Vt[:1]
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12), ids


def subtype_non_sense(t):
    """Rough sub-typing of the 'mention / noise' category, for display only."""
    if re.search(r"我是[绷老]|老绷|绷意|绷学|绷住哲学|绷罗大陆|绷神|绷马野郎", t):
        return "元语言玩梗（把该字当作被谈论的对象）"
    # verbatim repetition: a short chunk repeated many times
    if re.search(r"(.{2,6})\1{3,}", t):
        return "逐字复读（碎片化刷屏）"
    if re.search(r"绷往|绷弓子|绷硬|绷脆|绷亮|绷瓷|绷脸", t):
        return "异形/错别字/同形误匹配"
    if len(t) > 60:
        return "其他噪声（歌词、长文夹杂等）"
    return "其他噪声"


def main():
    Z, ids = load_repr()
    rows = {r["clean_id"]: r for r in rj(TYPOLOGY)}
    cat = np.array([rows[i]["category"] for i in ids])

    import csv
    # NOTE: written to the shared tables dir; the frozen five-category
    # deliverable in results/final_5cat/ must not be overwritten any more.
    out_dir = C.TABLES
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "typology_representative.csv")

    recs = []
    for c in CAT_ORDER:
        idx = [i for i in np.where(cat == c)[0]
               if rows[ids[i]].get("is_canonical") == 1
               and rows[ids[i]].get("context_sufficiency") == 1
               and MIN_LEN <= len(rows[ids[i]].get("text_norm") or "") <= MAX_LEN]
        if not idx:
            continue
        cen = Z[idx].mean(axis=0)
        cen /= np.linalg.norm(cen) + 1e-12
        sims = Z[idx] @ cen
        order = np.argsort(-sims)

        if c == "非义项用法":
            groups = collections.defaultdict(list)
            for k in order:
                i = idx[k]
                groups[subtype_non_sense(rows[ids[i]]["text_norm"])].append(
                    (i, float(sims[k])))
            print(f"\n{'='*80}\n【{c}】 候选 {len(idx)} 条（按子类抽样）")
            for sub, lst in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                print(f"\n  ── {sub}（该类共 {len(lst)} 条）")
                taken = []
                n = 0
                for i, s in lst:
                    if any(Z[i] @ Z[j] > DIVERSITY for j in taken):
                        continue
                    r = rows[ids[i]]
                    print(f"     [{s:.3f}] {r['ctime_iso'][:10]}  "
                          f"{r['text_norm'][:66]}")
                    recs.append({"category": c, "subtype": sub,
                                 "score_centroid_cos": round(s, 4),
                                 "clean_id": r["clean_id"],
                                 "date": r["ctime_iso"][:10],
                                 "text": r["text_norm"]})
                    taken.append(i)
                    n += 1
                    if n >= 3:
                        break
            continue

        print(f"\n{'='*80}\n【{c}】 候选 {len(idx)} 条，"
              f"取最贴近质心且互不重复的 {N_PER} 条")
        taken, n = [], 0
        for k in order:
            i = idx[k]
            if any(Z[i] @ Z[j] > DIVERSITY for j in taken):
                continue
            r = rows[ids[i]]
            s = float(sims[k])
            flag = "★" if r.get("like_count") and r["like_count"] > 500 else " "
            print(f"  {flag}[{s:.3f}] {r['ctime_iso'][:10]}  "
                  f"{r['text_norm'][:70]}")
            recs.append({"category": c, "subtype": "",
                         "score_centroid_cos": round(s, 4),
                         "clean_id": r["clean_id"],
                         "date": r["ctime_iso"][:10],
                         "text": r["text_norm"]})
            taken.append(i)
            n += 1
            if n >= N_PER:
                break

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["category", "subtype",
                                          "score_centroid_cos", "clean_id",
                                          "date", "text"])
        w.writeheader()
        w.writerows(recs)
    print(f"\n-> {out_csv}  ({len(recs)} rows)")


if __name__ == "__main__":
    main()
