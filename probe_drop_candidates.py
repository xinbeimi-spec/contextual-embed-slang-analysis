# -*- coding: utf-8 -*-
"""Read-only probe: which items are candidates for REMOVAL?

The user's instruction (2026-10-21) was to clean out two kinds of noise before
building the six-category typology:
  (1) 不相干的 — the character is present but the item is not about 绷 at all:
      homograph collisions (绷弓子 = slingshot), product/game names, ad copy,
      wholesale pasted marketing text;
  (2) 极少量的零散无法归类的 — dictionary senses that occur a handful of times
      and cannot support a category (繃脆/繃硬 intensifier, 繃瓷 crackle glaze).

Nothing is removed here. This script only measures how much each candidate rule
would touch, so the drop rule can be reviewed before it is allowed to act.

Run:  python src/analyze/probe_drop_candidates.py [--file <jsonl>]
"""
import os
import sys
import json
import re
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

RULES = [
    # (name, reason, regex)
    ("homograph_slingshot", "误匹配：绷弓子/绷弓＝弹弓", r"绷弓"),
    ("homograph_crackle", "误匹配：绷瓷/绷磁＝瓷器釉裂", r"绷[瓷磁]"),
    ("intensifier_beng", "极零散：绷脆/绷硬/绷亮＝程度义（方言）", r"绷(?:脆|硬|亮)"),
    ("ad_star_rating", "广告：★☆评分/带货模板", r"[★☆]"),
    ("ad_shop", "广告：电商话术", r"(?:淘宝|旗舰店|下单|链接|优惠券|包邮|亲测|回购)"),
    ("pasted_long_text", "广告/复制粘贴：超长文案", None),   # length rule
    ("only_in_emoticon", "误匹配：绷仅出现在表情包/标签名", None),
]

AD_LEN = 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(C.PROCESSED,
                                                  "beng_typology.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    print(f"{a.file}\nitems: {len(rows)}\n")

    hit = collections.defaultdict(list)
    for r in rows:
        t = r.get("text_norm") or ""
        for name, reason, pat in RULES:
            if pat is None:
                continue
            if re.search(pat, t):
                hit[name].append(r)
        if len(t) > AD_LEN:
            hit["pasted_long_text"].append(r)

    seen = set()
    for name, reason, pat in RULES:
        rs = hit.get(name, [])
        new = [r for r in rs if r.get("clean_id") not in seen]
        print(f"{name:22} n={len(rs):4d} (new={len(new):4d})  {reason}")
        seen |= {r.get("clean_id") for r in rs}

    print("\n" + "=" * 74)
    for name in ("homograph_slingshot", "homograph_crackle", "intensifier_beng"):
        print(f"\n### {name} (n={len(hit.get(name, []))})")
        for r in hit.get(name, [])[:15]:
            print(f"  [{r['category']}|{r.get('form_group')}] "
                  f"{(r.get('text_norm') or '')[:110]}")

    print(f"\n### length > {AD_LEN} chars (n={len(hit.get('pasted_long_text', []))})")
    for r in sorted(hit.get("pasted_long_text", []),
                    key=lambda r: -len(r.get("text_norm") or ""))[:12]:
        t = (r.get("text_norm") or "")
        print(f"  [{r['category']}|len={len(t)}] {t[:100]} ... {t[-40:]}")


if __name__ == "__main__":
    main()
