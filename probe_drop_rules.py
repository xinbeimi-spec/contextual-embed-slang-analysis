# -*- coding: utf-8 -*-
"""Review probe for the cleaning rules: what exactly would be dropped / excluded?

Prints to stdout with UTF-8 and ALSO writes the same listing to
logs/drop_review.txt so the review can be read without console recoding issues.

Run:  python src/analyze/probe_drop_rules.py [--file <jsonl>]
"""
import os
import sys
import json
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "clean"))
import config as C
from drop_rules import classify

sys.stdout.reconfigure(encoding="utf-8")

OUT_TXT = os.path.join(C.LOGS, "drop_review.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(C.PROCESSED,
                                                  "beng_typology.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]

    drop = collections.defaultdict(list)
    excl = collections.defaultdict(list)
    keep = 0
    for r in rows:
        d, x = classify(r.get("text_norm"))
        if d:
            drop[d].append(r)
        elif x:
            excl[x].append(r)
        else:
            keep += 1

    L = []
    L.append(f"file: {a.file}")
    L.append(f"items: {len(rows)}  keep: {keep}  "
             f"drop: {sum(len(v) for v in drop.values())}  "
             f"excluded_from_typology: {sum(len(v) for v in excl.values())}")
    L.append("")
    for title, d in (("DROP (irrelevant)", drop),
                     ("EXCLUDE FROM TYPOLOGY (tiny senses)", excl)):
        L.append(f"===== {title} =====")
        for reason, rs in d.items():
            L.append(f"  {reason}   n={len(rs)}")
        L.append("")
        for reason, rs in d.items():
            L.append(f"--- {reason}")
            for r in rs:
                L.append(f"    [{r.get('category')}|{r.get('form_group')}|"
                         f"{(r.get('ctime_iso') or '')[:7]}] "
                         f"{(r.get('text_norm') or '')[:150]}")
            L.append("")

    txt = "\n".join(L)
    open(OUT_TXT, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n-> {OUT_TXT}")


if __name__ == "__main__":
    main()
