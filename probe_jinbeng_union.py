# -*- coding: utf-8 -*-
"""Read-only probe v2: dump the union of narrow 紧绷 candidates for manual reading.

The candidate rule below is deliberately NARROW (it is meant to be reviewed by
hand, not trusted): literal 紧绷 / 绷得紧 / 绷着(脸|表情|神经) / psychological
subject + 绷. 难绷 is excluded because 表情难绷 is the evaluative hotword, not
the tense sense.

Writes nothing to data. Prints to stdout (redirect to a log file if needed).
"""
import os
import sys
import json
import re
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

TYPOLOGY = os.path.join(C.PROCESSED, "beng_typology.jsonl")

PSY_SUBJ = ("神情|表情|神色|神经|精神|心情|情绪|心态|氛围|气氛|空气|关系|局面|"
            "状态|脸上|脸|笑容|肌肉|身体|全身|整个人|身心|眉头|语气")
PATTERNS = [
    ("A_紧绷字面", r"紧绷"),
    ("B_绷得紧", r"绷(?:得|的|太|很|超)?.{0,1}紧"),
    ("C_绷着脸/表情/神经", r"绷着.{0,2}(?:脸|表情|神经|面|股)"),
    ("D_心理主语+绷", rf"(?:{PSY_SUBJ}).{{0,3}}绷"),
    ("E_绷+心理主语", rf"绷.{{0,3}}(?:{PSY_SUBJ})"),
]


def main():
    rows = [json.loads(l) for l in open(TYPOLOGY, encoding="utf-8") if l.strip()]
    print(f"total items: {len(rows)}\n")

    union = {}
    for name, pat in PATTERNS:
        rx = re.compile(pat)
        hit = [r for r in rows if rx.search(r.get("text_norm") or "")]
        print(f"{name:20} n={len(hit):4d}  "
              f"cats={dict(collections.Counter(r['category'] for r in hit))}")
        for r in hit:
            union.setdefault(r.get("clean_id"), (r, []))[1].append(name)

    print(f"\nunion n={len(union)}")
    print("=" * 78)
    # sort: items hit by more than one family first, then by year
    ordered = sorted(union.values(),
                     key=lambda t: (-len(t[1]),
                                    (t[0].get("ctime_iso") or "9999")))
    for r, names in ordered:
        t = (r.get("text_norm") or "").replace("\n", " ")
        print(f"[{r.get('cluster_k7')}|{r.get('form_group')}|{r['category']}|"
              f"{(r.get('ctime_iso') or '')[:7]}|{'+'.join(n.split('_')[0] for n in names)}]")
        print(f"    {t}")
    print(f"\nunion n={len(union)}")


if __name__ == "__main__":
    main()
