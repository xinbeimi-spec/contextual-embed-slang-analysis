# -*- coding: utf-8 -*-
"""Read-only probe: candidate cue families for a possible 「紧绷」 sense.

The user asked for a category meaning 神情紧绷 / 精神紧绷 (tense face, tense
nerves). The literal string 紧绷 is not enough: the corpus also expresses the
same sense with 绷着脸 / 绷着神经 / 脸绷得 / 整个人绷着. This script measures
how much each candidate cue family would pull in, so the category definition
can be chosen from data rather than guessed.

Nothing is written. Run:
    python src/analyze/probe_jinbeng_cues.py
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

SUBJECT_PSY = ("(?:神情|表情|神色|神经|精神|心情|情绪|氛围|气氛|空气|关系|局面|"
               "状态|脸上|脸|面具|笑容|肌肉|身体|全身|整个人|身心|眉|语气|"
               "声音|嗓音)")
CANDIDATES = [
    ("紧绷", r"紧绷"),
    ("紧+绷(任意)", r"紧.{0,2}绷"),
    ("绷+紧", r"绷.{0,1}紧"),
    ("神情/表情等+绷", SUBJECT_PSY + r".{0,4}绷"),
    ("绷+神情/表情等", r"绷.{0,4}" + SUBJECT_PSY),
    ("绷着(脸/表情/股)", r"绷着(?:个?脸|着?表情|着?股)"),
    ("绷得+紧", r"绷得.{0,2}紧"),
    ("绷+太紧/的紧", r"绷.{0,1}(?:太|很|好|超)紧"),
    ("神经/精神+绷", r"(?:神经|精神|脑子|心态).{0,3}绷"),
    ("绷+神经/精神", r"绷.{0,3}(?:神经|精神)"),
    ("脸+绷", r"脸.{0,3}绷"),
    ("绷+脸", r"绷.{0,2}脸"),
]


def main():
    rows = [json.loads(l) for l in open(TYPOLOGY, encoding="utf-8") if l.strip()]
    print(f"total items: {len(rows)}\n")

    seen_any = set()
    for name, pat in CANDIDATES:
        rx = re.compile(pat)
        hit = [r for r in rows if rx.search(r.get("text_norm") or "")]
        ids = {r.get("clean_id") for r in hit}
        print(f"{name:22} {len(hit):5d} new={len(ids - seen_any):5d}  "
              f"cats={dict(collections.Counter(r['category'] for r in hit))}")
        seen_any |= ids

    print("\n" + "=" * 78)
    print("samples for the psychologically-oriented families")
    print("=" * 78)
    keys = ("神情/表情等+绷", "绷着(脸/表情/股)", "绷+神情/表情等",
            "神经/精神+绷", "脸+绷", "绷+脸", "绷+紧", "绷+神经/精神")
    for name, pat in CANDIDATES:
        if name not in keys:
            continue
        rx = re.compile(pat)
        hit = [r for r in rows if rx.search(r.get("text_norm") or "")]
        print(f"\n### {name}  n={len(hit)}")
        for r in hit[:25]:
            print(f"  [{r.get('cluster_k7')}|{r.get('form_group')}|{r['category']}] "
                  f"{(r.get('text_norm') or '')[:90]}")


if __name__ == "__main__":
    main()
