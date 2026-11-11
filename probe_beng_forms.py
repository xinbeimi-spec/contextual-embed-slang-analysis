# -*- coding: utf-8 -*-
"""Report which 绷-forms actually occur, per category (design probe).

The user rejected the context-word version of the word-frequency figure: what
they want is the frequency of the TARGET-WORD FORMS themselves — 绷住, 紧绷,
绷带, 绷一下, 绷紧 … — inside each semantic cluster. This probe lists every
candidate form with counts so the canonical form list can be built from data
instead of guessed.

Candidate generation per occurrence of 绷:
  * a curated list of known forms matched longest-first (绷不住 / 绷住了 /
    绷住 / 紧绷 / 绷紧 / 绷带 / 绷一下 / 绷着脸 / 难绷 / 绷直 / 绷起来 …)
  * otherwise 绷 + the next 1-2 characters (that is where 绷一下 comes from)
  * for 紧绷 the 紧 before 绷 is included

Writes logs/forms_probe.txt and prints a summary. Read-only.
Run: python src/analyze/probe_beng_forms.py
"""
import os
import re
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join(C.PROCESSED, "beng_cdpc.jsonl")
OUT = os.path.join(C.LOGS, "forms_probe.txt")
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]

# curated, longest-first; the label is what we count as one "form"
KNOWN = [
    ("绷不住了", r"绷不住了"),
    ("绷不住", r"绷不住"),
    ("绷得住", r"绷得住"),
    ("绷住了", r"绷住了"),
    ("绷住", r"绷住"),
    ("紧绷", r"紧绷"),
    ("绷紧", r"绷紧"),
    ("绷带", r"绷带"),
    ("绷直", r"绷直"),
    ("绷一下", r"绷一下"),
    ("绷着", r"绷着"),
    ("绷起来", r"绷起来"),
    ("绷起来", r"绷起来"),
    ("难绷", r"难绷"),
    ("绷坏", r"绷坏"),
    ("绷铁", r"绷铁"),
    ("绷炸弹", r"绷炸弹"),
    ("绷断", r"绷断"),
    ("绷开", r"绷开"),
    ("绷线", r"绷线"),
    ("绷平", r"绷平"),
    ("绷脆", r"绷脆"),
    ("绷硬", r"绷硬"),
]
_KNOWN = [(lab, re.compile(p)) for lab, p in KNOWN]
RE_TARGET = re.compile(C.TARGET_CHAR)


def forms_of(text):
    """All 绷-forms in one text, ordered by position."""
    t = text or ""
    out = []
    i = 0
    while True:
        j = t.find(C.TARGET_CHAR, i)
        if j < 0:
            break
        lab = None
        for name, rx in _KNOWN:
            m = rx.match(t, j)
            if m:
                lab = name
                break
        if lab is None:
            # fall back to a 绷 + following 1-2 char window, trimmed of
            # punctuation and particles so "绷，"/"绷了" do not dominate
            w = t[j:j + 3]
            w = re.sub(r"[，。！？、：；\s~…—-]+$", "", w)
            lab = w if len(w) >= 2 else C.TARGET_CHAR
        out.append(lab)
        i = j + 1
    return out


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if not r.get("excluded_from_typology")]
    print(f"items: {len(rows)}")

    raw = collections.defaultdict(collections.Counter)     # category -> form
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["cdpc_category"]].append(r.get("text_norm") or "")
    for c, texts in by_cat.items():
        for t in texts:
            raw[c].update(forms_of(t))

    L = [f"items: {len(rows)}", ""]
    for c in CAT_ORDER:
        cnt = raw.get(c) or collections.Counter()
        L.append(f"=== {c}  n_items={len(by_cat.get(c, []))}  "
                 f"distinct_forms={len(cnt)} ===")
        for f, k in cnt.most_common(40):
            L.append(f"   {f:10} {k:5d}")
        L.append("")
    txt = "\n".join(L)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt[:4000])
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
