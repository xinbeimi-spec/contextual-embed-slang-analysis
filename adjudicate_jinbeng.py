# -*- coding: utf-8 -*-
"""Adjudicate the 紧绷 sense: candidate report -> human decisions -> jsonl.

Workflow (deliberately two-step, so the human decision is separable from the
rule that generated the candidates)

  1) REPORT
       python src/analyze/adjudicate_jinbeng.py --report
     writes logs/jinbeng_candidates.txt: every item whose text fires a 紧绷 cue
     (see jinbeng_cue.CUES), numbered, with year / cluster / form / full text.
     Numbering is a function of the FINAL corpus, so the report must be
     regenerated after any re-crawl or re-clean.

  2) DECIDE
     src/analyze/jinbeng_decisions.tsv holds the review, three tab-separated
     columns:
         decision <TAB> indices <TAB> reason
     e.g.
         keep    3,7,12  心理紧绷：神经/精神框架 + 紧张是谓语焦点
         reject  1,2,4   身体部位/皮肤的物理紧绷，无语用心理框架
     Every candidate must be covered by exactly one line; `--commit` refuses to
     write while any candidate is undecided or any index is unknown, so the
     decision table cannot silently lag behind the data.

  3) COMMIT
       python src/analyze/adjudicate_jinbeng.py --commit
     writes data/processed/jinbeng_adjudication.jsonl (one record per
     candidate: clean_id, decision, cue, reason, text, year) and prints the
     rule-vs-decision reconciliation that build_typology.py will also print.

Why adjudicate at all: the wide cue rule fires on many 难绷 / 绷不住 items
(「一脸难绷」「这表情真绷不住」), which are evaluative or failure-to-hold, not
tension. Per-item review is the only way to keep 评价义 and 忍不住义 clean.
"""
import os
import sys
import csv
import json
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from jinbeng_cue import cues_of, ADJ

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
FORMS = os.path.join(C.PROCESSED, "beng_forms.jsonl")
REPORT = os.path.join(C.LOGS, "jinbeng_candidates.txt")
DECISIONS = os.path.join(C.SRC, "analyze", "jinbeng_decisions.tsv")


def load_candidates():
    rows = [json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip()]
    # enrich with the clustering assignment when it already exists, so the
    # reviewer can see which cluster an item currently sits in
    extra = {}
    if os.path.exists(FORMS):
        for r in (json.loads(l) for l in open(FORMS, encoding="utf-8")
                  if l.strip()):
            extra[r.get("clean_id")] = r
    cand = []
    for r in rows:
        t = r.get("text_norm") or ""
        cs = cues_of(t)
        if cs:
            e = extra.get(r.get("clean_id")) or {}
            r["form_group"] = e.get("form_group")
            r["cluster_k7"] = e.get("cluster_k7")
            cand.append((r, cs))
    cand.sort(key=lambda x: (x[0].get("ctime_iso") or ""))
    return cand


def do_report(cand):
    L = [f"candidates: {len(cand)}   (corpus: "
         f"{sum(1 for _ in open(CLEAN, encoding='utf-8'))} records)", ""]
    for i, (r, cs) in enumerate(cand, 1):
        L.append(f"[{i:3d}] {(r.get('ctime_iso') or '')[:7]} "
                 f"form={r.get('form_group') or '?'} "
                 f"k7={r.get('cluster_k7')} "
                 f"canon={r.get('is_canonical')} ctx={r.get('context_sufficiency')} "
                 f"cue={'+'.join(cs)}")
        L.append(f"      id={r.get('clean_id')}")
        L.append(f"      {(r.get('text_norm') or '').replace(chr(10), ' ')}")
        L.append("")
    txt = "\n".join(L)
    os.makedirs(C.LOGS, exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"-> {REPORT}")


def read_decisions():
    if not os.path.exists(DECISIONS):
        return None
    dec = {}
    for line in open(DECISIONS, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise SystemExit(f"bad decisions line (need TAB): {line!r}")
        decision = parts[0].strip()
        if decision not in ("keep", "reject"):
            raise SystemExit(f"bad decision {decision!r} in line: {line!r}")
        reason = parts[2].strip() if len(parts) > 2 else ""
        for tok in parts[1].replace(" ", "").split(","):
            if not tok:
                continue
            i = int(tok)
            if i in dec:
                raise SystemExit(f"index {i} decided twice")
            dec[i] = (decision, reason)
    return dec


def do_commit(cand):
    dec = read_decisions()
    if dec is None:
        raise SystemExit(f"no decisions file: {DECISIONS}")
    undecided = [i for i in range(1, len(cand) + 1) if i not in dec]
    unknown = [i for i in dec if i < 1 or i > len(cand)]
    if undecided:
        raise SystemExit(f"{len(undecided)} candidates undecided: "
                         f"{undecided[:20]}{' ...' if len(undecided) > 20 else ''}")
    if unknown:
        raise SystemExit(f"indices out of range: {unknown[:20]}")

    out = []
    for i, (r, cs) in enumerate(cand, 1):
        decision, reason = dec[i]
        out.append({
            "clean_id": r.get("clean_id"),
            "decision": decision,
            "cue": cs,
            "reason": reason,
            "text": r.get("text_norm"),
            "year": (r.get("ctime_iso") or "")[:4],
            "canonical": r.get("is_canonical"),
            "context_sufficiency": r.get("context_sufficiency"),
            "decided_by": "human-in-the-loop (user decision 2026-10-21: add 紧绷义 "
                          "as 神情紧绷/精神紧绷)",
            "date": "2026-10-21",
        })
    with open(ADJ, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    k = sum(1 for r in out if r["decision"] == "keep")
    print(f"candidates {len(out)}  keep {k}  reject {len(out) - k}")
    print(f"-> {ADJ}")
    print("\nkept items:")
    for r in out:
        if r["decision"] == "keep":
            print(f"  {r['year']}  {r['text'][:80]}")
    cnt = collections.Counter(tuple(sorted(r["cue"])) for r in out
                              if r["decision"] == "keep")
    print("\ncue families among kept items:")
    for cs, n in cnt.most_common():
        print(f"  {'+'.join(cs):60} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    cand = load_candidates()
    if a.report:
        do_report(cand)
    if a.commit:
        do_commit(cand)
    if not a.report and not a.commit:
        print(__doc__)
        print(f"\ncandidates now: {len(cand)}")


if __name__ == "__main__":
    main()
