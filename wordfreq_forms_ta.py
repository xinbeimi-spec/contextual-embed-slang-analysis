# -*- coding: utf-8 -*-
"""Top-5 「塌」词形 per sense category (the frequency figure).

Same design as the 绷 version (wordfreq_forms_by_category.py) and the same three
rules from the handbook:

  1. A form may place 塌 FIRST or SECOND (倒塌 / 坍塌 / 崩塌 / 垮塌 put it second);
     matching only forward from 塌 silently loses half the vocabulary.
  2. Inflections collapse: 塌房了/塌房的/塌房啦 -> 塌房.
  3. Anything not covered by a canonical pattern is counted as 其他词形, so the
     column sums back to the number of 塌 occurrences and nothing is dropped.

The lexicon was NOT hand-guessed: n-grams of the corpus were counted first
(logs/ta/_ngrams.txt), then the canonical list below was built from the forms
that actually occur (>= ~50 times) plus the dictionary forms.

Categories are the FINE 10-class item-level labels of round result4 (the user
asked for the four collapse forms to stay separate). Counts are per occurrence;
per_1000 normalises for category size.

Outputs
  results/<slug>/<round>/wordfreq_top5.csv
  results/<slug>/<round>/wordfreq_all_forms.csv
  results/<slug>/<round>/wordfreq_fig_top5.png/.pdf
Run: python src/analyze/wordfreq_forms_ta.py
"""
import os
import re
import sys
import json
import csv
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc_result4.jsonl")
TOPN = 5
OTHER = "其他词形"

# (display, regex) — priority order: most specific / longest first
CANON = [
    ("人设崩塌", r"人设崩塌"),
    ("房子塌了", r"房子塌(?:了|啦)?|屋子塌"),
    ("一塌糊涂", r"一塌糊涂"),
    ("塌鼻梁", r"塌鼻(?:梁|子)"),
    ("塌房", r"塌房(?:了|的|啦|啊|过)?"),
    ("倒塌", r"倒塌(?:了|的|啦)?"),
    ("崩塌", r"崩塌(?:了|的|啦)?"),
    ("坍塌", r"坍塌(?:了|的|啦)?"),
    ("垮塌", r"垮塌(?:了|的)?"),
    ("塌方", r"塌方(?:了|的|啦)?"),
    ("塌陷", r"塌陷(?:了|的|啦)?"),
    ("天塌", r"天塌(?:了|下来)?"),
    ("塌塌", r"塌塌(?:的)?"),
    ("塌下来", r"塌下(?:来|去)"),
    ("塌掉", r"塌(?:掉|落)"),
    ("塌实", r"塌(?:实|心)"),
    ("塌了", r"塌了"),
]
PATTERNS = [(name, re.compile(rx)) for name, rx in CANON]

CAT_ORDER = ["物理·倒塌", "物理·坍塌", "物理·塌方", "物理·塌陷", "物理·塌了",
             "物理·其他", "塌房义项", "成语义项", "下垂凹陷义项", "非义项用法"]
# 2026-10-31 (user decision): the two residual classes are not senses, so they
# are excluded from the figure and from wordfreq_top5.csv. They stay in
# wordfreq_all_forms.csv and in the typology tables (the paper must still report
# that ~24% of the corpus is residual).
EXCLUDE_FROM_FIG = {"物理·其他", "非义项用法"}
CAT_COLOR = {"物理·倒塌": "#1F77B4", "物理·坍塌": "#2CA02C",
             "物理·塌方": "#FF7F0E", "物理·塌陷": "#D62728",
             "物理·塌了": "#9467BD", "物理·其他": "#8C564B",
             "塌房义项": "#E377C2", "成语义项": "#BCBD22",
             "下垂凹陷义项": "#17BECF", "非义项用法": "#7F7F7F"}


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def count_forms(text):
    """Return (Counter over canonical forms, n_target_occurrences)."""
    c = collections.Counter()
    matched_spans = []
    for name, rx in PATTERNS:
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < me and ms < e for ms, me in matched_spans):
                continue                     # already taken by a longer form
            if "塌" not in m.group(0):
                continue
            matched_spans.append((s, e))
            c[name] += 1
    n_t = text.count("塌")
    n_m = sum(c.values())
    if n_t > n_m:
        c[OTHER] += n_t - n_m
    return c, n_t


def main():
    rows = rj(SRC)
    n_items = len(rows)
    cat_of = {r["clean_id"]: r.get("cdpc_category") for r in rows}
    print(f"items={n_items}")

    per_cat = collections.defaultdict(collections.Counter)
    n_occ = collections.Counter()
    for r in rows:
        c, nt = count_forms(r["text_norm"])
        cat = cat_of[r["clean_id"]]
        per_cat[cat].update(c)
        n_occ[cat] += nt

    # ---- tables
    all_rows, top_rows = [], []
    for cat in CAT_ORDER:
        cnt = per_cat.get(cat, collections.Counter())
        size = sum(1 for v in cat_of.values() if v == cat)
        for form, n in cnt.most_common():
            all_rows.append({"category": cat, "form": form, "n": n,
                             "per_1000_occurrences": round(1000 * n / max(n_occ[cat], 1), 1),
                             "category_size": size})
        for rank, (form, n) in enumerate(cnt.most_common(TOPN), 1):
            if cat in EXCLUDE_FROM_FIG:
                continue
            top_rows.append({"category": cat, "rank": rank, "form": form, "n": n,
                             "share_in_category": round(n / max(sum(cnt.values()), 1), 3),
                             "category_size": size})

    p1 = os.path.join(C.TABLES, "wordfreq_top5.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
        w.writeheader()
        w.writerows(top_rows)
    p2 = os.path.join(C.TABLES, "wordfreq_all_forms.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    for r in top_rows:
        print(f"  {r['category']:<10} #{r['rank']} {r['form']:<8}{r['n']:>5}")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cats = [c for c in CAT_ORDER if per_cat.get(c) and c not in EXCLUDE_FROM_FIG]
    ncol = 4
    nrow = (len(cats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.2, 6.2))
    axes = axes.ravel()
    for ax, cat in zip(axes, cats):
        cnt = per_cat[cat].most_common(TOPN)
        forms = [f for f, _ in cnt][::-1]
        vals = [v for _, v in cnt][::-1]
        ax.barh(forms, vals, color=CAT_COLOR[cat], alpha=0.9)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v}", va="center", fontsize=7.5)
        ax.set_title(f"{cat}（n={sum(1 for v in cat_of.values() if v == cat)}）",
                     fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, max(vals) * 1.28 if vals else 1)
        ax.grid(axis="x", ls=":", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(cats):]:
        ax.axis("off")
    fig.suptitle(f"「{C.TARGET_WORD}」各义项中使用频率最高的前 {TOPN} 个词形"
                 f"（次数；同一词形的屈折变体已合并，未覆盖者计入「其他词形」）",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"wordfreq_fig_top5.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)
    print("->", p1)
    print("->", p2)


if __name__ == "__main__":
    main()
