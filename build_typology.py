# -*- coding: utf-8 -*-
"""Stage 4g: FINAL six-category typology (5 senses + 1 non-sense category).

Revision history
  2026-10-20  five categories: 物理义 / 忍住义 / 忍不住义 / 评价义 / 非义项用法
  2026-10-21  SIX categories. Two changes, both user decisions:
              (a) NEW 「紧绷义」 — 神情紧绷 / 精神紧绷 (facial & psychological
                  tension). The old rule sent every 紧绷 item to 非义项用法,
                  because cluster 6 was checked before the surface form.
              (b) RESIDUAL FIX — cluster 6 also swallowed genuine PHYSICAL
                  绷紧 uses (脚踝紧绷, 皮肤紧绷, 肌肉绷紧, 头皮紧绷). Those are
                  not mention-type items and are moved back to 物理义. Items
                  whose sense cannot be pinned down stay in 非义项用法.

Why not "five surface forms"
  Five forms cannot be recovered as five clusters: forcing k=5 always leaves a
  large merged blob, because 绷住 needs two or three clusters and a
  heterogeneous residual group always remains. That is reported, not hidden.

What is reported instead
  A six-way typology in which the last category is explicitly NOT a sense:
    物理义       绷带 / 绷紧 / 紧绷(身体)  -> dictionary tauten/bandage sense
    忍住义       绷住                      -> hold back laughter
    忍不住义     绷不住                    -> fail to hold back
    评价义       难绷                      -> evaluative "cringe/hard to watch"
    紧绷义       神情/精神紧绷              -> facial & psychological tension
                                            (cue-filtered + hand-adjudicated,
                                             see jinbeng_cue.py)
    非义项用法   residual cluster           -> mention-type: meta wordplay,
                                            verbatim repetition, spelling
                                            variants, homograph collisions

  The non-sense category is use/mention territory: the word is quoted or
  re-spelled rather than used to express a sense. Keeping it as a labelled
  category is more honest than deleting those items and inflating purity.

  Items flagged `excluded_from_typology` at the cleaning stage (dictionary
  senses too rare to carry a category: 工艺义 绣绷/绷绢, 程度义 绷脆/绷硬) are
  KEPT in the item table but kept OUT of the denominators; they are reported
  separately in results/tables/typology_excluded.csv.

Outputs
  data/processed/beng_typology.jsonl
  results/tables/typology_sizes.csv
  results/tables/typology_by_year.csv
  results/tables/typology_examples.csv
  results/tables/typology_excluded.csv
  results/tables/jinbeng_examples.csv
  results/figures/fig4_typology_stacked.png/.pdf
  results/figures/fig4b_typology_line.png/.pdf
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from jinbeng_cue import load_adjudication, cues_of

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
FORMS = os.path.join(C.PROCESSED, "beng_forms.jsonl")
OUT = os.path.join(C.PROCESSED, "beng_typology.jsonl")

# --- cluster -> category mapping for the k=7 solution.
# Re-derived for the expanded corpus; build_typology prints the cluster x form
# contingency it is relying on so a stale mapping cannot go unnoticed.
# (2026-10-21, n=4,488: 0=residual/mention, 1/2/6=绷住, 3=绷带, 4=难绷,
#  5=绷不住 -- the ids are NOT the same as in the 2,663-item run.)
NON_SENSE_CLUSTER = 0
BENGZHU_CLUSTERS = {1, 2, 6}
BENG_BUZHU_CLUSTER = 5
NANBENG_CLUSTER = 4
BANDAI_CLUSTER = 3

CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
CAT_COLOR = {"物理义": "#4C72B0", "忍住义": "#DD8452",
             "忍不住义": "#C44E52", "评价义": "#55A868",
             "紧绷义": "#8172B3", "非义项用法": "#BBBBBB"}

# 物理义 inside the residual cluster: a 绷紧/紧绷 item that names a body part or
# a material is a physical use, not a mention. Anything else stays residual.
PHYSICAL_FRAME = ("脚踝|小腿|膝盖|大腿|手臂|小臂|大臂|手腕|手指|脚趾|脖子|肩|"
                  "腰|背|腹部|腹肌|核心|肌肉|韧带|皮肤|头皮|毛孔|肚子|脸|"
                  "身体|全身|部位|铜丝|钢丝|布料|橡皮|弹簧|腹|臀|腿")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def form_label(t):
    for f, pats in (("绷不住", ["绷不住"]), ("绷带", ["绷带"]),
                    ("绷紧", ["绷紧", "紧绷"]), ("难绷", ["难绷"]),
                    ("绷住", ["绷住"])):
        for p in pats:
            if p in t:
                return f
    return "其他"


def assign(cluster_k7, form, text="", clean_id=None, jinbeng=None):
    """Explicit, auditable rule. Cluster id first, form as a cross-check."""
    import re
    # (a) the new sense wins, but ONLY for hand-adjudicated items
    if jinbeng and clean_id in jinbeng:
        return "紧绷义"
    if cluster_k7 == NON_SENSE_CLUSTER:
        # (b) residual fix: genuine physical 绷紧 uses are not mention-type
        if form == "绷紧" and re.search(PHYSICAL_FRAME, text or ""):
            return "物理义"
        return "非义项用法"
    if form in ("绷带", "绷紧"):
        return "物理义"
    if cluster_k7 in BENGZHU_CLUSTERS or form == "绷住":
        return "忍住义"
    if cluster_k7 == BENG_BUZHU_CLUSTER or form == "绷不住":
        return "忍不住义"
    if cluster_k7 == NANBENG_CLUSTER or form == "难绷":
        return "评价义"
    return "非义项用法"


def main():
    import argparse
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default=FORMS, help="input table (test hooks)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tables-dir", default=C.TABLES)
    ap.add_argument("--figures-dir", default=C.FIGURES)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    tables_dir, figures_dir = a.tables_dir, a.figures_dir
    os.makedirs(tables_dir, exist_ok=True)

    rows = rj(a.forms)
    jinbeng = load_adjudication()
    print(f"items: {len(rows)}   jinbeng keep-list: {len(jinbeng)}")

    # ---- cluster x form contingency (the mapping above must match this)
    cont = collections.defaultdict(collections.Counter)
    for r in rows:
        cont[r.get("cluster_k7")][r.get("form_group")] += 1
    print("\n=== cluster_k7 x surface form ===")
    for cl in sorted(cont, key=lambda x: (x is None, x)):
        tot = sum(cont[cl].values())
        top = ", ".join(f"{f}:{n}" for f, n in cont[cl].most_common(4))
        print(f"  cluster {cl}  n={tot:5d}  {top}")

    # Mapping sanity check: a stale cluster id (left over from the previous
    # corpus) would silently relabel thousands of items, so the dominant form
    # of every cluster must be compatible with the category it is given.
    EXPECTED = {"物理义": {"绷带", "绷紧"}, "忍住义": {"绷住"},
                "忍不住义": {"绷不住"}, "评价义": {"难绷"}}
    print("\n=== mapping check (cluster -> category vs dominant form) ===")
    bad = 0
    for cl in sorted(cont, key=lambda x: (x is None, x)):
        dom_form, dom_n = cont[cl].most_common(1)[0]
        dom_share = dom_n / sum(cont[cl].values())
        cat = assign(cl, dom_form, "", None, None)
        exp = EXPECTED.get(cat)
        ok = exp is None or dom_form in exp or dom_share < 0.5
        if not ok:
            bad += 1
        print(f"  cluster {cl:>3} -> {cat:8} dominant={dom_form}("
              f"{100*dom_share:.0f}%)  {'ok' if ok else '!! MISMATCH'}")
    if bad:
        print(f"\n  !! {bad} cluster(s) map to a category their dominant form "
              f"does not support -- update the constants at the top of this file"
              f" before trusting the output below.")

    for r in rows:
        t = r.get("text_norm") or ""
        r["category"] = assign(r["cluster_k7"], r.get("form_group"), t,
                               r.get("clean_id"), jinbeng)
        r["uses_word_as_mention"] = 1 if r["category"] == "非义项用法" else 0
        if r.get("category") == "紧绷义":
            r["jinbeng_cues"] = cues_of(t)
            r["jinbeng_reason"] = jinbeng.get(r.get("clean_id"), {}).get("reason")

    # items the cleaning stage declared too rare to carry a category
    flagged = [r for r in rows if r.get("excluded_from_typology")]
    scored = [r for r in rows if not r.get("excluded_from_typology")]
    print(f"\nexcluded_from_typology: {len(flagged)} (kept in the table, "
          f"out of denominators)")
    print(f"typology N: {len(scored)}")

    cnt = collections.Counter(r["category"] for r in scored)
    print("\n=== typology sizes ===")
    for c in CAT_ORDER:
        print(f"  {c:10} {cnt[c]:5d}  {100*cnt[c]/len(scored):5.1f}%")

    # cross-check: mention category must not hide physical 绷紧 items any more
    conflict = sum(1 for r in scored
                   if r["category"] == "非义项用法"
                   and r.get("cluster_k7") != NON_SENSE_CLUSTER)
    print(f"\nitems routed to 非义项用法 where the cluster disagreed: {conflict}")
    resid_phys = sum(1 for r in scored
                     if r["category"] == "物理义"
                     and r.get("cluster_k7") == NON_SENSE_CLUSTER)
    print(f"residual-cluster (k7={NON_SENSE_CLUSTER}) items re-routed to 物理义 "
          f"(residual fix): {resid_phys}")

    p1 = os.path.join(tables_dir, "typology_sizes.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "share"])
        for c in CAT_ORDER:
            w.writerow([c, cnt[c], round(cnt[c] / len(scored), 4)])
        w.writerow(["TOTAL", len(scored), 1.0])

    # ---- by year, analysis-ready only
    ready = [r for r in scored
             if r.get("is_canonical") == 1 and r.get("context_sufficiency") == 1]
    tab = collections.defaultdict(collections.Counter)
    for r in ready:
        y = (r.get("ctime_iso") or "")[:4]
        if y:
            tab[y][r["category"]] += 1
    p2 = os.path.join(tables_dir, "typology_by_year.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["year", "n"] + CAT_ORDER
                   + [f"share_{c}" for c in CAT_ORDER])
        for y in sorted(tab):
            t = tab[y]
            n = sum(t.values())
            w.writerow([y, n] + [t[c] for c in CAT_ORDER]
                       + [round(t[c] / n, 4) for c in CAT_ORDER])

    # ---- examples per category
    p3 = os.path.join(tables_dir, "typology_examples.csv")
    rng = np.random.default_rng(C.RANDOM_SEED)
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "clean_id", "ctime", "n_chars", "cluster_k7",
                    "form_group", "text"])
        for c in CAT_ORDER:
            pool = [r for r in scored if r["category"] == c
                    and r.get("context_sufficiency") == 1
                    and 12 <= len(r.get("text_norm") or "") <= 80]
            if not pool:
                continue
            pick = rng.choice(len(pool), size=min(12, len(pool)), replace=False)
            for i in pick:
                r = pool[int(i)]
                w.writerow([c, r.get("clean_id"),
                            (r.get("ctime_iso") or "")[:10],
                            len(r.get("text_norm") or ""), r.get("cluster_k7"),
                            r.get("form_group"), r.get("text_norm")])

    # ---- excluded (rare senses) + all 紧绷义 items, for the write-up
    p4 = os.path.join(tables_dir, "typology_excluded.csv")
    with open(p4, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clean_id", "ctime", "category_would_be", "exclusion_reason",
                    "text"])
        for r in flagged:
            w.writerow([r.get("clean_id"), (r.get("ctime_iso") or "")[:10],
                        r.get("category"), r.get("excluded_from_typology"),
                        r.get("text_norm")])

    p5 = os.path.join(tables_dir, "jinbeng_examples.csv")
    with open(p5, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clean_id", "ctime", "year", "cues", "reason", "text"])
        for r in sorted([r for r in scored if r["category"] == "紧绷义"],
                        key=lambda r: (r.get("ctime_iso") or "")):
            w.writerow([r.get("clean_id"), (r.get("ctime_iso") or "")[:10],
                        (r.get("ctime_iso") or "")[:4],
                        ";".join(r.get("jinbeng_cues") or []),
                        r.get("jinbeng_reason"), r.get("text_norm")])

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- figures
    if a.no_figures:
        print("\n(figures skipped: --no-figures)")
        print(f"\nanalysis-ready items: {len(ready)}")
        print(f"{'year':>6}{'n':>7}" + "".join(f"{c:>10}" for c in CAT_ORDER))
        for y in years:
            t = tab[y]
            n = sum(t.values())
            print(f"{y:>6}{n:>7}"
                  + "".join(f"{100*t[c]/n:>9.1f}%" for c in CAT_ORDER))
        for p in (p1, p2, p3, p4, p5, a.out):
            print("->", p)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    years = sorted(tab)
    ns = [sum(tab[y].values()) for y in years]
    shares = {c: [tab[y][c] / sum(tab[y].values()) for y in years]
              for c in CAT_ORDER}

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.stackplot(years, [shares[c] for c in CAT_ORDER],
                 labels=CAT_ORDER,
                 colors=[CAT_COLOR[c] for c in CAT_ORDER], alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_xlabel("年份")
    ax.set_ylabel("占比")
    ax.set_title("「绷」六类用法的历时构成（百分比堆叠）")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig4_typology_stacked.{ext}"),
                    dpi=300, bbox_inches="tight")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.2),
                                 height_ratios=[3, 1], sharex=True)
    for c in CAT_ORDER:
        a1.plot(years, shares[c], marker="o", ms=3.5,
                color=CAT_COLOR[c], label=c)
    a1.set_ylabel("占比")
    a1.set_ylim(0, 1)
    a1.set_title("「绷」各用法占比的历时变化")
    a1.legend(fontsize=8, frameon=False, ncol=3)
    a2.bar(years, ns, color="#999999")
    a2.set_ylabel("样本量 n")
    a2.set_xlabel("年份")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"fig4b_typology_line.{ext}"),
                    dpi=300, bbox_inches="tight")

    print(f"\nanalysis-ready items: {len(ready)}")
    print(f"{'year':>6}{'n':>7}" + "".join(f"{c:>10}" for c in CAT_ORDER))
    for y in years:
        t = tab[y]
        n = sum(t.values())
        print(f"{y:>6}{n:>7}" + "".join(f"{100*t[c]/n:>9.1f}%" for c in CAT_ORDER))

    for p in (p1, p2, p3, p4, p5, a.out):
        print("->", p)
    print("-> fig4_typology_stacked.png/.pdf")
    print("-> fig4b_typology_line.png/.pdf")


if __name__ == "__main__":
    main()
