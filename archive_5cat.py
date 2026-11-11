# -*- coding: utf-8 -*-
"""Archive the five-category artefacts before the corpus is expanded.

Why: the corpus is being enlarged (2,663 -> several thousand items) and the
typology gains a sixth category (紧绷义). Every derived artefact that was
computed in the OLD space must stop being silently reused, otherwise the
"reuse mu / never refit" rule turns into "mix two spaces".

Nothing is deleted. Everything is MOVED under data/processed/_archive_5cat/
and results/_archive_5cat/, and the moves are logged to
data/processed/_archive_5cat/MOVES.jsonl so any number can be traced back.

Usage
    python src/analyze/archive_5cat.py --dry-run     # show what would move
    python src/analyze/archive_5cat.py               # do it
"""
import os
import sys
import json
import glob
import shutil
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

DEST_P = os.path.join(C.PROCESSED, "_archive_5cat")
DEST_R = os.path.join(C.RESULTS, "_archive_5cat")

# directories (moved whole)
DIRS = [
    (os.path.join(C.PROCESSED, "emb"), os.path.join(DEST_P, "emb")),
    (os.path.join(C.PROCESSED, "emb_mask"), os.path.join(DEST_P, "emb_mask")),
    (os.path.join(C.PROCESSED, "pca"), os.path.join(DEST_P, "pca")),
]

# file globs relative to a root
PROCESSED_GLOBS = [
    "beng_forms.jsonl", "beng_typology.jsonl", "beng_k2.jsonl",
    "beng_clustered__*.jsonl", "jinbeng_adjudication.jsonl",
]
TABLE_GLOBS = [
    "typology_*.csv", "selected5_*.csv", "select_k_*.csv",
    "form_cluster_recovery.csv", "forms_k*_*.csv", "prototype_*.csv",
    "anisotropy_by_layer.csv", "centering_effect.csv",
]
FIGURE_GLOBS = [
    "fig1_five_category_space.*", "fig1c_k7_clusters.*",
    "fig2*_selected5_*.png", "fig2*_selected5_*.pdf",
    "fig2_selected5_heatmap.*", "fig4_typology_stacked.*",
    "fig4b_typology_line.*",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    moves = []
    for src, dst in DIRS:
        if os.path.exists(src):
            moves.append((src, dst))
    for g in PROCESSED_GLOBS:
        for p in sorted(glob.glob(os.path.join(C.PROCESSED, g))):
            moves.append((p, os.path.join(DEST_P, os.path.basename(p))))
    for g in TABLE_GLOBS:
        for p in sorted(glob.glob(os.path.join(C.TABLES, g))):
            moves.append((p, os.path.join(DEST_R, "tables",
                                          os.path.basename(p))))
    for g in FIGURE_GLOBS:
        for p in sorted(glob.glob(os.path.join(C.FIGURES, g))):
            moves.append((p, os.path.join(DEST_R, "figures",
                                          os.path.basename(p))))

    print(f"{len(moves)} artefacts to archive")
    seen_src = set()
    for src, dst in moves:
        if src in seen_src:
            continue
        seen_src.add(src)
        print(f"  {'[dry] ' if a.dry_run else ''}{os.path.relpath(src, C.ROOT)}"
              f"  ->  {os.path.relpath(dst, C.ROOT)}")
    moves = [m for m in moves if m[0] in seen_src]
    deduped = []
    for src, dst in moves:
        if (src, dst) not in deduped:
            deduped.append((src, dst))
    moves = deduped
    if a.dry_run:
        return

    log = []
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"  ! destination exists, skipping: {dst}")
            continue
        shutil.move(src, dst)
        log.append({"src": src, "dst": dst,
                    "at": datetime.datetime.now().isoformat(timespec="seconds")})

    with open(os.path.join(DEST_P, "MOVES.jsonl"), "w", encoding="utf-8") as f:
        for m in log:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nmoved {len(log)} artefacts -> {DEST_P} , {DEST_R}")
    print("results/final_5cat/ untouched (it is the frozen 5-category deliverable)")


if __name__ == "__main__":
    main()
