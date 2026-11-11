# -*- coding: utf-8 -*-
"""补一步：target 表示在更大的 k 下，绷带能不能分出来？

k=7 时 KMeans 也把 312 条绷带并进「其他」大簇。但固定 k 下 k-means 会把簇预算优先给
那些又大又散的组（绷住 1887 条、绷不住 1051 条），小而成团的绷带就拿不到簇。
这个脚本把 k 从 7 扫到 12，看绷带在哪个 k 才能拿到自己的簇。

判读
  * 若某个 k 下绷带独立成簇        -> target 向量里**有**绷带信息，只是 k=7 的预算不够
  * 若一直到 k=12 都分不出          -> 才是表示本身缺少该区分的信息

输出：logs/beng/probe_bandage_kmeans_k.txt
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cdpc_repr_compare_beng import load_repr, form_label, BENG_PROC, LOGDIR

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(BENG_PROC, "beng_clean.jsonl")


def main():
    clean = {r["clean_id"]: r for r in
             (json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip())}
    from sklearn.cluster import KMeans
    L = []
    for tag in ("masked", "target"):
        Z, ids = load_repr(os.path.join(BENG_PROC, "emb2", tag))
        forms = [form_label(clean[c]["text_norm"]) for c in ids]
        L.append(f"=== {tag}（KMeans，k=7..12）===")
        for k in range(7, 13):
            lab = KMeans(n_clusters=k, n_init=10,
                         random_state=C.RANDOM_SEED).fit_predict(Z)
            sizes = collections.Counter(lab.tolist())
            best_band, best_share, best_n = 0, 0.0, 0
            for c in sizes:
                cnt = collections.Counter(f for f, l in zip(forms, lab.tolist())
                                          if l == c)
                share = cnt["绷带"] / sizes[c]
                if cnt["绷带"] > best_band:
                    best_band, best_share, best_n = cnt["绷带"], share, sizes[c]
            best_info = f"(簇 n={best_n}, 绷带占比 {best_share:.0%})"
            # 独立成簇的判据：该簇至少 80% 是绷带、且簇本身不少于 150 条
            ok = best_band >= 150 and best_share >= 0.8
            L.append(f"  k={k:>2}  含绷带最多的簇：{best_band} 条 {best_info}  "
                     f"-> 独立成簇：{'是' if ok else '否'}")
        L.append("")
    txt = "\n".join(L)
    os.makedirs(LOGDIR, exist_ok=True)
    p = os.path.join(LOGDIR, "probe_bandage_kmeans_k.txt")
    open(p, "w", encoding="utf-8").write(txt)
    print(txt)
    print("->", p)


if __name__ == "__main__":
    main()
