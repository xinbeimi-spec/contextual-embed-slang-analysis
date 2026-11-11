# -*- coding: utf-8 -*-
"""判定性检验：target 表示分不出「绷带」，是**表示**的问题，还是 **CDPC 选峰**的问题？

做法：在同一份 target 向量上，把聚类器换成最普通的 KMeans（k=7，同随机种子），
看它能否把 321 条「绷带」聚成一个独立簇。同时用 masked 向量做同样的事作为对照。

判读
  * 若 target+KMeans 能分出绷带簇  -> 表示本身有信息，是 CDPC 的「峰值排名」把它吞掉了
    （绷带组密而贴边：rho 高但 delta 小，gamma 排不进前 k，于是没有中心点，
      最后 k-medoids 精修时被归到最近的某个中心）
  * 若 target+KMeans 也分不出      -> 才是表示本身的问题

输出：logs/beng/probe_bandage_kmeans.txt
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cdpc_repr_compare_beng import load_repr, form_label, BENG_PROC, LOGDIR

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(BENG_PROC, "beng_clean.jsonl")


def main():
    clean = {r["clean_id"]: r for r in
             (json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip())}
    L = []
    for tag in ("masked", "target"):
        Z, ids = load_repr(os.path.join(BENG_PROC, "emb2", tag))
        forms = [form_label(clean[c]["text_norm"]) for c in ids]
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=7, n_init=30,
                     random_state=C.RANDOM_SEED).fit_predict(Z)
        sizes = collections.Counter(lab.tolist())
        L.append(f"=== {tag} + KMeans(k=7) ===")
        band_basin = False
        for c in sorted(sizes):
            cnt = collections.Counter(f for f, l in zip(forms, lab.tolist())
                                      if l == c)
            top = ", ".join(f"{f}:{n}" for f, n in cnt.most_common(3))
            if cnt["绷带"] >= 0.5 * sizes[c] and sizes[c] >= 100:
                band_basin = True
            L.append(f"  C{c} n={sizes[c]:>5}  [{top}]")
        L.append(f"  -> 绷带是否独立成簇：{'是' if band_basin else '否'}")
        L.append("")
    txt = "\n".join(L)
    os.makedirs(LOGDIR, exist_ok=True)
    p = os.path.join(LOGDIR, "probe_bandage_kmeans.txt")
    open(p, "w", encoding="utf-8").write(txt)
    print(txt)
    print("->", p)


if __name__ == "__main__":
    main()
