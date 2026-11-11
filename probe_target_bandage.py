# -*- coding: utf-8 -*-
"""诊断：target 表示下「绷带」在 k=6..10 是否形成独立盆地？

背景：results/beng/result5 的对比显示，target 表示在 k=7 时把 312 条「绷带」
并进了以「其他/提及型」为主的 C0（865 条），于是「物理义」在类别统计里变成 0；
masked 表示在同样 k=7 下则把「绷带」单独分成 C1（374 条，绷带 306）。
本脚本只回答一个问题：换 k 能不能让 target 也把绷带分出来。
一旦确认 k 也救不回来，就说明这是表示本身的差异，而不是 k 选错。

输出：logs/beng/probe_target_bandage.txt
"""
import os
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, assign, kmedoids_refine
from cdpc_repr_compare_beng import (load_repr, form_label, CLEAN, BENG_PROC,
                                    LOGDIR)

sys.stdout.reconfigure(encoding="utf-8")


def main():
    clean = {r["clean_id"]: r for r in
             (json.loads(l) for l in open(CLEAN, encoding="utf-8") if l.strip())}
    L = []
    for tag, sub in (("target", "target"), ("masked", "masked")):
        root = os.path.join(BENG_PROC, "emb2", sub)
        Z, ids = load_repr(root)
        texts = [clean[c]["text_norm"] for c in ids]
        forms = [form_label(t) for t in texts]
        S = Z @ Z.T
        rho, delta, gamma, parent, dc = density_peaks(S, 2.0)
        L.append(f"=== {tag}  (N={len(ids)}, dc={dc:.4f}) ===")
        for k in range(6, 11):
            lab = assign(parent, np.argsort(-gamma)[:k], rho)
            lab, med = kmedoids_refine(Z, lab, iters=4)
            sz = collections.Counter(lab.tolist())
            band = []
            for c in sorted(sz):
                cnt = collections.Counter(f for f, l in zip(forms, lab.tolist())
                                          if l == c)
                band.append((c, sz[c], cnt["绷带"],
                             cnt.most_common(1)[0][0]))
            has_band_basin = any(b[2] >= 0.5 * b[1] and b[1] >= 100
                                 for b in band)
            L.append(f"  k={k:>2}  绷带独立成簇={'是' if has_band_basin else '否'}   "
                     + " | ".join(f"C{c}(n={n},绷带={nb},{dom})"
                                  for c, n, nb, dom in band))
        L.append("")
    txt = "\n".join(L)
    os.makedirs(LOGDIR, exist_ok=True)
    p = os.path.join(LOGDIR, "probe_target_bandage.txt")
    open(p, "w", encoding="utf-8").write(txt)
    print(txt)
    print("->", p)


if __name__ == "__main__":
    main()
