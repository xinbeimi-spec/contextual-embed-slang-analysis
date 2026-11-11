# -*- coding: utf-8 -*-
"""绷（beng）：用 TARGET 表示重新做 CDPC 聚类，并与 MASKED 表示逐项对比。

背景（2026-11-01）
  共享流水线 cdpc_pipeline.py 已经跟着第三个词（拉 la）走了，它内部的词形表与
  类别规则是「拉」的。要在**绷**上重跑，不能借用那份规则，否则会把绷的数据套上
  拉的类别。所以本脚本对绷是**自包含**的：词形表 / 类别规则 / 人工义项目录都在
  文件里写死，只复用 cluster_cdpc.py 里稳定的数学（密度峰 / 标签传播 / k-medoids）。

协议（两种表示完全一致，对比才公平）
  shards -> 第 9–12 层平均 -> 减均值 -> 去掉第 1 主成分(ABTT k=1)
        -> L2 归一化 -> 余弦相似度 -> CDPC(dc=第 2 百分位) -> k-medoids 精修 -> PCA(仅画图)
  唯一不同的是向量来源：
    masked = emb2/masked（目标字被 [MASK] 替换，取 MASK 位置隐状态）
    target = emb2/target（目标字 span 内 token 隐状态平均）

输出（results/beng/result5/）
  cdpc_fig1_clusters_masked.png/.pdf        各表示的聚类散点
  cdpc_fig1_clusters_target.png/.pdf
  cdpc_fig4_stacked_masked.png/.pdf         各表示的六类历时构成（独立成图）
  cdpc_fig4_stacked_target.png/.pdf
  cdpc_sizes_{masked,target}.csv            每簇大小 + 中心句
  cdpc_cluster_x_form_{masked,target}.csv   簇 × 词形（纯度来源）
  typology_sizes_{masked,target}.csv        六类数量与占比
  typology_by_year_{masked,target}.csv      年度 × 六类
  k_sweep_{masked,target}.csv               k=5..9 的词形纯度/silhouette/最小盆地
  repr_compare.csv                          两表示汇总 + ARI/NMI/同簇匹配率
  data/processed/beng/beng_cdpc_target.jsonl  逐条 target 版标注（供后续词频分析）

运行
  $PY src\\analyze\\cdpc_repr_compare_beng.py --k 7
"""
import os
import re
import sys
import json
import csv
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from cluster_cdpc import density_peaks, assign, kmedoids_refine

sys.stdout.reconfigure(encoding="utf-8")

LAYERS = (9, 10, 11, 12)

# 绷的 6 类（与 results/beng/result4 的 mask 结果完全一致，保证可比）
CAT_ORDER = ["物理义", "忍住义", "忍不住义", "评价义", "紧绷义", "非义项用法"]
CAT_COLOR = {"物理义": "#4C72B0", "忍住义": "#DD8452",
             "忍不住义": "#C44E52", "评价义": "#55A868",
             "紧绷义": "#8172B3", "非义项用法": "#BBBBBB"}
# 词形表（顺序 = 优先顺序；具体的在前）
FORMS_BENG = [("绷不住", ["绷不住"]), ("绷带", ["绷带"]),
              ("绷紧", ["绷紧", "紧绷"]), ("难绷", ["难绷"]),
              ("绷住", ["绷住"])]
# item 级判「其他」时，若句子里出现身体/材料名词，仍算物理义（残差归位规则）
PHYSICAL_FRAME = ("脚踝|小腿|膝盖|大腿|手臂|小臂|大臂|手腕|手指|脚趾|脖子|肩|"
                  "腰|背|腹部|腹肌|核心|肌肉|韧带|皮肤|头皮|毛孔|肚子|脸|"
                  "身体|全身|部位|铜丝|钢丝|布料|橡皮|弹簧|腹|臀|腿")

BENG_RAW = os.path.join(C.RAW_ROOT, "beng")
BENG_PROC = os.path.join(C.PROC_ROOT, "beng")
CLEAN = os.path.join(BENG_PROC, "beng_clean.jsonl")
ADJ = os.path.join(BENG_PROC, "jinbeng_adjudication.jsonl")
OUTDIR = os.path.join(C.RESULTS, "beng", "result5")
LOGDIR = os.path.join(C.ROOT, "logs", "beng")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def form_label(t):
    for f, pats in FORMS_BENG:
        for p in pats:
            if p in t:
                return f
    return "其他"


def item_category(text, clean_id, jinbeng):
    """条目级类别。人工判定过的紧绷义优先，其次按词形，最后残差按身体/材料名词兜底。"""
    if clean_id in jinbeng:
        return "紧绷义"
    f = form_label(text)
    if f in ("绷带", "绷紧"):
        return "物理义"
    if f == "绷不住":
        return "忍不住义"
    if f == "难绷":
        return "评价义"
    if f == "绷住":
        return "忍住义"
    if re.search(PHYSICAL_FRAME, text or ""):
        return "物理义"
    return "非义项用法"


def load_repr(root):
    """两种表示共用同一套变换，唯一区别是 root 目录。"""
    index = rj(os.path.join(root, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(root, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    Z = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    Z = Z - (Z @ Vt[:1].T) @ Vt[:1]           # ABTT k=1
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12), ids


def cluster(Z, k, dc_pct=2.0):
    S = Z @ Z.T
    rho, delta, gamma, parent, dc = density_peaks(S, dc_pct)
    labels = assign(parent, np.argsort(-gamma)[:k], rho)
    labels, medoids = kmedoids_refine(Z, labels)
    return labels, medoids, rho, gamma, dc


def pca2(Z):
    Zm = Z - Z.mean(axis=0, keepdims=True)
    _, _, V = np.linalg.svd(Zm, full_matrices=False)
    coords = Zm @ V[:2].T
    var2 = float(np.var(coords, axis=0).sum() / np.var(Z, axis=0).sum())
    return coords, var2


def silhouette(Z, labels):
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(Z, labels, metric="cosine",
                                  sample_size=min(1500, len(labels)),
                                  random_state=C.RANDOM_SEED))


def scatter(ax, coords, cats, medoids, title, var2):
    """在一个坐标轴里画一张散点（按六类着色 + 标出密度峰中心）。"""
    ycat = np.array(cats)
    for c in CAT_ORDER:
        m = ycat == c
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1],
                       s=30 if c == "紧绷义" else 7, alpha=0.55,
                       color=CAT_COLOR[c], edgecolors="none",
                       label=f"{c} (n={int(m.sum())})")
    mids = [medoids[c] for c in sorted(medoids)]
    ax.scatter(coords[mids, 0], coords[mids, 1], s=120, marker="X",
               c="#222222", edgecolors="white", linewidths=0.8, zorder=5)
    for c in sorted(medoids):
        i = medoids[c]
        ax.annotate(f"C{c}", (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(7, 6), fontsize=7)
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title(title, fontsize=10)
    ax.text(0.01, 0.01,
            f"注：聚类在 768 维余弦空间完成，此二维投影解释 {100*var2:.1f}% 方差，仅用于展示。",
            transform=ax.transAxes, fontsize=6.5, color="#555555")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--dc-pct", type=float, default=2.0)
    ap.add_argument("--round", default="result5",
                    help="写入 results/beng/<round>/，默认 result5（k=7）")
    a = ap.parse_args()

    # --round 决定输出目录：不同 k 各占一轮，互不覆盖
    global OUTDIR
    OUTDIR = os.path.join(C.RESULTS, "beng", a.round)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    clean = {r["clean_id"]: r for r in rj(CLEAN)}
    jinbeng = {r["clean_id"] for r in rj(ADJ) if r.get("decision") == "keep"}
    print(f"clean rows={len(clean)}  jinbeng keeps={len(jinbeng)}  k={a.k}")

    res = {}
    for tag, root in (("masked", C.EMB_MASK_DIR), ("target", C.EMB_DIR)):
        root = os.path.join(BENG_PROC, "emb2", os.path.basename(root))
        Z, ids = load_repr(root)
        labels, medoids, rho, gamma, dc = cluster(Z, a.k, a.dc_pct)
        sizes = collections.Counter(labels.tolist())
        coords, var2 = pca2(Z)
        sil = silhouette(Z, labels)
        texts = [clean[c]["text_norm"] for c in ids]
        cat_item = [item_category(t, c, jinbeng) for t, c in zip(texts, ids)]

        # 簇级类别 = 该簇条目级类别的多数
        ct = collections.defaultdict(collections.Counter)
        for lab, cat in zip(labels.tolist(), cat_item):
            ct[lab][cat] += 1
        cluster_cat = {c: ct[c].most_common(1)[0][0] for c in ct}
        purity = {c: ct[c].most_common(1)[0][1] / sum(ct[c].values()) for c in ct}
        cats = [cluster_cat[l] for l in labels.tolist()]
        for i, c in enumerate(ids):          # 人工判定的紧绷义在条目级保留
            if c in jinbeng:
                cats[i] = "紧绷义"

        print(f"\n[{tag}] N={len(ids)} k={len(sizes)} silhouette={sil:.4f} "
              f"dc={dc:.4f} PCA2={100*var2:.1f}% min={min(sizes.values())}")
        for c in sorted(sizes):
            forms = collections.Counter(form_label(texts[i])
                                        for i in np.where(labels == c)[0])
            top = ", ".join(f"{f}:{n}" for f, n in forms.most_common(3))
            print(f"  C{c} n={sizes[c]:>5} -> {cluster_cat[c]:8} "
                  f"purity={purity[c]:.2f}  [{top}]")

        # ---- 表：簇大小 / 簇 × 词形
        with open(os.path.join(OUTDIR, f"cdpc_sizes_{tag}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "size", "share", "category", "purity",
                        "silhouette_global", "medoid_clean_id", "medoid_text"])
            for c in sorted(sizes):
                m = medoids.get(c)
                w.writerow([c, sizes[c], round(sizes[c] / len(ids), 4),
                            cluster_cat[c], round(purity[c], 4), round(sil, 4),
                            ids[m] if m is not None else "",
                            texts[m][:150] if m is not None else ""])

        with open(os.path.join(OUTDIR, f"cdpc_cluster_x_form_{tag}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "size", "绷不住", "绷带", "绷紧", "难绷",
                        "绷住", "其他"])
            for c in sorted(sizes):
                cnt = collections.Counter(form_label(texts[i])
                                          for i in np.where(labels == c)[0])
                w.writerow([c, sizes[c], cnt["绷不住"], cnt["绷带"], cnt["绷紧"],
                            cnt["难绷"], cnt["绷住"], cnt["其他"]])

        # ---- 表：六类构成 / 年度构成（分母排除极零散义项，与 result4 口径一致）
        scored = [(i, cats[i]) for i in range(len(ids))
                  if not clean[ids[i]].get("excluded_from_typology")]
        cnt = collections.Counter(c for _i, c in scored)
        total = len(scored) or 1
        with open(os.path.join(OUTDIR, f"typology_sizes_{tag}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["category", "n", "share"])
            for c in CAT_ORDER:
                w.writerow([c, cnt[c], round(cnt[c] / total, 4)])
            w.writerow(["TOTAL", total, 1.0])

        ready = [(i, cats[i]) for i, c in scored
                 if clean[ids[i]].get("is_canonical") == 1
                 and clean[ids[i]].get("context_sufficiency") == 1]
        tab = collections.defaultdict(collections.Counter)
        for i, c in ready:
            y = (clean[ids[i]].get("ctime_iso") or "")[:4]
            if y:
                tab[y][c] += 1
        with open(os.path.join(OUTDIR, f"typology_by_year_{tag}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["year", "n"] + CAT_ORDER
                       + [f"share_{c}" for c in CAT_ORDER])
            for y in sorted(tab):
                n = sum(tab[y].values())
                w.writerow([y, n] + [tab[y][c] for c in CAT_ORDER]
                           + [round(tab[y][c] / n, 4) for c in CAT_ORDER])

        # 加权纯度：每簇「主导词形」的条数求和 / N（小簇不再被同等权重放大）
        pur_w = 0
        bandage_basin = 0
        for c in sizes:
            sub = collections.Counter(form_label(texts[i])
                                      for i in np.where(labels == c)[0])
            pur_w += sub.most_common(1)[0][1]
            if sub["绷带"] >= 0.5 * sizes[c] and sizes[c] >= 100:
                bandage_basin = 1
        res[tag] = dict(Z=Z, ids=ids, labels=labels, medoids=medoids,
                        sizes=sizes, texts=texts, sil=sil, var2=var2,
                        purity=purity, purity_w=pur_w / len(ids),
                        bandage_basin=bandage_basin, coords=coords, cats=cats,
                        cat_counts=cnt, total=total, tab=tab,
                        n_ready=len(ready), dc=dc)
        print(f"  [{tag}] form_purity_weighted={pur_w/len(ids):.4f}  "
              f"绷带独立成簇={bool(bandage_basin)}")

    # ---- target 版逐条标注（供后续词频/核查使用）
    r = res["target"]
    if a.round == "result5":
        out_jsonl = os.path.join(BENG_PROC, "beng_cdpc_target.jsonl")
    else:
        out_jsonl = os.path.join(BENG_PROC, f"beng_cdpc_target_{a.round}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, cid in enumerate(r["ids"]):
            rec = dict(clean[cid])
            rec["cdpc_cluster"] = int(r["labels"][i])
            rec["cdpc_category"] = r["cats"][i]
            rec["cdpc_is_medoid"] = 1 if r["medoids"].get(int(r["labels"][i])) == i else 0
            rec["representation"] = "target"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("\n->", out_jsonl)

    # ---- 对齐检验：两解的 ARI / NMI / 同簇最佳匹配率
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    assert res["masked"]["ids"] == res["target"]["ids"], "两份 id 顺序不一致"
    ari = adjusted_rand_score(res["masked"]["labels"], res["target"]["labels"])
    nmi = normalized_mutual_info_score(res["masked"]["labels"],
                                       res["target"]["labels"])
    ct = collections.Counter(zip(res["masked"]["labels"].tolist(),
                                 res["target"]["labels"].tolist()))
    best = {}
    for (cm, ctn), n in ct.items():
        if n > best.get(cm, (None, 0))[1]:
            best[cm] = (ctn, n)
    match_rate = sum(v[1] for v in best.values()) / len(res["masked"]["ids"])

    rows = []
    for tag in ("masked", "target"):
        r = res[tag]
        rows.append({"representation": tag, "k": a.k, "n": len(r["ids"]),
                     "silhouette": round(r["sil"], 4),
                     "form_purity_w": round(r["purity_w"], 4),
                     "form_purity_mean": round(
                         sum(r["purity"].values()) / len(r["purity"]), 4),
                     "bandage_basin": r["bandage_basin"],
                     "pca2_var": round(r["var2"], 4),
                     "min_basin": int(min(r["sizes"].values())),
                     "max_basin": int(max(r["sizes"].values())),
                     "dc": round(r["dc"], 4), "analysis_ready": r["n_ready"]})
    with open(os.path.join(OUTDIR, "repr_compare.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 六类占比对比表（两表示并排）
    with open(os.path.join(OUTDIR, "typology_share_compare.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "masked_n", "masked_share",
                    "target_n", "target_share", "delta_share"])
        for c in CAT_ORDER:
            mn, tn = res["masked"]["cat_counts"][c], res["target"]["cat_counts"][c]
            ms, ts = mn / res["masked"]["total"], tn / res["target"]["total"]
            w.writerow([c, mn, round(ms, 4), tn, round(ts, 4), round(ts - ms, 4)])

    # ---- k 扫描（两种表示各一遍，支撑「k=7 是否仍然合适」）
    for tag in ("masked", "target"):
        Z = res[tag]["Z"]
        S = Z @ Z.T
        rho, delta, gamma, parent, dc = density_peaks(S, a.dc_pct)
        with open(os.path.join(OUTDIR, f"k_sweep_{tag}.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["k", "silhouette", "form_purity", "min_basin", "max_basin"])
            for k in range(5, 10):
                lab = assign(parent, np.argsort(-gamma)[:k], rho)
                lab, med = kmedoids_refine(Z, lab, iters=4)
                sz = collections.Counter(lab.tolist())
                pur = 0
                for c in sz:
                    sub = collections.Counter(form_label(res[tag]["texts"][i])
                                              for i in np.where(lab == c)[0])
                    pur += sub.most_common(1)[0][1]
                w.writerow([k, round(silhouette(Z, lab), 4), round(pur / len(Z), 4),
                            int(min(sz.values())), int(max(sz.values()))])
                print(f"  [{tag}] k={k} purity={pur/len(Z):.4f} "
                      f"min_basin={min(sz.values())}")

    # ---- 图：单表示散点 + 并排对比 + 历时堆叠并排
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9

    for tag, label in (("masked", "[MASK] 表示"), ("target", "target 表示")):
        r = res[tag]
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        scatter(ax, r["coords"], r["cats"], r["medoids"],
                f"「绷」{label} · CDPC 聚类（k={len(r['sizes'])}，cosine，"
                f"silhouette={r['sil']:.3f}）", r["var2"])
        handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                          color=CAT_COLOR[c],
                          label=f"{c} (n={r['cat_counts'][c]})")
                   for c in CAT_ORDER if r["cat_counts"][c]]
        handles.append(Line2D([], [], marker="X", linestyle="", markersize=8,
                              color="#222222", label="密度峰中心（medoid）"))
        ax.legend(handles=handles, fontsize=7.5, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            q = os.path.join(OUTDIR, f"cdpc_fig1_clusters_{tag}.{ext}")
            fig.savefig(q, dpi=300, bbox_inches="tight")
            print("->", q)
        plt.close(fig)

    # 历时构成：每种表示各自一张独立图
    # （2026-11-01 用户要求：两种表示不要放在同一张图里，否则读者会以为是同一组数据）
    for tag, label in (("masked", "[MASK] 表示"), ("target", "target 表示")):
        tab = res[tag]["tab"]
        years = sorted(tab)
        shares = {c: [tab[y][c] / max(1, sum(tab[y].values())) for y in years]
                  for c in CAT_ORDER}
        fig, ax = plt.subplots(figsize=(9.8, 4.6))
        ax.stackplot(years, [shares[c] for c in CAT_ORDER], labels=CAT_ORDER,
                     colors=[CAT_COLOR[c] for c in CAT_ORDER], alpha=0.95)
        ax.set_ylim(0, 1)
        ax.set_xlabel("年份")
        ax.set_ylabel("占比")
        ax.set_title(f"「绷」六类用法的历时构成 · {label}"
                     f"（分析可用 n={res[tag]['n_ready']}）", fontsize=10.5)
        # 图例放到画布右侧外：压在色带上会看不清图例文字
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0, fontsize=7.5, frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            q = os.path.join(OUTDIR, f"cdpc_fig4_stacked_{tag}.{ext}")
            fig.savefig(q, dpi=300, bbox_inches="tight")
            print("->", q)
        plt.close(fig)

    # ---- 文字汇总
    L = [f"绷（beng）masked vs target 聚类对比（k={a.k}）", "",
         "form_purity_w = 按簇规模加权的词形纯度（论文用这个）；"
         "form_purity_mean = 未加权平均（辅助）",
         "bandage_basin = 是否存在「≥50% 绷带且 n≥100」的簇（1=物理义能成形）", "",
         f"{'repr':<8}{'silhouette':>11}{'purity_w':>10}{'purity_mean':>13}"
         f"{'bandage':>9}{'PCA2':>8}{'min':>6}{'max':>6}{'dc':>8}{'ready':>7}"]
    for r in rows:
        L.append(f"{r['representation']:<8}{r['silhouette']:>11.4f}"
                 f"{r['form_purity_w']:>10.4f}{r['form_purity_mean']:>13.4f}"
                 f"{r['bandage_basin']:>9}{r['pca2_var']*100:>7.1f}%"
                 f"{r['min_basin']:>6}{r['max_basin']:>6}"
                 f"{r['dc']:>8.4f}{r['analysis_ready']:>7}")
    L += ["", f"簇一致度：ARI={ari:.4f}  NMI={nmi:.4f}  同簇最佳匹配率={match_rate:.3f}",
          "", "簇级对应（masked -> target, 重叠条数）："]
    for cm in sorted(best):
        L.append(f"  C{cm} -> C{best[cm][0]}  ({best[cm][1]})")
    L += ["", "六类占比对比："]
    for c in CAT_ORDER:
        mn, tn = res["masked"]["cat_counts"][c], res["target"]["cat_counts"][c]
        ms = mn / res["masked"]["total"]
        ts = tn / res["target"]["total"]
        L.append(f"  {c:<8} masked {mn:>5} ({100*ms:>5.1f}%)   "
                 f"target {tn:>5} ({100*ts:>5.1f}%)   Δ={100*(ts-ms):+6.1f}pp")
    txt = "\n".join(L)
    log_name = ("repr_compare_beng.txt" if a.round == "result5"
                else f"repr_compare_beng_{a.round}.txt")
    open(os.path.join(LOGDIR, log_name), "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\n->", os.path.join(LOGDIR, log_name))


if __name__ == "__main__":
    main()
