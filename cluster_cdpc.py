# -*- coding: utf-8 -*-
"""Stage 4b (v2): CDPC -- density-peak clustering with cosine similarity.

=====================================================================
新手导读（先读这一段，再看代码）
=====================================================================
这个文件是**聚类算法的本体**：把每条语料句子看成一个 768 维向量，据此把相似句子
自动分成若干簇。方法叫 CDPC —— Density Peak Clustering with Cosine similarity
（余弦相似度的密度峰聚类）。全文只做三件事，对应三组函数：

  1) 算「密度」rho 和「峰高」delta -> density_peaks()
        rho_i   = 点 i 周围有多密（用余弦距离算）
        delta_i = 点 i 离「比它更密的点」有多远
        gamma_i = rho_i × delta_i   <- 又密又孤立，才是峰
     直觉：真正的簇中心，周围点很多（rho 大），且附近没有更密的点（delta 大）。

  2) 决定分几簇 k -> choose_k()
        把 gamma 从大到小排，最大的那几个点就是峰；峰的个数 = k。
        可以自动定（找相邻 gamma 的最大倍数落差），也可以人工指定。

  3) 把每个点分到某个峰 -> assign() 和 kmedoids_refine()
        assign()：按密度从高到低，每个点继承「父点」的标签（密度链传播）
        kmedoids_refine()：用「每簇里最居中的真实句子」当簇心，反复重分直到稳定
        （k-medoids 与 k-means 的区别：中心必须是真实样本，不是平均值）

=====================================================================
名词表
=====================================================================
* 余弦相似度 cos(a,b)：两向量夹角的余弦。1=完全同向，0=无关，-1=反向。
  文本向量只看方向、不看长度，所以比欧氏距离合适。
* 余弦距离 d = 1 - cos：把相似度翻成距离，越像越小。
* dc（截断距离）：多近才算邻居的阈值，取所有两两距离的第 2 百分位。
* 各向异性 anisotropy：文本向量会挤在一个窄锥里，随便两句余弦都 >0.7，聚类会退化。
  所以 cdpc_pipeline.load_masked() 先做「减均值 + 去掉第一主成分 + L2 归一化」。

=====================================================================
聚类在完整 768 维上做
=====================================================================
PCA/TSNE 只用于出图。前两个主成分只占约 28.5% 的方差，在二维里聚类是投影幻觉。

=====================================================================
相对 v1 修掉的 bug（为什么代码写得绕）
=====================================================================
Fixes over v1
  * delta of the global density maximum now uses the finite max distance
    (v1 used max over a matrix whose diagonal was inf -> gamma=inf -> k=2).
  * density-chain assignment rewritten: one pass in descending rho, inheriting
    the parent's label. v1's version collapsed every point onto the centre
    marker and silently fell back to nearest-centre assignment.
  * k is chosen from the largest multiplicative drop in sorted gamma, but only
    among gammas above mean+1sd, so a long noise tail cannot set k.
  * per-cluster surface forms now come from jieba word segmentation plus the
    target + 1-2 following chars, which removes cross-boundary artefacts such
    as "松绷" (from 轻松绷住).

Where the geometry is computed
  On the FULL 768-d centred vectors. PCA/TSNE is used only for the figure;
  the first 2 PCs carry just 28.5% of the variance, so clustering there would
  be an artefact of the projection.

Outputs (new files; the cleaned table is never modified)
  data/processed/beng_clustered.jsonl
  results/tables/cluster_sizes.csv
  results/tables/cluster_x_sensehint.csv
  results/tables/cluster_top_forms.csv
  results/tables/cdpc_diagnostics.csv
  results/tables/cdpc_gamma_ranking.csv
"""
import os
import sys
import json
import argparse
import collections

import numpy as np
import jieba

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")
jieba.setLogLevel(60)

EMB = C.EMB_DIR
CLEAN = os.path.join(C.PROCESSED, "beng_clean.jsonl")
OUT_JSONL = os.path.join(C.PROCESSED, "beng_clustered.jsonl")   # replaced by tag at runtime


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def density_peaks(S, dc_pct=2.0):
    """S: 余弦相似度矩阵 (N,N) -> rho, delta, gamma, parent, dc

    参数
        S      : (N, N) 余弦相似度，S[i, j] = cos(句i, 句j)，对角线是 1
        dc_pct : 用第几百分位数当截断距离 dc（默认 2，即第 2 百分位）

    返回
        rho    : (N,)   每个点的局部密度
        delta  : (N,)   每个点到「比自己更密的点」的最小距离
        gamma  : (N,)   峰分 = rho * delta
        parent : (N,)   每个点的「父点」下标（比自己密且最近的那个点），用于传播标签
        dc     : float  实际使用的截断距离

    为什么 dc 取低百分位：如果 dc 取全体平均距离，远处的点也会被算进密度，
    密度就变成「全局平均」，分不出局部峰。取第 2 百分位相当于「只看最近的邻居」，
    rho 因此接近 k-NN 密度。
    """
    # ---------------------------------------------------------------
    # 第 1 步：相似度 -> 距离
    # S 的对角线是 1（自己和自己），1-1=0，所以对角线距离为 0，这是对的。
    # ---------------------------------------------------------------
    D = 1.0 - S

    # 只保留有限值（防止 inf/nan 污染百分位计算）
    finite = D[np.isfinite(D)]
    # np.percentile(数组, 2) = 第 2 百分位数：有 2% 的距离比它更小
    dc = float(np.percentile(finite, dc_pct))

    # ---------------------------------------------------------------
    # 第 2 步：局部密度 rho
    #   对每个点 i，把「它对所有点的距离」用高斯核 exp(-(d/dc)^2) 加权求和。
    #   距离远 -> 权重趋近 0；距离近 -> 权重趋近 1。所以 rho 大 = 周围点多。
    #   (D / dc) ** 2 : numpy 逐元素做除法再平方
    #   .sum(axis=1)  : 沿「列」方向求和，得到每个点一个标量（长度 N）
    # ---------------------------------------------------------------
    rho = np.exp(-(D / dc) ** 2).sum(axis=1)

    # np.argsort(-rho) = 按 rho 从大到小排，返回的是「下标」，不是数值。
    # 例如 rho=[0.5, 9.0, 2.0] -> order=[1, 2, 0]
    order = np.argsort(-rho)

    delta = np.zeros(len(rho))                 # 初始化 delta 全 0
    parent = np.full(len(rho), -1, dtype=int)  # 初始化父点全 -1（=没有父点）

    # ---------------------------------------------------------------
    # 第 3 步：最密那个点的 delta 特殊处理
    #   它是密度最高的，按定义「没有比它更密的点」，所以约定：
    #   delta = 它到其他所有点的最大距离（离所有点最远，等于把整片区域兜住）。
    #   注意要用「有限值」的最大值：如果矩阵里有 inf，这里会算出 inf，
    #   gamma 就会是 inf，k 会被算成 2（这是 v1 的 bug）。
    # ---------------------------------------------------------------
    d_top = D[order[0]].copy()      # 最密那个点到所有点的距离（长度 N）
    d_top[order[0]] = -np.inf       # 把自己的距离设成 -inf，避免 max 选到自己
    delta[order[0]] = float(np.max(d_top))   # 取最大值 = 到最远点的距离

    # ---------------------------------------------------------------
    # 第 4 步：其他每个点的 delta 与父点
    #   按密度从高到低依次处理（rank=1 是第二密的点）：
    #   * higher = 所有比它更密的点（order 的前 rank 个）
    #   * d      = 它到这些更密点的距离
    #   * 取最小距离 = delta（离最近的更密点多远）
    #   * 那个最近的更密点就是它的 parent（后面传播标签要用）
    #   这是纯 Python 循环，N 大约几千，耗时几秒，可以接受。
    # ---------------------------------------------------------------
    for rank in range(1, len(order)):
        i = order[rank]              # 当前点的下标
        higher = order[:rank]        # 比它更密的所有点
        d = D[i, higher]             # 当前点到这些点的距离（向量）
        j = int(np.argmin(d))        # 最近的那个（在 higher 里的位置）
        delta[i] = float(d[j])       # 记录距离
        parent[i] = int(higher[j])   # 记录父点

    # ---------------------------------------------------------------
    # 第 5 步：峰分 gamma
    #   乘法而不是加法：rho 大但 delta 小（在密集区中间）不算峰；
    #                  delta 大但 rho 小（偏远孤点）也不算峰。
    # ---------------------------------------------------------------
    gamma = rho * delta
    return rho, delta, gamma, parent, dc


def choose_k(gamma, k=0, min_k=2, max_k=12):
    """决定分几簇（k = 峰的个数）。

    两种模式
        k > 0 : 直接用人工指定的 k（本文最终用 --k 7）
        k = 0 : 自动推断，规则见下

    自动推断的三步
        1. 把 gamma 从大到小排成 g
        2. 只考虑「显著高于平均」的峰：g > mean + 1*std
           （不这么限制，长尾里的小峰会让 k 变得很大）
        3. 在这个范围里找相邻两个峰之间「最大倍数落差」的位置：
           ratios[i] = g[i] / g[i+1]，取 argmax，k = 该位置
           直觉：真峰和噪声峰之间通常有一个台阶式断崖

    返回 (k, ratios, g)：k 是簇数，ratios 是落差数组（调试用），g 是排序后的 gamma
    """
    g = np.sort(gamma)[::-1]        # [::-1] 把升序倒成降序
    if k:
        return int(k), None, g      # 人工指定 k，直接返回

    thr = float(g.mean() + g.std())     # 显著性阈值：均值 + 1 个标准差
    cand = int((g > thr).sum())         # 有多少个峰超过阈值
    cand = max(min_k, min(cand, max_k))  # 夹在 [min_k, max_k] 之间

    # ---- 在合理范围内找最大乘性落差
    # hi 防止越界：最多看到第 max_k 个峰，且不超过数组长度-1
    hi = min(max_k + 1, len(g) - 1)
    # 相邻峰比值：g[i] / g[i+1]；+1e-12 防止除以 0
    ratios = g[min_k:hi] / (g[min_k + 1:hi + 1] + 1e-12)
    if len(ratios) == 0:
        return min_k, None, g
    # np.argmax 返回最大比值的位置（从 0 开始），+min_k 换成真正的 k
    best = int(np.argmax(ratios)) + min_k
    return best, ratios, g


def assign(parent, centers, rho):
    """按密度从高到低传播标签：每个点继承「父点」的标签。

    为什么能这样传播：父点一定比它更密，所以处理到它时，父点已经被分配过了。
    这就是「密度链」：点 -> 更密的邻居 -> 更更密的邻居 -> ... -> 某个峰。
    只有当一条链最终连到某个峰上，点才被分配；这正是密度峰聚类的核心思想。
    """
    lab = np.full(len(rho), -1, dtype=int)   # -1 表示「还没分配」
    for n, c in enumerate(centers):          # 先把每个峰自身标成自己的编号
        lab[c] = n

    for i in np.argsort(-rho):               # 从最密的点开始逐个处理
        if lab[i] == -1:                     # 只处理尚未分配的
            p = parent[i]                    # 它的父点（更密的邻居）
            # 正常情况父点已分配；极端情况（无父点）兜底归到 0 号簇
            lab[i] = lab[p] if p >= 0 else 0
    return lab


def kmedoids_refine(Xn, labels, iters=8):
    uniq = sorted(set(labels.tolist()))
    med = {}
    for c in uniq:
        idx = np.where(labels == c)[0]
        if not len(idx):
            continue
        sub = Xn[idx]
        med[c] = int(idx[np.argmax((sub @ sub.T).sum(1))])
    med = {c: m for c, m in med.items() if c >= 0}
    if not med:
        return labels, med
    mids = [med[c] for c in sorted(med)]
    for _ in range(iters):
        M = Xn[mids]
        newlab = np.asarray(mids)[np.argmax(Xn @ M.T, axis=1)]
        med2, mids2 = {}, []
        for c in sorted(set(newlab.tolist())):
            idx = np.where(newlab == c)[0]
            sub = Xn[idx]
            m = int(idx[np.argmax((sub @ sub.T).sum(1))])
            med2[c] = m
            mids2.append(m)
        labels, med, mids = newlab, med2, mids2
    order = sorted(med)
    mapping = {c: n for n, c in enumerate(order)}
    labels = np.array([mapping[l] for l in labels.tolist()], dtype=int)
    med = {mapping[c]: m for c, m in med.items()}
    return labels, med


def extract_forms(text):
    """Real word-forms of the target: jieba tokens + 绷+1/2 following chars."""
    out = []
    for w in jieba.lcut(text):
        if C.TARGET_CHAR in w:
            out.append(w)
    for i, ch in enumerate(text):
        if ch != C.TARGET_CHAR:
            continue
        for L in (1, 2):
            if i + L + 1 <= len(text):
                out.append(text[i:i + L + 1])
    return out


def top_forms(texts, topn=8, min_count=5):
    cnt = collections.Counter()
    for t in texts:
        cnt.update(extract_forms(t))
    items = [(g, c) for g, c in cnt.items()
             if c >= min_count and g != C.TARGET_CHAR and len(g) >= 2]
    items.sort(key=lambda x: (-x[1], -len(x[0])))
    picked, seen = [], []
    for g, c in items:
        if any(g in s for s in seen):      # keep the more specific longer form
            continue
        picked.append((g, c))
        seen.append(g)
        if len(picked) >= topn:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="0 = derive from gamma")
    ap.add_argument("--dc-pct", type=float, default=2.0)
    ap.add_argument("--form", default="", help="restrict to items containing this surface form")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--tag", default="pooled", help="suffix for outputs")
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()

    global OUT_JSONL
    def T(name):
        stem, ext = os.path.splitext(name)
        return os.path.join(C.TABLES, f"{stem}__{a.tag}{ext}")
    OUT_JSONL = os.path.join(C.PROCESSED, f"beng_clustered__{a.tag}.jsonl")

    z = np.load(os.path.join(EMB, "centered.npz"))
    X, ids = z["X"].astype(np.float32), list(z["ids"])
    rows = {r["clean_id"]: r for r in rj(CLEAN)}
    N, D = X.shape
    print(f"items={N} dim={D}")

    # optional restriction: cluster WITHIN one surface form, so that the
    # clusters cannot simply reproduce the surface distinction.
    if a.form or a.min_chars:
        keep = []
        for i, cid in enumerate(ids):
            ok = True
            if a.form:
                ok = ok and (a.form in rows[cid].get("text_norm", ""))
            if a.min_chars:
                ok = ok and (rows[cid].get("n_chars_norm", 0) >= a.min_chars)
            keep.append(ok)
        keep = np.array(keep)
        print(f"restriction: form={a.form!r} min_chars={a.min_chars} -> "
              f"{int(keep.sum())} of {N} items")
        ids = [cid for cid, k in zip(ids, keep) if k]
        X = X[keep]
        N = len(ids)
        rows = {c: rows[c] for c in ids}

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    off = S[np.triu_indices_from(S, 1)]
    print(f"cosine: mean={off.mean():.4f} p99={np.percentile(off, 99):.4f}")

    rho, delta, gamma, parent, dc = density_peaks(S, a.dc_pct)
    print(f"dc={dc:.4f}  rho=[{rho.min():.3f},{rho.max():.3f}]  "
          f"delta max={delta.max():.4f}")

    k, ratios, gsorted = choose_k(gamma, a.k)
    print(f"\nchosen k = {k}")
    print("top-15 gamma:", np.round(gsorted[:15], 3).tolist())

    centers = np.argsort(-gamma)[:k]
    labels = assign(parent, centers, rho)
    print(f"cluster sizes after propagation: "
          f"{dict(sorted(collections.Counter(labels.tolist()).items()))}")

    labels, medoids = kmedoids_refine(Xn, labels)
    sizes = collections.Counter(labels.tolist())
    print(f"\nclusters after medoid refinement: {len(sizes)}")
    for c in sorted(sizes):
        m = medoids.get(c)
        print(f"  cluster {c}: {sizes[c]:5d} | {rows[ids[m]]['text_norm'][:60]}")

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(Xn, labels, metric="cosine",
                           sample_size=min(1500, N),
                           random_state=C.RANDOM_SEED)
    print(f"silhouette(cosine, k={len(sizes)}): {sil:.4f}")

    hints = [rows[i].get("sense_hint", "unknown") for i in ids]
    tab = collections.defaultdict(collections.Counter)
    for lab, h in zip(labels.tolist(), hints):
        tab[lab][h] += 1
    hint_names = sorted(set(hints))

    import csv
    p1 = T("cluster_sizes.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "share", "medoid_clean_id",
                    "medoid_text"])
        for c in sorted(sizes):
            m = medoids.get(c)
            w.writerow([c, sizes[c], round(sizes[c] / N, 4),
                        ids[m] if m is not None else "",
                        rows[ids[m]]["text_norm"][:150] if m is not None else ""])

    p2 = T("cluster_x_sensehint.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size"] + hint_names)
        for c in sorted(sizes):
            w.writerow([c, sizes[c]] + [tab[c][h] for h in hint_names])

    p3 = T("cluster_top_forms.csv")
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "rank", "form", "count", "per_1000"])
        for c in sorted(sizes):
            idx = np.where(labels == c)[0]
            texts = [rows[ids[i]]["text_norm"] for i in idx]
            for r, (g, n) in enumerate(top_forms(texts), 1):
                w.writerow([c, sizes[c], r, g, n,
                            round(1000 * n / max(1, sizes[c]), 1)])

    p4 = T("cdpc_diagnostics.csv")
    with open(p4, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clean_id", "cluster", "rho", "delta", "gamma",
                    "is_centre", "text"])
        cs = set(centers.tolist())
        for i, cid in enumerate(ids):
            w.writerow([cid, int(labels[i]), round(float(rho[i]), 4),
                        round(float(delta[i]), 4), round(float(gamma[i]), 4),
                        1 if i in cs else 0, rows[cid]["text_norm"][:80]])

    p5 = T("cdpc_gamma_ranking.csv")
    with open(p5, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "clean_id", "rho", "delta", "gamma", "text"])
        for r, i in enumerate(np.argsort(-gamma)[:40], 1):
            w.writerow([r, ids[i], round(float(rho[i]), 4),
                        round(float(delta[i]), 4), round(float(gamma[i]), 4),
                        rows[ids[i]]["text_norm"][:70]])

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            r = dict(rows[cid])
            r["cluster_id"] = int(labels[i])
            r["is_medoid"] = 1 if medoids.get(int(labels[i])) == i else 0
            r["cdpc_rho"] = round(float(rho[i]), 5)
            r["cdpc_delta"] = round(float(delta[i]), 5)
            r["cdpc_gamma"] = round(float(gamma[i]), 5)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n--- contingency: cluster x surface hint ---")
    print("  cl   size  " + " ".join(f"{h[:13]:>13}" for h in hint_names))
    for c in sorted(sizes):
        print(f"  {c:<4}{sizes[c]:>6}  " +
              " ".join(f"{tab[c][h]:>13d}" for h in hint_names))

    print("\n--- top 绷-forms per cluster (jieba words + 绷+X windows) ---")
    for c in sorted(sizes):
        idx = np.where(labels == c)[0]
        texts = [rows[ids[i]]["text_norm"] for i in idx]
        print(f"  cluster {c} (n={sizes[c]}): "
              + ", ".join(f"{g}×{n}" for g, n in top_forms(texts, 8)))

    print("\n--- example sentences per cluster ---")
    for c_ in sorted(sizes):
        idx = np.where(labels == c_)[0]
        idx = idx[np.argsort(-np.array([len(rows[ids[i]]["text_norm"]) for i in idx]))]
        print(f"  [cluster {c_}] n={sizes[c_]}")
        for i in idx[:a.examples]:
            print("     ", rows[ids[i]]["text_norm"][:78])

    for p in (p1, p2, p3, p4, p5, OUT_JSONL):
        print("->", p)


if __name__ == "__main__":
    main()
