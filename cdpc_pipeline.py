# -*- coding: utf-8 -*-
"""CDPC clustering (cosine density peaks) + six-category typology -> result4.

What changed vs the previous solution
  * The clustering METHOD is now CDPC — Density Peak Clustering with COSINE
    similarity, exactly as requested on 2026-10-21:
        distance  d = 1 - cos(x_i, x_j)
        density   rho_i = sum_j exp(-(d_ij/dc)^2)      (dc = 2nd pct of d)
        delta_i  = min d(i, j) over points with higher rho
        gamma    = rho * delta                          (peak score)
    Cluster centres are the top-gamma points (k from the largest multiplicative
    gamma drop, or --k), and the rest is assigned by density-chain propagation,
    then refined with cosine k-medoids. No Euclidean distance is used anywhere
    in the clustering, and NO dimensionality reduction is used for clustering.
  * PCA is applied ONLY afterwards, to draw the 2-D figure (variance explained
    is printed and must go in the caption).
  * The core CDPC maths is reused from src/analyze/cluster_cdpc.py (same
    implementation, already unit-checked against its v1 bugs) rather than
    re-implemented, so there is a single source of truth.

Cluster -> category
  Item level: the surface form decides (塌房 -> 塌房义, 塌方/塌陷/坍塌/倒塌/塌了
  -> 物理坍塌义, 塌鼻梁 -> 下垂凹陷义, 一塌糊涂 -> 成语义, 塌实 -> 安定义,
  其他 -> 非义项用法 unless a built-structure frame is named), with any
  adjudicated rare-sense list (data/processed/<slug>/..._adjudication.jsonl)
  overriding everything. These names are PROVISIONAL for the first pass.
  Cluster level: a cluster takes the majority category of its items; its purity
  is reported, so a mixed cluster cannot silently claim a clean sense.

Inputs   data/processed/<slug>/<slug>_clean.jsonl, data/processed/emb2/masked/
Outputs  data/processed/<slug>/<slug>_cdpc.jsonl
         results/<slug>/result4/cdpc_*.csv, typology_*.csv, cdpc_fig*.png/.pdf
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
from cluster_cdpc import density_peaks, choose_k, assign, kmedoids_refine
from jinbeng_cue import load_adjudication

sys.stdout.reconfigure(encoding="utf-8")

CLEAN = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_clean.jsonl")
OUT_JSONL = os.path.join(C.PROCESSED, f"{C.WORD_SLUG}_cdpc.jsonl")
LAYERS = (9, 10, 11, 12)
# ===== 拉 (lā) revision, 2026-10-31 =====================================
# 拉's network layer is NOT one metaphor (as 塌房 was) but a family of separate
# compounds, so the coarse scheme separates "network-derived" from the concrete
# pull/recruit senses, and the fine scheme splits the four productive compounds
# that have different onset dates (拉胯 ~2020, 拉满 ~2019, 拉踩 ~2019, 拉黑 ~2008).
CAT_ORDER = ["拖拽义项", "网络派生义项", "招集义项", "演奏义项", "排泄义项",
             "方言义项", "非义项/其他"]
CAT_COLOR = {"拖拽义项": "#8C564B", "网络派生义项": "#D62728",
             "招集义项": "#17BECF", "演奏义项": "#2CA02C",
             "排泄义项": "#BCBD22", "方言义项": "#E377C2",
             "非义项/其他": "#D9D9D9"}

CAT_ORDER_FINE = ["物理·拉面", "物理·拖拽", "网络·拉胯", "网络·拉满",
                  "网络·拉踩", "网络·拉黑", "招集·拉票", "演奏义项",
                  "排泄义项", "方言惯用", "非义项/其他"]
# 2026-10-31 (user decision): the five social/network categories get five
# DISTINCT hues (red / orange / purple / blue / cyan) instead of neighbouring
# tabs of the same colour scale -- the previous 拉踩(深紫) vs 拉黑(浅紫) pair was
# unreadable on a scatter of 5.6k translucent dots. The residual class is a pale
# grey on purpose: it is 37% of the data and must recede into the background.
CAT_COLOR_FINE = {"物理·拉面": "#8C564B",   # brown
                  "物理·拖拽": "#C49C94",   # light brown
                  "网络·拉胯": "#D62728",   # red
                  "网络·拉满": "#FF7F0E",   # orange
                  "网络·拉踩": "#9467BD",   # purple
                  "网络·拉黑": "#1F77B4",   # blue
                  "招集·拉票": "#17BECF",   # cyan
                  "演奏义项": "#2CA02C",    # green
                  "排泄义项": "#BCBD22",    # olive
                  "方言惯用": "#E377C2",    # pink
                  "非义项/其他": "#D9D9D9"}  # pale grey
# merged grain is kept only so old command lines still run; for 拉 it is an
# alias of the coarse scheme (no pairwise merge decision has been made yet).
CAT_ORDER_M = CAT_ORDER
CAT_COLOR_M = CAT_COLOR
# switched by --cat-grain in main()
GRAIN = "coarse"

# 拉's concrete frames: things one physically pulls. Used ONLY for a bare 拉
# with no fixed form (拉他一把 / 拉过来 / 把门拉开), which is the biggest bucket
# in this corpus (unknown hint ~48%).
PHYSICAL_FRAME = (r"车|车门|车窗|门|窗|链|绳|箱|包|手|被子|窗帘|行李|缰绳|网|"
                  r"鱼|杆|弓|抽屉|拉环|手刹|刹车|拉手|缆|绳子|桌|椅|沙发|"
                  r"箱子|袋子|衣服|袖子|裤|鞋|闸|开关|电线|管子|管道")

# 2026-10-31: the 拉-analogue of 塌's FANDOM_CUE. 拉 used as a bare ADJECTIVE
# meaning "bad" (太拉了 / 这么拉 / 拉中拉 / 笑拉了) carries no fixed compound,
# so it needs a cue; otherwise it lands in the residual bucket and the network
# layer looks smaller than it is.
NET_EVAL_CUE = re.compile(
    r"(?:太|这么|那么|真|好|特别|有点|挺|属实|越来越|确实|真的)拉了?|"
    r"拉中拉|笑拉了|拉一裤|拉得一批|拉的不行|拉了胯")

# 2026-10-31 (user decision): 拉完了 counts as the evaluative sense, EXCEPT when
# it is a literal "finish V-ing" with an object right after it: 拉完了屎 /
# 拉完稀 / 拉完二胡 / 拉完琴 / 拉完面 / 拉完冲 / 拉完擦. Those ~16 items are
# excreting/playing/noodle talk, not "X is bad".
LAWAN_LITERAL = re.compile(r"拉完了?(?:屎|稀|二胡|琴|面|冲|擦)")

FORMS_LA = [("拉胯", ["拉胯", "拉跨", "拉垮", "拉完了"]),
            ("拉满", ["拉满"]),
            ("拉踩", ["拉踩", "拉爆"]),
            ("拉黑", ["拉黑"]),
            ("拉票", ["拉票", "拉群", "拉赞助", "拉关系", "拉帮结派", "拉人"]),
            ("拉面", ["拉面"]),
            ("演奏", ["拉二胡", "拉小提琴", "拉琴", "拉弦"]),
            ("排泄", ["拉屎", "拉肚子", "拉稀"]),
            ("方言", ["拉家常", "拉呱", "拉话"]),
            ("拖拽", ["拉车", "拉手", "拉链", "拉杆", "拉住", "拉开", "拉走",
                     "拉回", "拉起", "拉紧", "拉绳", "拉扯", "拉上来",
                     "拉下来", "拉过来", "拉出来"])]



def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def form_label(t):
    for f, pats in FORMS_LA:
        for p in pats:
            if p == "拉完了":
                # user decision 2026-10-31: literal "finish V-ing" stays literal
                if "拉完了" in t and not LAWAN_LITERAL.search(t):
                    return f
                continue
            if p in t:
                return f
    return "其他"


def item_category(text, clean_id, jinbeng):
    """Item-level category for 拉. PROVISIONAL: labels exist to read clusters."""
    t = text or ""
    form = form_label(t)
    net = form in ("拉胯", "拉满", "拉踩", "拉黑")
    if GRAIN == "fine":
        if form == "拉胯":
            return "网络·拉胯"
        if form == "拉满":
            return "网络·拉满"
        if form == "拉踩":
            return "网络·拉踩"
        if form == "拉黑":
            return "网络·拉黑"
        if form == "拉票":
            return "招集·拉票"
        if form == "拉面":
            return "物理·拉面"
        if form == "演奏":
            return "演奏义项"
        if form == "排泄":
            return "排泄义项"
        if form == "方言":
            return "方言惯用"
        if form == "拖拽":
            return "物理·拖拽"
        # bare 拉: adjective use first (它常与物体框架词共现，如「车太拉了」)，
        # then the pull frame, then the residual
        if NET_EVAL_CUE.search(t):
            return "网络·拉胯"
        if re.search(PHYSICAL_FRAME, t):
            return "物理·拖拽"
        return "非义项/其他"
    # coarse
    if net:
        return "网络派生义项"
    if form == "拉票":
        return "招集义项"
    if form in ("拉面", "拖拽"):
        return "拖拽义项"
    if form == "演奏":
        return "演奏义项"
    if form == "排泄":
        return "排泄义项"
    if form == "方言":
        return "方言义项"
    if NET_EVAL_CUE.search(t):
        return "网络派生义项"
    if re.search(PHYSICAL_FRAME, t):
        return "拖拽义项"
    return "非义项/其他"


def load_masked(root=None):
    # ------------------------------------------------------------------
    # 这个函数负责「把向量读进来并变成可聚类的样子」，分三步：
    #   ① 读 shard：向量当初是按每 256 条一个 .npz 文件存的（叫 shard/分片），
    #      index.jsonl 里记着每条 clean_id 存在哪个分片的第几行。
    #   ② 取 last-4 层平均：向量形状是 (句子数, 层数=13, 维度=768)，
    #      我们只取第 9,10,11,12 层（0 层是词嵌入，含大量字形噪声），求平均。
    #   ③ 去各向异性 + 归一化：减均值、去掉第一主成分、L2 归一化。
    # 返回 (Z, ids)：Z 是 (N,768) 的 numpy 数组，ids 是与之对齐的 clean_id 列表。
    #
    # 2026-10-31: root is a parameter now. The SAME pipeline can read the
    # target-span representation (emb2/target) so the two representations can be
    # compared under an identical clustering protocol -- that comparison is the
    # only way to tell whether the diffuse cloud is caused by masking.
    # ------------------------------------------------------------------
    root = root or C.EMB_MASK_DIR
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
    # xs 原来是 (N,13,768) 的列表，np.stack 把它们摞成一个三维数组。
    # [:, LAYERS, :] 表示「所有句子，只要第 9/10/11/12 层，所有维度」
    # .mean(axis=1) 在第 1 维（层）上求平均 -> 变成 (N,768)
    # astype(np.float32)：float16 省空间但容易溢出，算之前转成 float32
    X = np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1)
    # centring + ABTT(k=1) fitted on this item set, then L2-normalise: the
    # space in which cosine is meaningful (anisotropy removed)
    # 第 1 步去中心：算每一维的均值 mu，形状 (1,768)（keepdims 保证还是一行）
    # 每一个向量都减掉这个均值 -> 数据以原点为中心，余弦相似度才有意义。
    # （不减均值时，几乎所有文本向量都在同一个方向上，余弦都在 0.7 以上。）
    mu = X.mean(axis=0, keepdims=True)
    Z = X - mu
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    comp = Vt[:1]
    # 第 2 步 ABTT（All-but-the-Top）：去掉方差最大的那 1 个方向。
    #   Z @ comp.T  : 把每个向量投影到第一主成分方向，得到 (N,1) 的坐标
    #   (Z @ comp.T) @ comp : 再乘回方向向量，得到「它在那个方向上的分量」
    #   Z - 分量    : 减掉，就等于把整个数据集沿这个方向压平
    # 目的：文本向量的第一主成分常常只是一个「高频虚词」方向，去掉它以后
    # 剩下的方向才更能体现语义差异。
    Z = Z - (Z @ comp.T) @ comp
    # 第 3 步 L2 归一化：把每个向量缩放到长度 1。
    #   np.linalg.norm(..., axis=1, keepdims=True) 逐行求长度，形状 (N,1)
    #   +1e-12 防止长度为 0 时除以 0
    # 归一化后，两个向量点积 = 余弦相似度（矩阵乘法 Z @ Z.T 就是相似度矩阵）。
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="0 = derive k from gamma")
    ap.add_argument("--dc-pct", type=float, default=2.0)
    # 2026-10-31: for 塌 the density basins track lexicalised forms, so a
    # cluster-majority label erases the small senses (成语 96 / 下垂 92 items).
    # --label-mode item keeps the category at item level and uses the clusters
    # as descriptive structure only; 'cluster' is the beng-era behaviour.
    ap.add_argument("--label-mode", choices=["item", "cluster"], default="item")
    ap.add_argument("--cat-grain", choices=["coarse", "fine", "merged"],
                    default="coarse",
                    help="coarse: 物理义项 is one class; fine: 倒塌/坍塌/塌方/"
                         "塌陷/塌了 separate; merged: round-5 scheme")
    # 2026-10-31: representation switch. 'masked' is the default (target char
    # replaced by [MASK]); 'target' reads the untouched span vector. Running the
    # identical clustering on both is what tells us whether the diffuse cloud is
    # an artefact of masking or a property of the corpus.
    ap.add_argument("--rep", choices=["masked", "target"], default="masked")
    a = ap.parse_args()

    if (C.WORD, C.WORD_SLUG) != ("拉", "la"):
        ap.error("This CDPC pipeline currently contains 拉-specific categories. "
                 "Set LING_WORD=拉 and LING_SLUG=la before running it.")

    # the grain changes both the label set and the palette
    global GRAIN, CAT_ORDER, CAT_COLOR
    if a.cat_grain == "fine":
        GRAIN = "fine"
        CAT_ORDER, CAT_COLOR = CAT_ORDER_FINE, CAT_COLOR_FINE
    elif a.cat_grain == "merged":
        GRAIN = "merged"
        CAT_ORDER, CAT_COLOR = CAT_ORDER_M, CAT_COLOR_M
    print(f"cat grain = {a.cat_grain}  ({len(CAT_ORDER)} classes)")

    Z, ids = load_masked(C.EMB_DIR if a.rep == "target" else C.EMB_MASK_DIR)
    rows = {r["clean_id"]: r for r in rj(CLEAN)}
    jinbeng = load_adjudication()
    N = len(ids)
    print(f"items={N} dim={Z.shape[1]}  representation={a.rep} last-4 mean")
    print(f"jinbeng adjudicated keeps: {len(jinbeng)}")

    # ---- 余弦相似度矩阵 S：(N,N)，S[i,j] = 句子 i 与句子 j 的余弦相似度。
    # Z 已归一化，所以一次矩阵乘法就够；对角线 S[i,i]=1（自己和自己）。
    # N=4488 时这个矩阵约 2000 万个数，占用约 80MB，现代机器没问题。
    S = Z @ Z.T
    # 取值看看分布：np.triu_indices_from(S, 1) 取「上三角、不含对角线」的下标，
    # 也就是所有 i<j 的两两组合（避免重复统计，也避免对角线上的 1）。
    off = S[np.triu_indices_from(S, 1)]
    print(f"cosine similarity: mean={off.mean():.4f} p1={np.percentile(off,1):.4f} "
          f"p99={np.percentile(off,99):.4f}")

    # ---- 核心一步：算密度 rho、峰高 delta、峰分 gamma（算法细节见 cluster_cdpc.py）
    # 注意传进去的是「相似度」，函数内部自己转成距离 d = 1 - cos。
    rho, delta, gamma, parent, dc = density_peaks(S, a.dc_pct)
    print(f"dc={dc:.4f}  rho range=[{rho.min():.3f},{rho.max():.3f}]")
    # ---- 定簇数 k：--k 给了就用给的；没给就按 gamma 的断层自动推断
    # _ratios 是自动推断时的相邻倍数落差（这次没用上，所以叫 _ratios）
    k, _ratios, gsorted = choose_k(gamma, a.k)
    print(f"chosen k = {k}   (top-12 gamma: "
          f"{np.round(gsorted[:12], 3).tolist()})")

    # 峰 = gamma 最大的 k 个点的「下标」（不是 gamma 值本身）
    centers = np.argsort(-gamma)[:k]
    # 把其余点按密度链分到最近的峰：得到每条句子的簇编号
    labels = assign(parent, centers, rho)
    print("sizes before refinement:",
          dict(sorted(collections.Counter(labels.tolist()).items())))
    # 精修：用「每簇里最居中的真实句子」当簇心反复重分，直到编号稳定。
    # medoids 是 {簇号: 该簇中心句子的下标}，后面画图要用它标出峰的位置。
    labels, medoids = kmedoids_refine(Z, labels)
    sizes = collections.Counter(labels.tolist())
    print(f"clusters after medoid refinement: {len(sizes)}")

    from sklearn.metrics import silhouette_score
    # 轮廓系数：衡量「同类够近、异类够远」，范围 -1..1，越大越好，
    # 但它偏向「少而紧」的分法，所以只用它做参考，不用它选 k。
    # sample_size 只抽 1500 个点算，避免 N 大时太慢。
    sil = silhouette_score(Z, labels, metric="cosine",
                           sample_size=min(1500, N), random_state=C.RANDOM_SEED)
    print(f"silhouette(cosine, k={len(sizes)}) = {sil:.4f}")

    # ---- item level categories, then cluster majorities
    # ---- 簇 -> 语义类别
    # 先给每条句子单独判类别（item 级）：按表面词形判，如 绷带->物理义、绷住->忍住义。
    # 人工判定过的稀有义项（如「紧绷义」）在这里就已经覆盖。
    cat_item = [item_category(rows[cid].get("text_norm") or "", cid, jinbeng)
                for cid in ids]
    ct = collections.defaultdict(collections.Counter)
    for lab, cat in zip(labels.tolist(), cat_item):
        ct[lab][cat] += 1
    cluster_cat = {c: ct[c].most_common(1)[0][0] for c in ct}
    purity = {c: ct[c].most_common(1)[0][1] / sum(ct[c].values()) for c in ct}

    print("\n=== cluster -> majority category (purity) ===")
    for c in sorted(sizes):
        forms = collections.Counter(
            form_label(rows[ids[i]]["text_norm"]) for i in np.where(labels == c)[0])
        top = ", ".join(f"{f}:{n}" for f, n in forms.most_common(3))
        print(f"  cluster {c:>2} n={sizes[c]:>5}  -> {cluster_cat[c]:8} "
              f"purity={purity[c]:.2f}  forms[{top}]")

    # 簇级决定最终类别：同一个簇里的句子共享该簇的「多数类别」。
    # 代价是少数派句子会被改类，所以每簇的纯度必须打出来。
    if a.label_mode == "cluster":
        cat = [cluster_cat[l] for l in labels.tolist()]
    else:
        # item 级：类别只由该条自己的词形/框架决定，簇完全不参与投票。
        # 这样 成语义项(96)/下垂凹陷义项(92) 这类小类不会被并入大盆地。
        cat = list(cat_item)
    # the hand-adjudicated rare-sense items survive either mode: the category is
    # defined by the sense and there may be only a handful of them.
    n_jb = 0
    for i, cid in enumerate(ids):
        if cid in jinbeng:
            cat[i] = "安定义项"
            n_jb += 1
    print(f"label mode = {a.label_mode}; adjudicated rare-sense keeps: {n_jb}")

    # cluster x category crosstab: keeps the mixing visible even when labels are
    # assigned at item level (this is what a mixed basin must be reported with).
    p0 = os.path.join(C.TABLES, "cdpc_cluster_x_category.csv")
    with open(p0, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size"] + CAT_ORDER + ["majority", "purity"])
        for c in sorted(sizes):
            idxs = np.where(labels == c)[0]
            w.writerow([c, sizes[c]] + [sum(1 for i in idxs if cat[i] == cc)
                                        for cc in CAT_ORDER]
                       + [cluster_cat[c], round(purity[c], 4)])

    # ---- PCA only for the figure
    # ---- PCA 降维，**只为了画二维图**，聚类已经在 768 维做完了。
    # 这一步和 load_masked 里的 SVD 是两回事：这里取前 2 个主成分当 x/y 坐标。
    Zm = Z - Z.mean(axis=0, keepdims=True)
    _, _, V2 = np.linalg.svd(Zm, full_matrices=False)
    # 把每个 768 维向量投影到前 2 个主成分上，得到 (N,2) 的坐标
    coords = Zm @ V2[:2].T
    # 二维解释了原空间多少方差：这个数要写进图注（本次约 28.9%），
    # 提醒读者「这张图只是投影，不是聚类空间本身」。
    var2 = float(np.var(coords, axis=0).sum() / np.var(Z, axis=0).sum())
    print(f"PCA(2-D) explains {100*var2:.1f}% of the variance of the clustering "
          f"space (used ONLY for the figure)")

    os.makedirs(C.TABLES, exist_ok=True)

    # ---- tables
    p1 = os.path.join(C.TABLES, "cdpc_cluster_sizes.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "share", "category", "purity",
                    "medoid_clean_id", "medoid_text"])
        for c in sorted(sizes):
            m = medoids.get(c)
            w.writerow([c, sizes[c], round(sizes[c] / N, 4), cluster_cat[c],
                        round(purity[c], 4), ids[m] if m is not None else "",
                        (rows[ids[m]]["text_norm"][:150] if m is not None else "")])

    p2 = os.path.join(C.TABLES, "cdpc_cluster_x_form.csv")
    FORMS = [f for f, _ in FORMS_LA] + ["其他"]
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size"] + FORMS + ["category", "purity"])
        for c in sorted(sizes):
            idx = np.where(labels == c)[0]
            cnt = collections.Counter(form_label(rows[ids[i]]["text_norm"])
                                      for i in idx)
            w.writerow([c, sizes[c]] + [cnt[f] for f in FORMS]
                       + [cluster_cat[c], round(purity[c], 4)])

    p3 = os.path.join(C.TABLES, "cdpc_diagnostics.csv")
    cen = set(centers.tolist())
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clean_id", "cluster", "category", "rho", "delta", "gamma",
                    "is_centre", "year", "text"])
        for i, cid in enumerate(ids):
            r = rows[cid]
            w.writerow([cid, int(labels[i]), cat[i], round(float(rho[i]), 5),
                        round(float(delta[i]), 5), round(float(gamma[i]), 5),
                        1 if i in cen else 0,
                        (r.get("ctime_iso") or "")[:4],
                        (r.get("text_norm") or "")[:100]])

    p4 = os.path.join(C.TABLES, "cdpc_top_gamma.csv")
    with open(p4, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "clean_id", "cluster", "rho", "delta", "gamma",
                    "is_centre", "text"])
        for r, i in enumerate(np.argsort(-gamma)[:40], 1):
            w.writerow([r, ids[i], int(labels[i]), round(float(rho[i]), 5),
                        round(float(delta[i]), 5), round(float(gamma[i]), 5),
                        1 if i in cen else 0,
                        (rows[ids[i]].get("text_norm") or "")[:80]])

    # ---- typology tables (same schema as the previous version, CDPC labels)
    scored = [(i, cat[i]) for i in range(N)
              if not rows[ids[i]].get("excluded_from_typology")]
    cnt = collections.Counter(c for _i, c in scored)
    total = len(scored)
    p5 = os.path.join(C.TABLES, "typology_sizes.csv")
    with open(p5, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "share"])
        for c in CAT_ORDER:
            w.writerow([c, cnt[c], round(cnt[c] / total, 4)])
        w.writerow(["TOTAL", total, 1.0])

    ready = [(i, cat[i]) for i, c in scored
             if rows[ids[i]].get("is_canonical") == 1
             and rows[ids[i]].get("context_sufficiency") == 1]
    tab = collections.defaultdict(collections.Counter)
    for i, c in ready:
        y = (rows[ids[i]].get("ctime_iso") or "")[:4]
        if y:
            tab[y][c] += 1
    p6 = os.path.join(C.TABLES, "typology_by_year.csv")
    with open(p6, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["year", "n"] + CAT_ORDER + [f"share_{c}" for c in CAT_ORDER])
        for y in sorted(tab):
            n = sum(tab[y].values())
            w.writerow([y, n] + [tab[y][c] for c in CAT_ORDER]
                       + [round(tab[y][c] / n, 4) for c in CAT_ORDER])

    rng = np.random.default_rng(C.RANDOM_SEED)
    p7 = os.path.join(C.TABLES, "typology_examples.csv")
    with open(p7, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category", "clean_id", "ctime", "n_chars", "cluster",
                    "form_group", "text"])
        for c in CAT_ORDER:
            pool = [i for i, cc in scored if cc == c
                    and rows[ids[i]].get("context_sufficiency") == 1
                    and 12 <= len(rows[ids[i]].get("text_norm") or "") <= 80]
            if not pool:
                continue
            pick = rng.choice(len(pool), size=min(12, len(pool)), replace=False)
            for j in pick:
                i = pool[int(j)]
                r = rows[ids[i]]
                w.writerow([c, ids[i], (r.get("ctime_iso") or "")[:10],
                            len(r.get("text_norm") or ""), int(labels[i]),
                            form_label(r.get("text_norm") or ""),
                            r.get("text_norm")])

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            r = dict(rows[cid])
            r["cdpc_cluster"] = int(labels[i])
            r["cdpc_category"] = cat[i]
            r["cdpc_is_medoid"] = 1 if medoids.get(int(labels[i])) == i else 0
            r["cdpc_rho"] = round(float(rho[i]), 5)
            r["cdpc_delta"] = round(float(delta[i]), 5)
            r["cdpc_gamma"] = round(float(gamma[i]), 5)
            r["cdpc_is_centre"] = 1 if i in cen else 0
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    # ---- 画图
    # keep 是布尔数组：被标为「极零散义项」的条目不在图上着色（它们也不进分母）
    keep = np.array([not rows[cid].get("excluded_from_typology") for cid in ids])
    ycat = np.array(cat)
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    # 2026-10-31: draw the residual class FIRST and paler, so the coloured
    # senses are not buried under 37% of pale dots; then draw the real senses on
    # top with a slightly larger marker.
    plot_order = (["非义项/其他"] if "非义项/其他" in CAT_ORDER else []) + \
                 [c for c in CAT_ORDER if c != "非义项/其他"]
    for c in plot_order:
        m = (ycat == c) & keep
        if not m.any():
            continue
        res = (c == "非义项/其他")
        ax.scatter(coords[m, 0], coords[m, 1],
                   s=4 if res else 9,
                   alpha=0.22 if res else 0.62,
                   color=CAT_COLOR[c], edgecolors="none",
                   label=f"{c} (n={int(m.sum())})")
    # 把每个簇的中心句画成黑色 X，并标 C0/C1/… 方便和表格对照
    mids = [medoids[c] for c in sorted(medoids)]
    ax.scatter(coords[mids, 0], coords[mids, 1], s=120, marker="X",
               c="#222222", edgecolors="white", linewidths=0.8, zorder=5)
    for c in sorted(medoids):
        i = medoids[c]
        ax.annotate(f"C{c}", (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(7, 6), fontsize=8)
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title(f"「{C.TARGET_WORD}」CDPC 密度峰聚类（k={len(sizes)}，余弦相似度）")
    # custom legend handles: the raw scatter sizes would make the legend useless
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      color=CAT_COLOR[c],
                      label=f"{c} (n={int(((ycat == c) & keep).sum())})")
               for c in CAT_ORDER if ((ycat == c) & keep).any()]
    handles.append(Line2D([], [], marker="X", linestyle="", markersize=8,
                          color="#222222", label="密度峰中心（medoid）"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
    ax.text(0.01, 0.01,
            f"注：聚类在 768 维余弦空间完成，此 PCA 二维投影仅解释该空间方差的 "
            f"{100*var2:.1f}%，仅用于展示。",
            transform=ax.transAxes, fontsize=7.5, color="#555555", va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_fig1_clusters.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    years = sorted(tab)
    ns = [sum(tab[y].values()) for y in years]
    shares = {c: [tab[y][c] / sum(tab[y].values()) for y in years]
              for c in CAT_ORDER}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.stackplot(years, [shares[c] for c in CAT_ORDER], labels=CAT_ORDER,
                 colors=[CAT_COLOR[c] for c in CAT_ORDER], alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_xlabel("年份")
    ax.set_ylabel("占比")
    ax.set_title(f"「{C.TARGET_WORD}」{len(CAT_ORDER)}类用法的历时构成"
                 f"（CDPC 聚类，百分比堆叠）")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_fig4_stacked.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.2),
                                 height_ratios=[3, 1], sharex=True)
    for c in CAT_ORDER:
        a1.plot(years, shares[c], marker="o", ms=3.5, color=CAT_COLOR[c], label=c)
    a1.set_ylabel("占比")
    a1.set_ylim(0, 1)
    a1.set_title(f"「{C.TARGET_WORD}」各用法占比的历时变化（CDPC 聚类）")
    a1.legend(fontsize=8, frameon=False, ncol=3)
    a2.bar(years, ns, color="#999999")
    a2.set_ylabel("样本量 n")
    a2.set_xlabel("年份")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        q = os.path.join(C.FIGURES, f"cdpc_fig4b_line.{ext}")
        fig.savefig(q, dpi=300, bbox_inches="tight")
        print("->", q)

    print(f"\nanalysis-ready items: {len(ready)}   typology N: {total}")
    print(f"{'year':>6}{'n':>7}" + "".join(f"{c:>10}" for c in CAT_ORDER))
    for y in years:
        n = sum(tab[y].values())
        print(f"{y:>6}{n:>7}"
              + "".join(f"{100*tab[y][c]/n:>9.1f}%" for c in CAT_ORDER))

    for p in (p1, p2, p3, p4, p5, p6, p7, OUT_JSONL):
        print("->", p)
    # keep a per-round copy as well: <slug>_cdpc.jsonl is always "current", so
    # without this a later round would silently overwrite the previous round's
    # item-level labels.
    import shutil
    round_copy = os.path.join(C.PROCESSED,
                              f"{C.WORD_SLUG}_cdpc_{C.RESULT_SET}.jsonl")
    shutil.copyfile(OUT_JSONL, round_copy)
    print("->", round_copy)


if __name__ == "__main__":
    main()
