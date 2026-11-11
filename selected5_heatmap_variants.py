# -*- coding: utf-8 -*-
"""Two versions of the five-sentence similarity heatmap.

v1 (kept as-is):  position 4 = the "绷住挑战" template sentence
v2 (new)       :  position 4 = a NON-template 忍住义 sentence

Keeping both is deliberate: v1 shows what happens when the prototype is a
formulaic meme line, v2 shows the same analysis with a prototypical sentence.
The difference between them is itself reportable -- it shows how much a single
prototype choice moves a 5x5 matrix.

Everything else (sentences 1,2,3,5, the stored mu/PCA transform, the centring
protocol) is identical between the two, so the comparison isolates one variable.

Outputs
  results/figures/fig2a_selected5_v1_template4.png/.pdf   (copy of the original)
  results/figures/fig2b_selected5_v2_nontemplate4.png/.pdf
  results/tables/selected5_cosine_v1_v2.csv
"""
import os
import sys
import json
import shutil
import collections

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

TYPOLOGY = os.path.join(C.PROCESSED, "beng_typology.jsonl")
EMB = C.EMB_DIR
PARAMS = os.path.join(EMB, "center_params.npz")
LAYERS = (9, 10, 11, 12)
WIN = 22

FIXED = [
    ("物理义", "要那么多绷带干嘛？缠成木乃伊么"),
    ("忍不住义", "一想到十年后互联网主力军的童年回忆是奶龙就绷不住了"),
    ("评价义", "要说真正难呆的还得是老美自己发的火星图片里能看到矿泉水瓶和苍蝇"),
]
# 评价义 sentence has 难绷, fix the typo above at runtime
FIXED[2] = ("评价义", "要说真正难绷的还得是老美自己发的火星图片里能看到矿泉水瓶和苍蝇")

S4_V1 = ("忍住义", "如果你连续78次绷住的话 我后面的780万就是你的了")
S4_V2 = ("忍住义", "有没有人跟我一样，本来是要绷住的，结果看了UP主一眼，破了")
S5 = ("忍住义", "你好，我是百万美元，如果你绷住，身后的野崩先生就是你的了")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_corpus():
    index = rj(os.path.join(EMB, "index.jsonl"))
    by = collections.defaultdict(list)
    for r in index:
        if r.get("status") == "ok":
            by[r["shard"]].append(r)
    xs, ids = [], []
    for si in sorted(by):
        emb = np.load(os.path.join(EMB, "shards", f"shard_{si:03d}.npz"))["emb"]
        for r in sorted(by[si], key=lambda z: z["row"]):
            xs.append(emb[r["row"]])
            ids.append(r["clean_id"])
    return np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1), ids


def transform(X, mu, comp):
    Z = X - mu
    if comp is not None and len(comp):
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def embed_target(s, tok, model):
    idx = s.find(C.TARGET_CHAR)
    if idx < 0:
        raise ValueError(s)
    enc = tok(s, return_tensors="pt", return_offsets_mapping=True,
              truncation=True, max_length=C.MAX_LEN)
    offs = enc.pop("offset_mapping")[0].tolist()
    span = [i for i, (a, b) in enumerate(offs) if a != b and a <= idx < b]
    with torch.no_grad():
        o = model(**enc, output_hidden_states=True)
    hs = torch.stack(o.hidden_states, dim=0)[:, 0, :, :]
    return hs[list(LAYERS)][:, span, :].mean(dim=1).mean(dim=0).cpu().float().numpy()


def label_of(s, win=WIN):
    idx = s.find(C.TARGET_CHAR)
    if len(s) <= win:
        return s
    lo = max(0, min(idx - win // 2, len(s) - win))
    hi = lo + win
    return ("…" if lo else "") + s[lo:hi] + ("…" if hi < len(s) else "")


def main():
    # ---- preserve the original figure under an explicit v1 name
    src_png = os.path.join(C.FIGURES, "fig2_selected5_heatmap.png")
    src_pdf = os.path.join(C.FIGURES, "fig2_selected5_heatmap.pdf")
    for src, name in ((src_png, "fig2a_selected5_v1_template4.png"),
                      (src_pdf, "fig2a_selected5_v1_template4.pdf")):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(C.FIGURES, name))
            print("preserved original ->", name)

    rows = {r["clean_id"]: r for r in rj(TYPOLOGY)}
    params = np.load(PARAMS)
    mu, comp = params["mu"], params["pca_components"]
    Xu, ids = load_corpus()
    by_text = collections.defaultdict(list)
    for i, cid in enumerate(ids):
        by_text[(rows[cid].get("text_norm") or "").strip()].append(i)

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(C.EMBED_MODEL)
    model = AutoModel.from_pretrained(C.EMBED_MODEL)
    model.eval()

    variants = {}
    for tag, s4 in (("v1", S4_V1), ("v2", S4_V2)):
        items = FIXED + [s4, S5]
        vecs, src = [], []
        for cat, s in items:
            s2 = " ".join(s.split())
            hit = by_text.get(s2) or by_text.get(s)
            if hit:
                vecs.append(Xu[hit[0]])
                src.append("corpus")
            else:
                vecs.append(embed_target(s2, tok, model))
                src.append("fresh")
        V = np.stack(vecs)
        Z = transform(V, mu, comp)
        variants[tag] = dict(items=items, S=Z @ Z.T, src=src)
        print(f"\n[{tag}] sources: {src}")

    # ---- comparison table
    S1, S2 = variants["v1"]["S"], variants["v2"]["S"]
    print("\n=== v1 (template at position 4) ===")
    labs = [f"{i+1}" for i in range(5)]
    for i in range(5):
        print("  " + "".join(f"{S1[i, j]:>8.3f}" for j in range(5)))
    print("=== v2 (non-template at position 4) ===")
    for i in range(5):
        print("  " + "".join(f"{S2[i, j]:>8.3f}" for j in range(5)))

    for tag, S in (("v1", S1), ("v2", S2)):
        within = S[3, 4]
        off = [S[i, j] for i in range(5) for j in range(5)
               if i != j and variants[tag]["items"][i][0]
               != variants[tag]["items"][j][0]]
        print(f"{tag}: 同义项(4,5)={within:.3f}  跨义项均值={np.mean(off):.3f}")

    # typicality of the two candidate sentence-4 against the 忍住义 centroid
    cat = np.array([rows[i]["category"] for i in ids])
    Zc = transform(Xu, mu, comp)
    cent = {}
    for c in set(cat.tolist()):
        idx = np.where(cat == c)[0]
        v = Zc[idx].mean(axis=0)
        cent[c] = v / (np.linalg.norm(v) + 1e-12)
    print("\n=== 句子4 两个候选对本类质心的相似度 ===")
    for tag, s4 in (("v1", S4_V1), ("v2", S4_V2)):
        s2 = " ".join(s4[1].split())
        hit = by_text.get(s2) or by_text.get(s4[1])
        v = Xu[hit[0]] if hit else embed_target(s2, tok, model)
        z = transform(v[None, :], mu, comp)[0]
        print(f"  {tag} [{hit and 'corpus' or 'fresh'}]: "
              f"忍住义={float(z @ cent['忍住义']):.3f}  "
              f"忍不住义={float(z @ cent['忍不住义']):.3f}  "
              f"评价义={float(z @ cent['评价义']):.3f}  "
              f"物理义={float(z @ cent['物理义']):.3f}")

    # ---- csv
    import csv
    p = os.path.join(C.TABLES, "selected5_cosine_v1_v2.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for tag in ("v1", "v2"):
            it = variants[tag]["items"]
            w.writerow([f"{tag}: 句子清单"])
            for i, (c_, s_) in enumerate(it):
                w.writerow([i + 1, c_, s_, variants[tag]["src"][i]])
            w.writerow([])
            w.writerow(["cosine"] + [f"{i+1}. {label_of(s)}"
                                     for i, (_, s) in enumerate(it)])
            S = variants[tag]["S"]
            for i in range(5):
                w.writerow([f"{i+1}. {label_of(it[i][1])}"]
                           + [round(float(v), 4) for v in S[i]])
            w.writerow([])

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    out_names = {"v1": "fig2a_selected5_v1_template4",
                 "v2": "fig2b_selected5_v2_nontemplate4"}
    for tag in ("v1", "v2"):
        S = variants[tag]["S"]
        it = variants[tag]["items"]
        ticks = [f"{i+1}. {label_of(s)}" for i, (_, s) in enumerate(it)]
        fig, ax = plt.subplots(figsize=(9.6, 8.0))
        im = ax.imshow(S, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(ticks, rotation=38, ha="right", fontsize=8.5)
        ax.set_yticklabels(ticks, fontsize=8.5)
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(S[a, b]) > 0.45 else "black")
        sub = ("第4句＝“绷住挑战”模板句" if tag == "v1"
               else "第4句＝非模板的忍住义句")
        ax.set_title(f"「绷」代表性句子的语义相似度\n（{sub}）",
                     fontsize=11.5, pad=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.72)
        cb.set_label("cosine similarity")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fp = os.path.join(C.FIGURES, f"{out_names[tag]}.{ext}")
            fig.savefig(fp, dpi=300, bbox_inches="tight")
        print("->", out_names[tag] + ".png/.pdf")

    print("->", p)


if __name__ == "__main__":
    main()
