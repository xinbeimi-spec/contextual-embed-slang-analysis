# -*- coding: utf-8 -*-
"""Similarity heatmap for the five hand-selected prototype sentences.

Per the user's request this version:
  * uses ONLY the target-word vector (no masked panel);
  * labels both axes with the SENTENCES, not the category names.

Because the sentences are 20-40 characters, axis labels are produced by taking a
window centred on the target character (so 绷 is always visible) and marking the
cut with an ellipsis. The full sentences are kept in the CSV and printed at the
end, so nothing is lost.

The heatmap values are cosine similarities between contextualised vectors after
the SAME centring + ABTT transformation stored with the corpus (never re-fitted).
Because of that centring, the corpus-wide mean cosine is ~0, so only the
RELATIVE ordering of cells is interpretable, not the absolute sign.
"""
import os
import sys
import json
import collections

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

TYPOLOGY = os.path.join(C.PROCESSED, "beng_typology.jsonl")
EMB_UNMASK = C.EMB_DIR
PARAMS = os.path.join(C.EMB_DIR, "center_params.npz")
LAYERS = (9, 10, 11, 12)

# (category, sentence) -- exactly as supplied by the user
ITEMS = [
    ("物理义", "要那么多绷带干嘛？缠成木乃伊么"),
    ("忍不住义", "一想到十年后互联网主力军的童年回忆是奶龙就绷不住了"),
    ("评价义", "要说真正难绷的还得是老美自己发的火星图片里能看到矿泉水瓶和苍蝇"),
    ("忍住义", "如果你连续78次绷住的话 我后面的780万就是你的了"),
    ("忍住义", "你好，我是百万美元，如果你绷住，身后的野崩先生就是你的了"),
]
WIN = 22          # characters shown per axis label


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_corpus(root):
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
    return np.stack(xs).astype(np.float32)[:, LAYERS, :].mean(axis=1), ids


def transform(X, mu, comp):
    Z = X - mu
    if comp is not None and len(comp):
        Z = Z - (Z @ comp.T) @ comp
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def embed_target(s, tok, model):
    idx = s.find(C.TARGET_CHAR)
    if idx < 0:
        raise ValueError(f"target char missing: {s}")
    enc = tok(s, return_tensors="pt", return_offsets_mapping=True,
              truncation=True, max_length=C.MAX_LEN)
    offs = enc.pop("offset_mapping")[0].tolist()
    span = [i for i, (a, b) in enumerate(offs) if a != b and a <= idx < b]
    with torch.no_grad():
        o = model(**enc, output_hidden_states=True)
    hs = torch.stack(o.hidden_states, dim=0)[:, 0, :, :]     # (L+1, T, H)
    sel = hs[list(LAYERS)]
    return sel[:, span, :].mean(dim=1).mean(dim=0).cpu().float().numpy()


def label_of(s, win=WIN):
    """Window centred on the target char, so 绷 is always shown."""
    idx = s.find(C.TARGET_CHAR)
    if len(s) <= win:
        return s
    half = win // 2
    lo = max(0, min(idx - half, len(s) - win))
    hi = lo + win
    out = s[lo:hi]
    return ("…" if lo > 0 else "") + out + ("…" if hi < len(s) else "")


def main():
    rows = {r["clean_id"]: r for r in rj(TYPOLOGY)}
    params = np.load(PARAMS)
    mu, comp = params["mu"], params["pca_components"]

    Xu, ids = load_corpus(EMB_UNMASK)
    by_text = collections.defaultdict(list)
    for i, cid in enumerate(ids):
        by_text[(rows[cid].get("text_norm") or "").strip()].append(i)

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(C.EMBED_MODEL)
    model = AutoModel.from_pretrained(C.EMBED_MODEL)
    model.eval()

    print("=== sentence resolution ===")
    vecs = []
    for cat, s in ITEMS:
        s2 = " ".join(s.split())
        hit = by_text.get(s2) or by_text.get(s)
        if hit:
            vecs.append(Xu[hit[0]])
            print(f"  [corpus] {cat:6} <- {s2[:46]}")
        else:
            vecs.append(embed_target(s2, tok, model))
            print(f"  [fresh ] {cat:6} <- {s2[:46]}")
    V = np.stack(vecs)
    Z = transform(V, mu, comp)
    S = Z @ Z.T

    labels = [label_of(s) for _, s in ITEMS]
    print("\n=== cosine matrix (target-word vectors) ===")
    print(f"{'':26}" + "".join(f"{i+1:>9}" for i in range(len(labels))))
    for i, (cat, s) in enumerate(ITEMS):
        print(f"{i+1}. {labels[i]:22}" + "".join(f"{S[i, j]:>9.3f}"
                                                 for j in range(len(labels))))
    print(f"\n同义项（4 vs 5）: {S[3,4]:.3f}")
    off = [S[i, j] for i in range(5) for j in range(5)
           if i != j and ITEMS[i][0] != ITEMS[j][0]]
    print(f"跨义项均值       : {np.mean(off):.3f}")

    # ---- csv
    import csv
    p1 = os.path.join(C.TABLES, "selected5_cosine.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "category", "full_sentence"])
        for i, (cat, s) in enumerate(ITEMS):
            w.writerow([i + 1, cat, s])
        w.writerow([])
        w.writerow(["representation", "target_span_mean (centred+ABTT, same "
                                        "transform as corpus)"])
        w.writerow([""] + [f"{i+1}. {labels[i]}" for i in range(len(labels))])
        for i in range(len(labels)):
            w.writerow([f"{i+1}. {labels[i]}"]
                       + [round(float(v), 4) for v in S[i]])

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
    ticks = [f"{i+1}. {lab}" for i, lab in enumerate(labels)]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(ticks, rotation=38, ha="right", fontsize=8.5)
    ax.set_yticklabels(ticks, fontsize=8.5)
    for a in range(len(labels)):
        for b in range(len(labels)):
            ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(S[a, b]) > 0.45 else "black")
    ax.set_title("「绷」代表性句子的语义相似度（上下文化向量余弦）",
                 fontsize=12, pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.72)
    cb.set_label("cosine similarity")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(C.FIGURES, f"fig2_selected5_heatmap.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("\n->", p)

    print("\n=== full sentences (for the caption) ===")
    for i, (cat, s) in enumerate(ITEMS):
        print(f"  {i+1}. [{cat}] {s}")
    print("->", p1)


if __name__ == "__main__":
    main()
