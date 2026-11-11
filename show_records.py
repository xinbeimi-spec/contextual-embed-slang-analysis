# -*- coding: utf-8 -*-
"""小工具：打印几条原始（raw）与加工后（processed）记录，方便肉眼核对数据。

为什么需要它
  数据都是 JSONL（一行一条记录），直接双击打不开、用记事本看又挤成一团。
  这个脚本把若干条记录排成「人话」打印出来，可按年份/关键词过滤、可随机抽。

三个数据文件（= 三个加工阶段；这里的「文件」不是神经网络的「层」，别混）
  raw     data/raw/<slug>/bilibili/hits_comment.jsonl   ← 爬下来的原始命中，未加工
  clean   data/processed/<slug>/<slug>_clean.jsonl      ← 清洗后主表（加了几列清洗标记）
  cdpc    data/processed/<slug>/<slug>_cdpc.jsonl       ← 聚类与义项标注（clean 再加类别）

用法（在 D:\\ling_project 下）
  python src\\show_records.py                       # 默认：raw 3 条 + cdpc 3 条
  python src\\show_records.py --n 5                 # 每个文件各打印 5 条
  python src\\show_records.py --which raw           # 只看原始数据
  python src\\show_records.py --year 2016           # 只看 2016 年的
  python src\\show_records.py --grep 绷带           # 只看正文里含「绷带」的
  python src\\show_records.py --random --seed 7     # 随机抽 3 条（可复现）
  python src\\show_records.py --json                # 额外打印完整 JSON（字段全看）
  python src\\show_records.py --out logs\\sample.txt # 同时写文件（控制台乱码时用这个）
  python src\\show_records.py --slug la             # 看别的词（la / ta / beng）

注意
  Windows 控制台有时会把中文显示成乱码。如果打印出来是乱码，加 --out 参数把结果写到
  文件里，再用记事本/VSCode 打开那个文件即可（文件一定是 UTF-8）。
"""
import os
import sys
import json
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sys.stdout.reconfigure(encoding="utf-8")

ROOT = C.ROOT


def paths_for(slug):
    """按词返回三个数据文件的路径（不存在就是 None）。"""
    raw = os.path.join(ROOT, "data", "raw", slug, "bilibili", "hits_comment.jsonl")
    proc = os.path.join(ROOT, "data", "processed", slug)
    clean = os.path.join(proc, f"{slug}_clean.jsonl")
    cdpc = os.path.join(proc, f"{slug}_cdpc.jsonl")
    return raw, clean, cdpc


def read_jsonl(path):
    """一次性读入（文件只有几 MB，读进内存最简单）。一行解析失败就跳过并提示。"""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  （{os.path.basename(path)} 有 {bad} 行解析失败，已跳过）")
    return rows


def text_of(r):
    """不同文件的正文列名不一样，统一取一个。"""
    return r.get("text_raw") or r.get("text_clean") or r.get("text_norm") or ""


def match(r, args):
    """年份 + 关键词过滤。两个条件都为空则全部通过。"""
    if args.year:
        if (r.get("ctime_iso") or "")[:4] != str(args.year):
            return False
    if args.grep:
        if args.grep not in text_of(r) and args.grep not in (r.get("text_norm") or ""):
            return False
    return True


def pick(rows, n, args):
    """抽样：--random 就随机（受 --seed 控制，可复现），否则取前 n 条。"""
    if args.random:
        rng = random.Random(args.seed)
        return rng.sample(rows, min(n, len(rows)))
    return rows[:n]


def show_block(tag, i, r, layer, show_json):
    """打印一条记录。layer 决定显示哪些标注字段。"""
    t = text_of(r)
    lines = []
    head = (f"[{tag} #{i}]  {(r.get('ctime_iso') or '')[:19]}  "
            f"{r.get('thread_bvid') or r.get('bvid') or ''}  "
            f"《{(r.get('thread_title') or '')[:28]}》")
    lines.append(head)
    if layer == "raw":
        lines.append(f"  item_id_hash: {r.get('item_id_hash')}   "
                     f"类型: {r.get('content_type')}   字数: {r.get('n_chars')}   "
                     f"点赞: {r.get('like_count')}   来源检索词: {r.get('source_query')}")
    else:
        lines.append(f"  clean_id: {r.get('clean_id')}   字数: {r.get('n_chars_norm')}   "
                     f"去重代表: {r.get('is_canonical')}   "
                     f"可判定: {r.get('context_sufficiency')}   "
                     f"点赞: {r.get('like_count')}")
        if layer == "cdpc":
            lines.append(f"  ★ 义项: {r.get('cdpc_category')}   聚类簇: {r.get('cdpc_cluster')}   "
                         f"空间表示: {r.get('representation', 'masked(默认)')}")
    lines.append("  ── 正文 ──")
    lines.append("  " + t.replace("\n", " "))
    if show_json:
        lines.append("  ── 完整 JSON ──")
        lines.append("  " + json.dumps(r, ensure_ascii=False))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="打印几条 raw / processed 记录看看")
    ap.add_argument("--slug", default="beng", help="哪个词：beng / la / ta（默认 beng）")
    ap.add_argument("--which", default="both",
                    choices=["raw", "clean", "cdpc", "both"],
                    help="看哪个文件：raw(原始) / clean(清洗后) / cdpc(带义项) / both(raw+cdpc)")
    ap.add_argument("--n", type=int, default=3, help="每个文件打印几条（默认 3）")
    ap.add_argument("--year", default="", help="只看某一年，如 2016")
    ap.add_argument("--grep", default="", help="只看正文含该关键词的记录，如 绷带")
    ap.add_argument("--random", action="store_true", help="随机抽（默认取前 n 条）")
    ap.add_argument("--seed", type=int, default=2026, help="随机种子（配合 --random）")
    ap.add_argument("--json", action="store_true", help="额外打印完整 JSON")
    ap.add_argument("--out", default="", help="把输出同时写到文件（控制台乱码时用）")
    a = ap.parse_args()

    raw_p, clean_p, cdpc_p = paths_for(a.slug)
    layers = []
    if a.which in ("raw", "both"):
        layers.append(("raw", raw_p, "raw"))
    if a.which in ("clean",):
        layers.append(("clean", clean_p, "clean"))
    if a.which in ("cdpc", "both"):
        # 没有 cdpc 时自动退回 clean，避免报错看不懂
        layers.append(("cdpc", cdpc_p if os.path.exists(cdpc_p) else clean_p,
                       "cdpc" if os.path.exists(cdpc_p) else "clean"))

    out = [f"词: {a.slug}    过滤: 年份={a.year or '全部'}  关键词={a.grep or '无'}   "
           f"抽样: {'随机(seed=%d)' % a.seed if a.random else '前 %d 条' % a.n}", ""]
    stats = []
    for tag, path, layer in layers:
        if not os.path.exists(path):
            out.append(f"[{tag}] 文件不存在：{path}")
            continue
        rows = read_jsonl(path)
        hit = [r for r in rows if match(r, a)]
        picked = pick(hit, a.n, a)
        out.append(f"===== {tag.upper()}  {os.path.relpath(path, ROOT)}  =====")
        out.append(f"共 {len(rows)} 条，符合过滤条件 {len(hit)} 条，下面打印 {len(picked)} 条")
        out.append("")
        for i, r in enumerate(picked, 1):
            out.append(show_block(tag, i, r, layer, a.json))
        stats.append(f"{tag}: 总 {len(rows)} / 命中过滤 {len(hit)}")

    txt = "\n".join(out)
    try:
        print(txt)
        print("统计：" + "；".join(stats))
    except BrokenPipeError:
        # 用 `| Select-Object -First N` 或 `| more` 时，管道会提前关闭；
        # Python 这时会报 BrokenPipeError，忽略即可（数据已经打印出去了）。
        pass
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(txt + "\n")
        print(f"已写入文件：{p}（用记事本/VSCode 打开看，UTF-8）")


if __name__ == "__main__":
    main()
