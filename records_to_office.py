# -*- coding: utf-8 -*-
"""把 show_records.py 打印的那几条记录，直接做成 Excel + PPT（方便看清、方便汇报）。

为什么需要它
  PowerShell 里看长句子很难受（换行、乱码、字小）。这个脚本用同一套过滤/抽样条件
  选出记录，然后：
    * 写一个 Excel（表格，含完整句子、义项、时间、来源、字数、点赞）
    * 写一个 PPT（一条记录一页，句子放大、自动换行、按长度自动调字号）

用法（与 show_records.py 的参数完全一致）
  $PY src\\records_to_office.py                          # 默认 raw 3 条 + cdpc 3 条
  $PY src\\records_to_office.py --which cdpc --n 5        # 只看带义项的 5 条
  $PY src\\records_to_office.py --grep 绷带 --n 3         # 正文含「绷带」的 3 条
  $PY src\\records_to_office.py --year 2016 --n 3
  $PY src\\records_to_office.py --random --seed 7 --n 5
  $PY src\\records_to_office.py --outdir results\\beng\\exports --name 汇报样张

注意：这个脚本要用**带 python-pptx / openpyxl 的 Python**（本机打包运行时自带），
     不能直接用 nlp 那个 conda 环境。见文件末尾的说明。
"""
import os
import sys
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from show_records import paths_for, read_jsonl, match, pick, text_of

sys.stdout.reconfigure(encoding="utf-8")

# 五类义项的颜色（原始数据没有义项，用灰色）
CAT_COLOR = {"物理义": (0x4C, 0x72, 0xB0), "忍住义": (0xDD, 0x84, 0x52),
             "忍不住义": (0xC4, 0x4E, 0x52), "评价义": (0x55, 0xA8, 0x68),
             "紧绷义": (0x81, 0x72, 0xB3), "非义项用法": (0x9E, 0x9E, 0x9E)}
GREY = (0x77, 0x77, 0x77)


def collect(a):
    """按 --which / 过滤 / 抽样 选记录，返回统一结构的列表。"""
    raw_p, clean_p, cdpc_p = paths_for(a.slug)
    layers = []
    if a.which in ("raw", "both"):
        layers.append(("原始", raw_p))
    if a.which == "clean":
        layers.append(("清洗后", clean_p))
    if a.which in ("cdpc", "both"):
        layers.append(("标注", cdpc_p if os.path.exists(cdpc_p) else clean_p))

    recs = []
    for tag, path in layers:
        if not os.path.exists(path):
            continue
        rows = read_jsonl(path)
        hit = [r for r in rows if match(r, a)]
        for r in pick(hit, a.n, a):
            recs.append({
                "层级": tag,
                "义项": r.get("cdpc_category") or "",
                "正文": text_of(r),
                "时间": (r.get("ctime_iso") or "")[:19].replace("T", " "),
                "年份": (r.get("ctime_iso") or "")[:4],
                "视频": r.get("thread_title") or "",
                "BV": r.get("thread_bvid") or r.get("bvid") or "",
                "字数": r.get("n_chars_norm") or r.get("n_chars") or len(text_of(r)),
                "点赞": r.get("like_count"),
                "簇": r.get("cdpc_cluster"),
            })
    return recs


def font_size_for(text):
    """按句子长度选字号：太长就缩小，保证不溢出。"""
    n = len(text)
    if n <= 40:
        return 28
    if n <= 80:
        return 24
    if n <= 140:
        return 20
    if n <= 220:
        return 17
    return 15


def write_xlsx(recs, path, a):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "记录"
    head = ["序号", "层级", "义项", "正文（完整）", "发送时间", "年份", "视频标题",
            "BV号", "字数", "点赞", "簇号"]
    ws.append(head)
    for i, r in enumerate(recs, 1):
        ws.append([i, r["层级"], r["义项"], r["正文"], r["时间"], r["年份"],
                   r["视频"], r["BV"], r["字数"], r["点赞"], r["簇"]])

    fill = PatternFill("solid", fgColor="DCE6F1")
    thin = Side(style="thin", color="BFBFBF")
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = fill
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    widths = [6, 8, 10, 78, 20, 7, 34, 14, 6, 8, 6]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in range(2, ws.max_row + 1):
        for col in (4, 7):
            ws.cell(row, col).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = max(30, 15 * (
            len(ws.cell(row, 4).value or "") // 40 + 1))
    ws.freeze_panes = "A2"

    # 第二个工作表：说明来源与抽样条件，答辩时被问到可直接翻
    ws2 = wb.create_sheet("说明")
    info = [("项", "内容"),
            ("目标词", a.slug),
            ("数据来源", f"data/{a.slug}/ 下 raw / clean / cdpc 三份文件"),
            ("抽样条件",
             f"层级={a.which}；年份={a.year or '全部'}；关键词={a.grep or '无'}；"
             f"{'随机 seed=' + str(a.seed) if a.random else '取前 %d 条' % a.n}"),
            ("义项来源", "cdpc 文件里的 cdpc_category（masked 表示、k=7 的六类标注）"),
            ("生成时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("生成脚本", "src/records_to_office.py")]
    for row in info:
        ws2.append(list(row))
    for c in ws2[1]:
        c.font = Font(bold=True)
        c.fill = fill
    ws2.column_dimensions["A"].width = 14
    ws2.column_dimensions["B"].width = 86
    for row in range(2, ws2.max_row + 1):
        ws2.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(path)
    return path


def write_pptx(recs, path, a):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---- 封面
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(11.7), Inches(1.2))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = f"「{a.slug}」语料记录样张"
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    tb2 = s.shapes.add_textbox(Inches(0.85), Inches(3.9), Inches(11.7), Inches(1.6))
    tf = tb2.text_frame
    tf.word_wrap = True
    cond = (f"层级：{a.which}｜年份：{a.year or '全部'}｜"
            f"关键词：{a.grep or '无'}｜"
            f"{'随机抽样 seed=' + str(a.seed) if a.random else '取前 %d 条' % a.n}")
    for i, line in enumerate([cond,
                              "来源：B 站评论（data/ 下 raw → clean → cdpc 三个加工阶段）",
                              f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(16)
        r.font.color.rgb = RGBColor(*GREY)

    # ---- 每条记录一页
    for i, rec in enumerate(recs, 1):
        s = prs.slides.add_slide(blank)
        # 左上：义项标签（原始数据没有义项时显示「原始数据」）
        tag = rec["义项"] or "原始数据"
        color = CAT_COLOR.get(rec["义项"], GREY)
        chip = s.shapes.add_textbox(Inches(0.6), Inches(0.45), Inches(3.4),
                                    Inches(0.6))
        p = chip.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = f"  {tag}  "
        r.font.size = Pt(20)
        r.font.bold = True
        r.font.color.rgb = RGBColor(*color)

        # 右上：序号 / 时间 / 字数
        meta = s.shapes.add_textbox(Inches(8.4), Inches(0.45), Inches(4.3),
                                    Inches(0.9))
        tf = meta.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
        r = tf.paragraphs[0].add_run()
        r.text = f"#{i}   {rec['时间']}   {rec['字数']} 字   赞 {rec['点赞']}"
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(*GREY)

        # 正文：大字号、自动换行、按长度自适应
        body = s.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(12.0),
                                    Inches(4.6))
        tf = body.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = rec["正文"]
        r.font.size = Pt(font_size_for(rec["正文"]))
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        p.line_spacing = 1.25

        # 底部：视频标题 / BV / 簇号
        foot = s.shapes.add_textbox(Inches(0.7), Inches(6.35), Inches(12.0),
                                    Inches(0.8))
        tf = foot.text_frame
        tf.word_wrap = True
        cls = f"｜簇 C{rec['簇']}" if rec["簇"] is not None else ""
        for j, line in enumerate([f"《{rec['视频'][:40]}》{cls}",
                                  f"{rec['BV']}｜来源文件：{rec['层级']}"]):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            r = p.add_run()
            r.text = line
            r.font.size = Pt(11)
            r.font.color.rgb = RGBColor(*GREY)

    # ---- 结尾：抽样说明（可删）
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(5.6))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = ["抽样说明（汇报时备查）",
             f"· 数据来源：B 站评论，2016–2026；{a.slug} 词共 4,647 条原始命中",
             "· 三个加工阶段：raw（原始）→ clean（清洗后）→ cdpc（聚类与义项标注）",
             "· 义项标注：masked 表示 + CDPC 聚类（k=7），六类归并",
             "· 本页记录仅作样例展示，不代表该类别的整体分布",
             "　（整体分布见 results 下的 typology_sizes.csv 与聚类图）"]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(18 if i == 0 else 14)
        r.font.bold = (i == 0)
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22) if i == 0 else RGBColor(*GREY)

    prs.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description="把 show_records 的记录写成 Excel + PPT")
    ap.add_argument("--slug", default="beng")
    ap.add_argument("--which", default="both",
                    choices=["raw", "clean", "cdpc", "both"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--year", default="")
    ap.add_argument("--grep", default="")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outdir", default=os.path.join("results", "beng", "exports"))
    ap.add_argument("--name", default="records")
    a = ap.parse_args()

    recs = collect(a)
    if not recs:
        raise SystemExit("没有选到记录：换个过滤条件试试（--which / --year / --grep）")
    outdir = a.outdir if os.path.isabs(a.outdir) else os.path.join(C.ROOT, a.outdir)
    os.makedirs(outdir, exist_ok=True)
    xlsx = os.path.join(outdir, f"{a.name}.xlsx")
    pptx = os.path.join(outdir, f"{a.name}.pptx")

    write_xlsx(recs, xlsx, a)
    write_pptx(recs, pptx, a)

    print(f"选到 {len(recs)} 条记录：")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. [{r['义项'] or '原始'}] {r['正文'][:50]}")
    print(f"\n-> {xlsx}")
    print(f"-> {pptx}")


if __name__ == "__main__":
    main()
