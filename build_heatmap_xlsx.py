# -*- coding: utf-8 -*-
"""把 5 句样本与余弦矩阵写成 Excel（openpyxl，新建工作簿）。

为什么单独一个脚本：热力图的坐标轴只能放短标签，**完整句子**必须另有载体，
所以这里生成 `semantic_heatmap_5senses.xlsx`，三个工作表：

  1) 样本句     每条：序号 / 义项 / 完整句子（清洗后）/ 原始文本 / clean_id /
                 发送时间 / 年份 / 簇号 / 词形 / 字数 / 点赞
  2) 余弦矩阵   5×5 数值 + 行列标题 + 下方「句号 → 完整句子」对照，方便对着矩阵读
  3) 说明       抽样方法、随机种子、候选池大小、向量表示、读数注意事项

输入（由 src/analyze/make_semantic_heatmap.py 生成）：
  results/beng/result4/semantic_heatmap_5senses_samples.csv
  results/beng/result4/semantic_heatmap_5senses_matrix.csv
输出：
  results/beng/result4/semantic_heatmap_5senses.xlsx
"""
import os
import csv
import json

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results", "beng", "result4")
SAMP = os.path.join(OUTDIR, "semantic_heatmap_5senses_samples.csv")
# 正式 CSV 被占用时，热力图脚本会写 .new.csv；这里自动用较新的那份
_n = SAMP.replace(".csv", ".new.csv")
if os.path.exists(_n) and (not os.path.exists(SAMP)
                           or os.path.getmtime(_n) > os.path.getmtime(SAMP)):
    SAMP = _n
MAT = os.path.join(OUTDIR, "semantic_heatmap_5senses_matrix.csv")
META = os.path.join(OUTDIR, "semantic_heatmap_5senses_meta.json")
OUT = os.path.join(OUTDIR, "semantic_heatmap_5senses.xlsx")

HEAD_FILL = PatternFill("solid", fgColor="DCE6F1")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def num(v, cast=int):
    """CSV 读出来都是字符串；能转成数字的转成数字，保证 Excel 里是数值而不是文本。"""
    try:
        return cast(v)
    except (TypeError, ValueError):
        return v


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def style_header(ws, row=1):
    for cell in ws[row]:
        if cell.value is None:
            continue
        cell.font = Font(bold=True)
        cell.fill = HEAD_FILL
        cell.border = BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)


def main():
    samples = read_csv(SAMP)
    matrix = read_csv(MAT)
    meta = json.load(open(META, encoding="utf-8"))

    wb = Workbook()

    # ---------------------------------------------------------------- 样本句
    ws = wb.active
    ws.title = "样本句"
    # 列序：0 序号 / 1 义项 / 2 短标签 / 3 完整句子 / 4 原始文本 / 5 clean_id /
    #       6 发送时间 / 7 年份 / 8 簇号 / 9 词形 / 10 字数 / 11 点赞
    ws.append(list(samples[0]))          # 第 1 行 = 表头
    for row in samples[1:]:
        vals = list(row)
        vals[0] = num(vals[0])          # 序号 -> 整数
        vals[7] = num(vals[7])          # 年份 -> 整数
        vals[8] = num(vals[8])          # 簇号 -> 整数
        vals[10] = num(vals[10])        # 字数 -> 整数
        vals[11] = num(vals[11])        # 点赞 -> 整数（可能为空）
        ws.append(vals)
    style_header(ws)
    widths = [6, 10, 26, 70, 70, 20, 20, 8, 8, 10, 8, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for r in range(2, ws.max_row + 1):
        ws.cell(r, 4).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(r, 5).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 60
    ws.freeze_panes = "A2"

    # ---------------------------------------------------------------- 余弦矩阵
    ws2 = wb.create_sheet("余弦矩阵")
    for i, row in enumerate(matrix):
        if i == 0:
            ws2.append(list(row))        # 表头
        else:
            # 第 1 列是「句号·义项」文本，其余是数值 -> 转成 float 才能做数值格式
            ws2.append([row[0]] + [num(v, float) for v in row[1:]])
    style_header(ws2)
    ws2.column_dimensions["A"].width = 16
    for c in range(2, len(matrix[0]) + 1):
        ws2.column_dimensions[get_column_letter(c)].width = 14
    for r in range(2, len(matrix) + 1):
        for c in range(2, len(matrix[0]) + 1):
            cell = ws2.cell(r, c)
            cell.number_format = "+0.000;-0.000"
            cell.alignment = Alignment(horizontal="center")
            cell.border = BORDER
            if r - 1 == c - 1:                 # 对角线 = 1，标粗方便看
                cell.font = Font(bold=True)
        ws2.cell(r, 1).border = BORDER

    # 矩阵下方补一张「句号 → 完整句子」对照，读矩阵时不用翻页
    start = len(matrix) + 3
    ws2.cell(start, 1, "句号 → 完整句子（对照用）").font = Font(bold=True)
    ws2.cell(start + 1, 1, "句号").font = Font(bold=True)
    ws2.cell(start + 1, 2, "义项").font = Font(bold=True)
    ws2.cell(start + 1, 3, "完整句子（清洗后）").font = Font(bold=True)
    for i, row in enumerate(samples[1:], 1):
        ws2.cell(start + 1 + i, 1, f"句{i}")
        ws2.cell(start + 1 + i, 2, row[1])
        cell = ws2.cell(start + 1 + i, 3, row[3])
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws2.row_dimensions[start + 1 + i].height = 42
    ws2.merge_cells(start_row=start, start_column=1, end_row=start, end_column=3)

    # ---------------------------------------------------------------- 说明
    ws3 = wb.create_sheet("说明")
    lines = [
        ("项目", "内容"),
        ("目标词", "绷（beng）"),
        ("数据来源", f"{meta['source']}（result4 同一批标注，未做任何改动）"),
        ("抽样范围", "5 个义项：物理义 / 忍住义 / 忍不住义 / 评价义 / 紧绷义"),
        ("为什么没有非义项用法",
         "「非义项用法」不是词义（提及而非使用：复读、玩梗、错别字），所以不参与抽样"),
        ("抽样方式", "每个义项从候选池中随机抽 1 句"),
        ("随机种子", str(meta["seed"]) + "（固定，可复现；改种子即换一组句子）"),
        ("候选池条件",
         "is_canonical=1（去重代表）+ context_sufficiency=1（人可判定）"
         f" + 长度 {meta['length_filter'][0]}–{meta['length_filter'][1]} 字"
         " + 排除 excluded_from_typology"),
        ("候选池大小",
         "；".join(f"{k} {v} 条" for k, v in meta["pool_sizes"].items())),
        ("向量表示", meta["representation"]),
        ("相似度", "余弦相似度 = 两个 L2 归一化向量的点积"),
        ("读数警告",
         "句向量已去均值并移除第一主成分，矩阵绝对值不可作阈值，只有相对高低有意义"),
        ("生成脚本",
         "src/analyze/make_semantic_heatmap.py（抽样与画图）、"
         "src/analyze/build_heatmap_xlsx.py（本表）"),
    ]
    for row in lines:
        ws3.append(list(row))
    style_header(ws3)
    ws3.column_dimensions["A"].width = 22
    ws3.column_dimensions["B"].width = 92
    for r in range(2, ws3.max_row + 1):
        ws3.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws3.row_dimensions[r].height = 34 if r > 3 else 20

    wb.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
