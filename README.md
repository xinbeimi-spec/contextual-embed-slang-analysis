# 动态词向量与语义演变分析

本仓库整理动态词向量分析项目的可复现代码。当前代码按研究流程分为四个模块：

| 模块 | 目录 | 主要用途 |
|---|---|---|
| 爬虫 | `src/crawl/` | Bilibili 视频发现、弹幕/评论抓取 |
| 数据清理 | `src/clean/` | 文本规范化、目标词定位、噪声/专名过滤、近重复标记与清理报告 |
| 词向量 | `src/embed/` | 预训练模型检查、目标词和掩码上下文向量抽取、中心化与层诊断 |
| 数据分析与作图 | `src/analyze/` | CDPC 聚类、义项/历时统计、热力图、散点图及词频图 |

`src/config.py` 是共享配置入口。路径根据仓库位置解析；目标词、短名和结果轮次可通过环境变量 `LING_WORD`、`LING_SLUG`、`LING_RESULT` 设置。默认目标词为 `哈`（`ha`），但部分分析脚本是针对特定词撰写的，运行前应阅读脚本说明并核对适用词与输入路径。
## Project Materials / 项目材料

- [Presentation / 项目展示 (PDF)](./语言学pre%2011.10.pdf)
- [Research Report / 项目报告 (PDF)](./A%20Contextual%20Embedding%20Analysis%20of%20Four%20Chinese%20Internet%20Terms.pdf)

## 本公开仓库的范围

仓库忽略 `data/`、`results/`、`logs/`、模型权重及 Cookie 配置。原始评论/弹幕和处理数据含用户生成内容、标题和可关联的哈希标识；请勿在未完成隐私审查、脱敏及平台条款核对前公开。图表和结果也未纳入本次代码整理。`examples/sample_cleaned.jsonl` 是纯合成示例，仅用于展示字段结构，不含项目原始内容。

Cookie 文件应保存在本地 `config/`，且已加入忽略规则。只使用空文件/本地凭据，不要把 Cookie、`.env` 或令牌提交到 Git。此次检查发现 Cookie 曾以明文保存在项目配置文件中；请在对应平台退出/撤销这些会话并重新登录，空文件不能使已泄露凭据失效。

## 环境

建议使用 Python 3.10 或 3.11。安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本项目模型使用 `hfl/chinese-roberta-wwm-ext`。

## 主流程参考

```powershell
# 设置目标词（示例）
$env:LING_WORD = "哈"
$env:LING_SLUG = "ha"

# 爬取（先检查参数与平台访问权限）
python src/crawl/crawl_bili.py --help

# 清理
python src/clean/clean_corpus.py

# 准备模型并抽取目标词向量；掩码向量用于语境聚类
python src/embed/prepare_model.py
python src/embed/extract_embeddings.py
python src/embed/extract_masked_vectors.py
python src/embed/layer_probe.py
# 可选：对 target 表示做各向异性诊断与中心化；CDPC 对 masked 表示会在流水线中处理
python src/embed/center_embeddings.py

# 查看具体分析脚本的参数与适用目标词，再运行
python src/analyze/cdpc_pipeline.py --help
```

`cdpc_pipeline.py` 当前只实现「拉」的目标词分类逻辑，会拒绝其他目标词；哈、绷、塌的分析入口和图表脚本分散在 `src/analyze/`，需按各脚本说明选择，不能把同一条分析命令直接套用到所有词。

爬虫中的 `probe_*`、`diag_*` 和 `_probe_*` 脚本属于 Bilibili 研究诊断工具，不是常规流水线入口。

## 代码结构注意事项

项目历史上由多个目标词的研究轮次逐步扩展。`src/crawl/`、`src/clean/`、`src/embed/` 和 `src/analyze/` 是按功能分层；分析目录还含目标词专用与探索性脚本，不应把所有脚本视为一个通用流水线。
