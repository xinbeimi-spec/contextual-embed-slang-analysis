# -*- coding: utf-8 -*-
"""Project-wide configuration for the Chinese neologism semantic-change study.

Paths are resolved relative to this checkout. Every downstream module imports
this file FIRST, so that environment variables are set before huggingface_hub /
transformers are imported anywhere in the process.
"""
import os
import sys

# ---------------------------------------------------------------- environment
# Must run before `import transformers` / `import huggingface_hub` anywhere.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Store model weights outside the repository, on the C: profile by default.
# Override with LING_HF_HOME on systems with a different layout.
_HF_HOME = os.environ.get("LING_HF_HOME", os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "huggingface"))
_HF_HOME = os.path.abspath(os.path.expandvars(os.path.expanduser(_HF_HOME)))
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
# Windows: creating symlinks needs SeCreateSymbolicLinkPrivilege (WinError 1314
# otherwise). Tell huggingface_hub to copy blobs instead of linking them.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------- paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA = os.path.join(ROOT, "data")
RAW_ROOT = os.path.join(DATA, "raw")
PROC_ROOT = os.path.join(DATA, "processed")
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")

# ---------------------------------------------------------- per-word layout
# 2026-10-24 decision: EVERY target word gets its own folder, so the words never
# mix:   data/raw/<slug>/        raw Bilibili harvest (_backups/, _manifest.jsonl)
#        data/processed/<slug>/  clean table, vectors (emb2/), labels, drops
# For a new word, set LING_WORD / LING_SLUG (env) or edit the two defaults below.
#   $env:LING_WORD="破防"; $env:LING_SLUG="pofang"
# ------------------------------------------------------------------
# 按词分目录的开关（2026-10-24 起）。换新词时**不用改代码**，只设环境变量：
#   $env:LING_WORD = "破防" ; $env:LING_SLUG = "pofang"
# WORD      : 目标词本身，会作为 TARGET_CHAR 参与「文本里找目标词」的计算
# WORD_SLUG : 英文短名，只用来当文件夹名（不能有中文，避免路径编码问题）
# os.environ.get(名, 默认值)：有环境变量就用它，没有就用默认值
# ------------------------------------------------------------------
# 2026-11-09: the project moved on to the FOURTH target word 哈 (hā).
# 绷 (beng), 塌 (ta) and 拉 (la) are finished and archived under data/,
# results/, logs/ <beng|ta|la>/; to re-run anything for them set the env vars
# explicitly, e.g.  $env:LING_WORD="拉"; $env:LING_SLUG="la".
# The DEFAULT is now 哈/ha on purpose: a forgotten env var must never append
# 哈's harvest into a finished corpus.
WORD = os.environ.get("LING_WORD", "哈")
WORD_SLUG = os.environ.get("LING_SLUG", "ha")
# 本词的原始数据根目录 = data/raw/<slug>/（每个词一个，不互相污染）
RAW = os.path.join(RAW_ROOT, WORD_SLUG)
# 本词的加工数据根目录 = data/processed/<slug>/
PROCESSED = os.path.join(PROC_ROOT, WORD_SLUG)
BILI_DIR = os.path.join(RAW, "bilibili")

# Result layout decided 2026-10-21 (plain names, no "final_5cat" style):
#   results/result1/  cat4|cat5|cat6 + evidence  -> 五类版归档（旧语料）
#   results/result2/  cat4|cat5|cat6 + evidence  -> 六类版归档（扩量语料, KMeans）
#   results/result3/  flat, no subfolders        -> 上一轮六类（KMeans）现役输出
#   results/result4/  flat, no subfolders        -> 现役输出：CDPC 聚类 + 词频分析
# Per-word rule (2026-10-27): EVERY word, including beng, writes to
# results/<slug>/resultN/.  beng's delivered results were moved to
# results/beng/result1..4, so the old `RESULTS if slug == "beng"` branch would
# have re-created an empty results/result4 -- removed.
RESULT_ROOT = os.path.join(RESULTS, WORD_SLUG)
# Which ROUND of results the scripts write into (2026-10-31).  A re-run of the
# clustering with a different k or a different category scheme is a NEW round
# and must not overwrite the previous one:   $env:LING_RESULT="result2"
RESULT_SET = os.environ.get("LING_RESULT", "result1")
RES4 = os.path.join(RESULT_ROOT, RESULT_SET)
FIGURES = RES4
TABLES = RES4
# Per-word logs (2026-10-27): same convention as data/ and results/ — each word's
# run logs, probe dumps and review files live in logs/<slug>/ so words never mix.
LOGS = os.path.join(ROOT, "logs", WORD_SLUG)

for _d in (RAW, PROCESSED, FIGURES, TABLES, LOGS,
           os.path.join(RAW, "bilibili")):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------ embeddings
# ALL vectors for the MERGED corpus live in one place (2026-10-21 decision):
#   data/processed/emb2/target/   target-span pooled vectors  (formerly emb/)
#   data/processed/emb2/masked/   [MASK]-position vectors     (formerly emb_mask/)
# They are two REPRESENTATIONS of the same corpus, not two data sets. Scripts
# must read these constants instead of hard-coding either path.
EMB2 = os.path.join(PROCESSED, "emb2")
EMB_DIR = os.path.join(EMB2, "target")
EMB_MASK_DIR = os.path.join(EMB2, "masked")
for _d in (EMB2, EMB_DIR, EMB_MASK_DIR):
    os.makedirs(_d, exist_ok=True)

# Local credentials. Never commit or print these values.
CONFIG_DIR = os.path.join(ROOT, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)
BILI_COOKIE_FILE = os.path.join(CONFIG_DIR, "bili_cookie.txt")

# --------------------------------------------------------------------- study
# 2026-11-09: date of THIS harvest batch (哈). Keep it as the batch label; the
# per-item crawl_date is written from here, and "old vs new batch" separation
# must be done with a pre-expansion snapshot, not with this constant.
CRAWL_DATE = "2026-11-09"
# The target word is driven by the per-word layout above (LING_WORD / WORD).
# Everything specific to the word lives in the keyword / regex lists below and
# MUST be edited when the study moves to the next word.
# 目标词与「命中过滤串」都由 WORD 派生，避免两处不一致：
# TARGET_WORD 是给人看的名字，TARGET_CHAR 是真正拿去 text.find() 的字符串。
# 对多字词（如「破防」）同样成立，因为所有偏移量都用 len(TARGET_CHAR) 计算。
TARGET_WORD = WORD

# ---- 哈 (hā) keyword design, written 2026-11-09 -------------------------
# 哈 is the hardest word so far AT THE QUERY LEVEL, because it is
#   (a) one of the most frequent characters in Chinese comment text (哈哈哈),
#   (b) a transliteration syllable inside hundreds of proper nouns
#       (哈尔滨 哈佛 哈利波特 哈萨克 哈雷 哈勃 哈瓦那 哈达 …), and
#   (c) polyphonic: hā (breathe out / laugh / bend), hǎ (foolish; surname),
#       hà (rare).
# The bare character therefore yields a huge AND very noisy pool. The design
# below keeps the laughter layer (which is the actual language-change story)
# and the dictionary senses that anchor the prototypes, and deliberately does
# NOT query the transliteration layer: those records still arrive through the
# bare-char query and are removed later by drop_rules.py.
#
#   dictionary senses
#   1. 张口呼气   breathe out    哈气 哈手 哈气成霜
#   2. 笑声       laughter       哈哈 哈哈大笑 哈哈哈
#   3. 弯腰       bend           哈腰 点头哈腰
#   4. 哈欠       yawn           打哈欠 哈欠连天
#   5. 傻/蠢(方言) foolish        哈巴狗 哈儿 哈宝 哈戳戳
#   6. 粗心       careless       马大哈
#   7. 变质味     rancid         哈喇味 哈喇子(口水)
#   8. 敷衍       perfunctory   打哈哈
#   9. 哈哈镜     funhouse mirror
#
#   network layer (the reason the word is interesting)
#  10. 笑声延长   哈哈哈 / 哈哈哈哈            2016-   dominant
#  11. 疑问/没听清 哈？ / 哈                   2018-   single char + ？
#  12. 吓人       哈人 (= 吓人, 方言借入)      2020-
#  13. 可爱/萌     哈基米 (hajimi)              2023-   peak 2023-2024
#  14. 招呼       哈喽 / 哈啰 (hello)           2016-
#  15. 追捧       哈韩 / 哈日 / 哈美            2000s-  older network stratum
#  16. happy      哈皮                         2015-
#
# Design consequence: the laughter layer is a FORM (repetition of 哈), not a
# compound, so no query can isolate it -- every query returns it. That is a
# measurement problem rather than a crawling problem, which is why the bare
# char is kept and why the clean stage must keep 哈哈 as a form instead of
# collapsing it to a bare 哈 (see clean_corpus: it caps runs at 3).
SEARCH_KEYWORDS = [
    "哈",            # bare char: the only way senses 1-5, 10-11 get in
    "哈哈哈",        # laughter layer, explicitly
    "哈人",          # 2020+  吓人
    "哈基米",        # 2023+  hajimi
    "哈喽",          # greeting / loan
    "哈皮",          # happy
]

# Traditional / dictionary usages. Without these the corpus would be almost all
# laughter strings and would have no pre-2020 data on the concrete senses.
# 哈尔滨/哈萨克/哈佛 are NOT queried on purpose (see the design note above).
SEARCH_KEYWORDS_TRAD = [
    "哈欠",
    "哈腰",
    "哈气",
    "马大哈",
    "哈巴狗",
    "打哈哈",
    "哈哈镜",
]
SEARCH_ORDERS = ["totalrank", "click", "pubdate"]

# Productive 2020-2026 vocabulary. NOTE: 哈's network layer is again a family
# of INDEPENDENT items (哈人 / 哈基米 / 哈喽 / 哈皮 / 哈韩), not inflections of
# one compound -- and the biggest item, the laughter string 哈哈哈, is a form
# rather than a compound. That mixture is itself an RQ1 finding and must be
# stated, not smoothed over.
SEARCH_KEYWORDS_NEW = [
    "哈基米",
    "哈人",
    "哈哈哈",
    "哈喽",
    "哈皮",
    "哈韩",
    "哈日",
]
# years the network layer is expected in (哈人 from ~2020, 哈基米 2023-2024,
# 哈韩/哈日 are a 2000s stratum that barely reaches bilibili). One year early.
NEW_TERM_YEARS = (2020, 2026)

# Forms explicitly EXCLUDED from the corpus.
# Same rule as for 绷/塌/拉: a respelling that contains NO 哈 grapheme would
# conflate 字形 with 语音形式 and break the "target character" definition.
# 蛤？(= 哈？) is such a respelling: the character 蛤 carries no 哈, so it is
# out of scope by definition and stays excluded rather than being crawled.
EXCLUDED_FORMS = []

# 2026-11-09: supplementary stratum for 哈皮/哈批 = 傻/憨 (Sichuan dialect).
# The main harvest is dominated by the streamer 凉哈皮, so the dialect word has
# to be searched separately, with the dialect SPELLING (哈批) and dialect
# collocations. These queries are used by `--kwset hapi` and by crawl_hapi.py;
# they are NOT part of SEARCH_KEYWORDS, so a normal discovery run will not
# re-crawl this stratum.
SEARCH_KEYWORDS_DIALECT = [
    "哈批",
    "你个哈皮",
    "哈皮得很",
    "哈皮 四川话",
    "四川话 哈批",
    "重庆话 哈皮",
    "哈皮 方言",
    "哈皮 骂人",
    "贵州话 哈皮",
    "哈皮 什么意思",
]

# Senses 3-5 are thin on platform text and need dedicated sweeps:
#   (a) dictionary example sentences anchor the prototypes;
#   (b) the surface forms below are searched explicitly.
# A2 = 哈欠/弯腰/粗心/敷衍: health talk, crosstalk, dialect clips -- i.e.
# exactly the older, lower-engagement part of the whole pool.
A2_SEARCH_KEYWORDS = [
    "哈欠",
    "哈腰",
    "哈气",
    "马大哈",
    "哈巴狗",
    "打哈哈",
    "哈哈镜",
]
A2_PROTOTYPE_SOURCE = "词典例句（现代汉语词典/汉典，哈 义项：张口呼气/笑声/弯腰/哈欠/傻/粗心/敷衍）"

# comment/danmaku filter: a record counts as a hit if the target string appears.
# It is the FULL word (1 char for 绷, 2 for 破防); all offset code uses
# len(TARGET_CHAR), so a multi-character word stays correct.
TARGET_CHAR = WORD

# Sense-coding scheme -- STALE: this is 拉 (la)'s draft, kept only as a shape
# reference. 哈 needs its own sense table written from scratch at the analysis
# stage (do NOT reuse these labels).
# Freeze only after a 120-150 sentence pilot. Never derive categories from
# clustering output (that would make RQ3 circular).
# The pinyin is part of the definition because 拉 is polyphonic: 拉家常/拉呱 is
# lá, the pull/play senses are lā. Same grapheme, and the corpus cannot tell
# them apart without the collocation.
SENSE_SCHEMA_DRAFT = {
    "A1": {
        "label": "pull_drag",
        "pinyin": "lā",
        "definition": "拖拽、牵引：拉车、拉手、拉门、拉链、拉住、拉开、拉走",
        "cues": ["拉车", "拉手", "拉门", "拉链", "拉住", "拉开", "拉走", "拉绳"],
    },
    "A2": {
        "label": "play_bowed_string",
        "pinyin": "lā",
        "definition": "拉奏弦乐器：拉二胡、拉小提琴、拉琴",
        "cues": ["拉二胡", "拉小提琴", "拉琴", "拉弦"],
    },
    "A3": {
        "label": "excrete",
        "pinyin": "lā",
        "definition": "排泄：拉屎、拉肚子、拉稀（口语）",
        "cues": ["拉屎", "拉肚子", "拉稀"],
    },
    "A4": {
        "label": "recruit_rally",
        "pinyin": "lā",
        "definition": "招募集结：拉群、拉票、拉赞助、拉关系、拉帮结派",
        "cues": ["拉群", "拉票", "拉赞助", "拉关系", "拉帮结派", "拉人"],
    },
    "A5": {
        "label": "chat_dialect",
        "pinyin": "lá",
        "definition": "方言/口语聊天：拉家常、拉呱、拉话（lá，与 A1 同形不同音）",
        "cues": ["拉家常", "拉呱", "拉话"],
    },
    "B1": {
        "label": "block_social",
        "definition": "网络：拉黑（社交切断，2008+）",
        "cues": ["拉黑"],
    },
    "B2": {
        "label": "be_bad",
        "definition": "网络：拉胯/拉跨（差劲、不济）、拉了（完了、不行了），2020-2021+",
        "cues": ["拉胯", "拉跨", "拉了"],
    },
    "B3": {
        "label": "max_out",
        "definition": "网络：拉满（把程度/效果拉到最大），2020+",
        "cues": ["拉满"],
    },
    "B4": {
        "label": "trample",
        "definition": "网络：拉踩、拉爆（贬低他人 / 碾压式胜过），2019+",
        "cues": ["拉踩", "拉爆"],
    },
    "D": {
        "label": "other_unknown",
        "definition": "专名/音译/误匹配/无上下文无法判定（拉丁、沙拉、法拉利…）",
        "cues": [],
    },
}

# --------------------------------------------------------------------- seeds
RANDOM_SEED = 42

# time axis: search/crawl window actually observed in this environment
YEAR_MIN = 2016
YEAR_MAX = 2026

# ------------------------------------------------------------------ bilibili
BILI_API = "https://api.bilibili.com"
BILI_MIXIN_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]
UA_PC = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
REQUEST_DELAY = 1.2      # seconds, polite crawling
MAX_RETRY = 4

# ------------------------------------------------------------------- modeling
# Locked decision: encoder only, bidirectional, no decoder-only LLM.
EMBED_MODEL = "hfl/chinese-roberta-wwm-ext"
EMBED_FALLBACK = "hfl/chinese-macbert-base"   # already cached locally
EMBED_LAYERS = "last4_mean"                    # provisional; layer probe decides
MAX_LEN = 128
BATCH_SIZE = 16

if __name__ == "__main__":
    print("ROOT      :", ROOT)
    print("target    :", TARGET_WORD)
    print("keywords  :", len(SEARCH_KEYWORDS))
    print("year span :", YEAR_MIN, "-", YEAR_MAX)
    print("model     :", EMBED_MODEL)
    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"))
