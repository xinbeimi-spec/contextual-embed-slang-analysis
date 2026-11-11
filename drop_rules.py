# -*- coding: utf-8 -*-
"""Drop / exclusion rules for the cleaning stage -- 哈 (hā), 2026-11-09.

The two-way split is unchanged from 绷/塌/拉, because the consequences differ:

1. IRRELEVANT  -> the item is NOT a use of the character at all.
   Removed from the analysis table and written, with a reason, to
   data/processed/<slug>/dropped_items.jsonl so the decision stays auditable
   and reversible. Never deleted from the raw harvest.

2. TINY_SENSE  -> the item IS a use of 哈, but instantiates a sense that occurs
   a handful of times in the whole corpus. Per the 2026-10-21 decision such
   senses are too rare to carry a category: they stay in the corpus (deleting
   genuine senses would bias the material) but are flagged
   `excluded_from_typology` and kept OUT of the typology denominators.

What is specific to 哈 (measured on the 5,806-record 2026-11-09 harvest):

* 音译/专名 is the main noise source: 哈尔滨, 哈萨克(斯坦), 哈佛, 哈利波特,
  哈士奇, 哈工大/哈理工/哈啤 ... 467 records (8.0%) would be pure proper
  nouns. As with 拉, the rule is NOT "the text contains 哈尔滨" -- that would
  delete "哈尔滨人表示哈哈哈" (a genuine laughter use). The rule is: after
  masking those tokens, EVERY 哈 is gone. NOISE_TOKENS is exported so
  clean_corpus.py can put the target offset on the first *meaningful* 哈.

* 哈皮 is a special case and it is NOT noise: the user confirmed on 2026-11-09
  that 哈皮 is a Sichuan-dialect word (哈 = 傻/憨, so 哈皮/哈批 = fool). But
  the corpus is dominated by a bilibili streamer whose handle IS 凉哈皮: of the
  796 records containing 哈皮, 750 have 哈 only inside 哈皮, and the top
  comment threads are all his videos. Dropping every 哈皮 would delete the
  dialect word; keeping every 哈皮 would let one person's nickname pose as a
  dialect sense. The rule is therefore VIDEO-LEVEL: inside a video that is
  demonstrably his (handle in the title, or >= HAPI_NAME_MIN records that use
  哈皮 as the only 哈) 哈皮 counts as a nickname and is dropped; everywhere
  else it stays as the dialect item and is flagged as a tiny sense.

* 单字/超短评论: `哈？` (huh?) is a genuine interrogative use and must survive
  the MIN_CHARS filter, even though it is 2 characters. clean_corpus.py keeps
  pure 哈-strings of length >= 2; this module only has to not drop them.

* 二哈 (husky nickname) is deliberately NOT in NOISE_TOKENS: it is a
  lexicalised colloquial compound rather than a proper noun, and it is left in
  the corpus for the analysis stage to treat as a non-sense usage if needed.
"""
import re
import collections

# Tokens in which 哈 is a transliteration syllable or part of a proper name and
# therefore carries no lexical meaning of its own. Sorted longest-first so that
# 哈利波特 is masked before 哈利 and 哈萨克斯坦 before 哈萨克.
# Extend after inspecting dropped_items.jsonl / the residual stratum.
NOISE_TOKENS = [
    # places / ethnonyms / universities / brands
    "哈尔滨", "哈萨克斯坦", "哈萨克", "哈佛", "哈弗", "哈工大", "哈理工",
    "哈工程", "哈师大", "哈医大", "哈商大", "哈三中", "哈九中", "哈一中",
    "哈院",
    "哈啤", "哈市", "哈局", "哈站", "哈东", "哈西", "哈瓦那", "哈达",
    "哈马斯", "哈耶克", "哈维尔", "哈桑", "哈迪", "哈根达斯", "哈勃",
    "哈雷", "娃哈哈",
    # fiction / characters (a very dense source of 哈 on bilibili)
    "哈利波特", "哈利", "哈姆雷特", "哈里",
    # 2026-11-09 (cluster round 1): Harry-Potter shipping abbreviations. These
    # are 哈 = Harry (德哈 Draco/Harry, 斯哈 Snape/Harry, 哈德, 哈赫), i.e. the
    # proper-noun layer, but not caught by "哈利". They contaminated the 哈基米
    # basin of the first clustering run (德哈 alone = 54 records).
    "德哈", "斯哈", "哈德", "哈赫",
    # places / brands / handles found in the residual stratum
    "哈密瓜", "努尔哈赤", "京哈线", "嘻哈", "大哈鱼",
    # animals: transliteration of "husky", no lexical 哈
    "哈士奇",
    # a bilibili streamer's handle (see module docstring)
    "凉哈皮",
]
NOISE_TOKENS.sort(key=len, reverse=True)

# A video counts as the streamer's if it has at least this many records whose
# only 哈 sits inside 哈皮. His vlogs have neutral titles, so the count is the
# only reliable signal for those.
HAPI_NAME_MIN = 5

# 2026-11-09: the streamer-level signal above misses threads about him that are
# not on his channel ("哈皮真的很敬业 第五一哥实至名归"). Text-level person
# reference is detectable by what follows the name; a Sichuan-dialect 哈皮
# never continues with these function words (it is a predicate: "你个哈皮",
# "哈皮得很"). The dialect cues below override the person rule.
RE_HAPI_PERSON = re.compile(
    r"哈皮(的|是|在|真|确|要|被|居|第|讲|说|玩|打|直|怎|今|昨|下|这|那|老|"
    r"哥|姐|更|新|粉|水|视|主|和|跟|给|帮|带|教|一|两|也|都|就|还|又|不|没|"
    r"很|太|会|能|可|想|爱|帅|敬|厉|牛|强|菜)")
DIALECT_CUES = ("傻", "憨", "瓜", "批", "锤子", "方言", "四川", "成都", "重庆",
                "宝器", "戳戳", "龟儿子", "哈儿", "贵州")

# 2026-11-09: metalinguistic comments -- the speaker TALKS ABOUT the word
# ("这个哈皮在贵州话里面是骂人的意思", "以为是那个骂人的哈皮", "happy 哈皮
# 一语双关"). These are not usages of a sense, but they are the strongest
# evidence that the dialect sense exists and is recognised by speakers, so
# they must not be thrown away with the nickname records. They are kept and
# flagged excluded_from_typology (they do not instantiate a sense).
META_WORDS = ("骂人", "骂", "意思", "方言", "话里", "说法", "谐音", "音译",
              "英文", "英语", "happy", "四川", "重庆", "贵州", "名字耽误")
RE_META = re.compile(
    r"(?:哈皮|哈批)[^。！？!?\n]{0,24}(?:" + "|".join(META_WORDS) + r")"
    r"|(?:" + "|".join(META_WORDS) + r")[^。！？!?\n]{0,24}(?:哈皮|哈批)")


_META_NAME_CTX = ("老师", "主播", "解说", "直播", "录播", "有意思", "没意思")


def meta_about_word(text):
    """True when the comment talks ABOUT the word 哈皮/哈批, not about a name.

    Two false-positive families are excluded explicitly: the streamer's
    catchphrase "哈皮老师你什么意思" (name + 老师/主播/解说) and ordinary
    "很有意思" (interesting), which is not about the word at all.
    """
    t = text or ""
    for m in RE_META.finditer(t):
        span = m.group(0)
        if any(w in span for w in _META_NAME_CTX):
            continue
        return True
    return False

# ---------------------------------------------------------------- irrelevant
IRRELEVANT = [
    # pure onomatopoeia / filler: "哈" repeated with nothing else is the
    # laughter use itself, so it is NOT dropped here -- it is kept by
    # clean_corpus.py as a short pure-哈 item. Nothing to add yet.
]

# Ad signals, scored (same scheme as 拉, 2026-10-31).
AD_HARD = ["购买链接", "小黄车", "橱窗", "加微信", "加V"]
AD_SOFT = ["淘宝", "旗舰店", "优惠券", "包邮", "私信我"]
AD_DECOR = ["★", "☆"]


def ad_score(text):
    hard = sum(1 for m in AD_HARD if m in text)
    soft = sum(1 for m in AD_SOFT if m in text)
    decor = sum(1 for m in AD_DECOR if m in text)
    return hard, soft + 0.5 * decor


# ---------------------------------------------------------------- tiny sense
# 哈皮 survives only where it is not the streamer's nickname; in this harvest
# that is a few dozen records, so it cannot carry a category yet.
TINY_SENSE = [
    ("dialect_hapi",
     "极零散方言义：四川话 哈皮/哈批 = 傻、憨（与人名 凉哈皮 同形，仅保留非人名用例），n 很小",
     re.compile(r"哈皮|哈批")),
]


def _mask(text):
    t = text or ""
    for tok in NOISE_TOKENS:
        if tok in t:
            t = t.replace(tok, "\u0000" * len(tok))
    return t


def noise_only(text):
    """True when every 哈 in `text` belongs to a 音译/专名 token."""
    return "哈" not in _mask(text)


def hapi_is_name_only(text):
    """True when 哈皮 is the only carrier of 哈 in the item."""
    return "哈皮" in (text or "") and "哈" not in _mask(text).replace("哈皮", "")


def hapi_name_videos(rows):
    """bvid set of videos where 哈皮 demonstrably names the streamer.

    Two independent signals (union):
      * the handle appears in the video title;
      * the thread contains >= HAPI_NAME_MIN records whose only 哈 is 哈皮.
    """
    cnt = collections.Counter()
    named = set()
    for r in rows:
        t = r.get("text_norm") or r.get("text_raw") or ""
        b = r.get("thread_bvid")
        if b is None:
            continue
        if "哈皮" in t and "哈" not in _mask(t).replace("哈皮", ""):
            cnt[b] += 1
        title = re.sub(r"\s+", "", r.get("thread_title") or "")
        if "哈皮" in title or "凉哈皮" in title:
            named.add(b)
    named |= {b for b, n in cnt.items() if n >= HAPI_NAME_MIN}
    return named


def hapi_dialect_evidence(text):
    """True when a record from the supplementary stratum really shows the
    dialect word (or talks about it as a dialect word), rather than the
    streamer / the cartoon 哈皮父子.

    2026-11-09: the supplement was crawled from dialect queries, but the word
    is so dominated by names that most hits are still names; without this
    filter the stratum would import ~280 name records into a "dialect" bucket.
    """
    t = text or ""
    if "哈批" in t:
        return True
    if meta_about_word(t):
        return True
    if "哈皮" in t and any(w in t for w in DIALECT_CUES + ("骂", "难听", "名字耽误")):
        return True
    return False


def classify(text, bvid=None, name_bvids=frozenset()):
    """Return (drop_reason | None, excluded_from_typology | None)."""
    t = text or ""
    hard, soft = ad_score(t)
    if hard or soft >= 2:
        return ("ad_marketing: 广告/带货模板（硬标记 购买链接/小黄车/橱窗/加微信，"
                "或 ≥2 个软标记）", None)
    for _name, reason, rx in IRRELEVANT:
        if rx.search(t):
            return f"{_name}: {reason}", None
    if noise_only(t):
        return ("proper_noun_only: 目标字全部落在音译/专名里"
                "（哈尔滨/哈萨克/哈佛/哈利波特/哈士奇…），句内没有 哈 的实义用法"), None
    if meta_about_word(t):
        return None, ("metalinguistic: 元语言评论（说话人谈论 哈皮/哈批 的"
                      "意思或方言色彩），保留但不计入义项分母")
    if hapi_is_name_only(t) and bvid in name_bvids:
        return ("nickname_only: 哈皮 是人名/昵称（凉哈皮 等 UP 主），"
                "不是四川方言义"), None
    # text-level person reference ("哈皮真的很敬业"), with a dialect override
    if "哈皮" in t and "哈批" not in t and RE_HAPI_PERSON.search(t):
        if not any(w in t for w in DIALECT_CUES):
            return ("nickname_only: 哈皮 在此句中是人物指称（主播 凉哈皮），"
                    "不是四川方言义"), None
    for _name, reason, rx in TINY_SENSE:
        if rx.search(t):
            return None, f"{_name}: {reason}"
    return None, None
