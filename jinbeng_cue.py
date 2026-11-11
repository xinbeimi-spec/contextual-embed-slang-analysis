# -*- coding: utf-8 -*-
"""The 紧绷 sense: cue detection + adjudication plumbing.

Definition (user decision 2026-10-21)
  「紧绷义」= 神情紧绷 / 精神紧绷: facial or psychological TENSION.
  It is a sense of the word, not a surface form: the corpus expresses it as
  神经/精神/表情/神情/情绪 + 绷, as 紧绷, and as 绷着脸/绷着表情.

Why this is cue + human adjudication instead of a pure rule
  A wide rule (神情|表情|精神|神经 ... + 绷) fires on 85 items in the 2,663-item
  corpus, but most of them are 难绷 / 绷不住 false positives
  (一脸难绷, 这表情真绷不住). Only ~10 are the tension sense. A rule alone would
  therefore contaminate 评价义 and 忍不住义. So the rule is used ONLY to build a
  candidate list, every candidate is read in full, and the decision is stored
  per item in data/processed/jinbeng_adjudication.jsonl
      {clean_id, decision: keep|reject, cue, reason, text, decided_by, date}
  build_typology.py then applies the adjudicated list, and prints the
  rule-vs-adjudication reconciliation so the two can never silently drift.

What counts as 紧绷义 (KEEP)
  * the tensing is in the FACE / EXPRESSION / MIND / NERVES / ATMOSPHERE:
      「神经紧绷会猝死」「时刻紧绷神经」「精神紧绷，需要化学极乐一下」
      「脸上的肌肉都是紧绷的」「别绷着脸了」「全程绷着脸」
  * the tension is the point of the clause, even if a body part is named
      「一紧张，肌肉绷紧，动作就僵了」  (cause = nerves)

What does NOT count (REJECT)
  * physical tautness of an object or of a body part as such, with no
    psychological frame: 脚踝紧绷 / 皮肤紧绷 / 肚子发硬紧绷 / 酸痛紧绷的肌肉
    / 绷带 / 绷不紧的桌面 — these stay 物理义 (or 非义项).
  * 难绷 / 绷不住 evaluative or failure-to-hold readings
      「一脸难绷」「这表情真绷不住」
  * mention-type items (repetition, wordplay on the string)
"""
import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

ADJ = os.path.join(C.PROCESSED, "jinbeng_adjudication.jsonl")

# ---- cue families (candidate generation ONLY; never a label on its own)
FACE = r"(?:神情|表情|神色|面孔|脸色|脸|眉头|嘴角|笑容)"
PSY = r"(?:神经|精神|情绪|心态|心情|心理|气氛|氛围|空气|关系|局面|气场|状态)"
BODY = r"(?:肌肉|身体|全身|整个人|身心|腿|脚|手|肩|背|腰|脖子|头皮|肚子|腹部|腹肌|膝盖|小腿|核心)"

CUES = [
    # (cue name, regex)
    ("literal_jinbeng", r"紧绷"),
    ("jinbeng_reduplicated", r"紧绷绷"),
    ("face_psy_subject_then_beng", FACE + r".{0,4}绷|" + PSY + r".{0,4}绷"),
    ("beng_then_face_psy_subject", r"绷.{0,4}" + FACE + r"|绷.{0,4}" + PSY),
    ("beng_zhe_face", r"绷着(?:个|着)?(?:脸|表情|面孔|神经|面)"),
    ("beng_de_jin", r"绷得.{0,2}紧|绷的.{0,2}紧|绷(?:太|很|好)紧"),
    ("body_tension_with_psy_frame",
     r"(?:紧张|焦虑|压力|害怕|担心|拘束|不安).{0,12}" + BODY + r".{0,6}绷"
     r"|" + BODY + r".{0,6}绷.{0,12}(?:紧张|焦虑|压力|不安)"),
]
_CUES = [(n, re.compile(p)) for n, p in CUES]

# Cue families that work by linking a facial/psychological subject to 绷. They
# are meaningless when the 绷 in the sentence is already part of a DIFFERENT
# target form (绷带 / 难绷 / 绷不住): in "绷带没脸吧" or "一脸难绷" the window
# happens to contain 脸, and the relational cue fires on nothing but the
# neighbourhood. Suppressing those families removes exactly the false positives
# that dominated the 74-candidate trial run, without touching literal 紧绷.
RELATIONAL = {"face_psy_subject_then_beng", "beng_then_face_psy_subject",
              "beng_zhe_face", "beng_de_jin", "body_tension_with_psy_frame"}
BLOCKING_FORMS = ("难绷", "绷带", "绷不住")


def cues_of(text):
    """Which cue families fire on this text (empty list = not a candidate)."""
    t = text or ""
    fired = [n for n, rx in _CUES if rx.search(t)]
    if any(f in t for f in BLOCKING_FORMS):
        fired = [n for n in fired if n not in RELATIONAL]
    return fired


def load_adjudication(path=ADJ):
    """clean_id -> record, for decision == 'keep' only."""
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("decision") == "keep":
            out[r["clean_id"]] = r
    return out


def is_jinbeng(clean_id, keep_map):
    return clean_id in keep_map
