"""Derive the sentence -> (domain, score) lexicon from the 3-user ground truth.

Every daily record is a sequence of template sentences separated by 「。」.
Sentences 1..6 always describe, in order, 食事 / 入浴 / 運動 / 排泄 / 睡眠 / 認知・意欲.
Later sentences are optional: a 介助負担 sentence, a リスク note, and a
「注意点として、…がみられる」 summary.  Inside the sample data every sentence
maps to exactly one score, so the lexicon is a plain lookup table.

Sentences that occur only in the 7-user test set are added by hand in
MANUAL_ENTRIES with a rationale; ambiguous ones get a fractional (expected)
score, which is what minimises MAE on the per-user 14-day mean.

Run:  python -m care_score.build_lexicon
Writes care_score/lexicon.json
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_RECORDS = ROOT / "data/sample_3users/care_hackathon_ground_truth_records_3users.csv"
OUT = Path(__file__).resolve().parent / "lexicon.json"

DOMAINS = ["食事", "入浴", "運動", "排泄", "睡眠", "認知_意欲"]
SCORE_COLS = {d: f"{d}_score" for d in DOMAINS}
SCORE_COLS["介助負担"] = "介助負担_score"
SCORE_COLS["リスク"] = "リスク_score"

# Sentences in the trailing (optional) part of a record, classified by hand
# from the sample data.  介助負担 sentences always agree with 介助負担_score,
# note sentences always agree with リスク_score.
BURDEN_SENTENCES = {
    "日常動作に介助が必要な場面が多い",
    "拒否があり対応に時間を要した",
    "複数場面で介助を要した",
    "入浴または排泄で介助が必要",
    "ふらつきがあり常時見守りを要した",
    "介助負担は少ない",
    "見守り不要で実施できた",
}
NOTE_SENTENCES = {
    "経過は概ね安定",
    "食欲低下や水分摂取不足に注意",
    "経過観察が必要",
    "夜間覚醒があり注意が必要",
    "傾眠や軽度の混乱に注意",
    "軽度の疲労感はあるが大きな問題なし",
    "特記事項なし",
    "状態は安定",
    "夜間覚醒後の様子を確認",
    "服薬拒否があり確認が必要",
}

# Sentences seen only in the test set.  {sentence: (domain, score, why)}
MANUAL_ENTRIES: dict[str, tuple[str, float, str]] = {
    # 食事 -------------------------------------------------------------
    "数口から3割程度の摂取にとどまる": ("食事", 1, "worse than 少量のみ(2); 食事低下 flagged in the same record"),
    # 入浴 -------------------------------------------------------------
    "一部介助で入浴実施": ("入浴", 4, "between 介助下(3) and 声かけのみ/自立(5); U005/U009 profiles say 自立または一部介助"),
    "洗体時のみ介助を要した": ("入浴", 4, "partial help only, same tier as 一部介助"),
    "見守りと一部介助で入浴を終えた": ("入浴", 4, "partial help only, same tier as 一部介助"),
    "入浴は行わず清拭で対応": ("入浴", 2, "清拭 = 2 in sample (本日は清拭のみ実施, 部分清拭); 注意点 never lists 入浴拒否・困難 with it"),
    # 運動 -------------------------------------------------------------
    "集団体操に最後まで参加": ("運動", 5, "full participation; 体操に概ね参加できた is 4"),
    "車椅子でホールまで移動し活動を見学": ("運動", 1.5, "watched only: between 不参加(1) and 短時間参加(2); no 注意点 flags to disambiguate"),
    # 睡眠 -------------------------------------------------------------
    "睡眠がやや浅い": ("睡眠", 3.5, "milder than 眠りが浅く日中の眠気が強い(2); 注意点 never lists 睡眠不良 with it, so >=3; 3 or 4"),
    "夜間覚醒が3回以上あり": ("睡眠", 1.5, "worse than 覚醒2回(3); 睡眠不良 flagged so <=2; 1 or 2"),
    # 介助負担 -----------------------------------------------------------
    "混乱があり介助困難": ("介助負担", 5, "same tier as 拒否があり対応に時間を要した(5); 介助負担大 flagged"),
    "必要時の声かけ程度で過ごせた": ("介助負担", 1, "low but not zero; U009 profile 介助負担は全体として低い"),
    "軽い確認のみで対応可能": ("介助負担", 1, "low but not zero"),
    # リスク notes -------------------------------------------------------
    "便秘傾向と睡眠状態に注意": ("リスク", 4, "always co-occurs with 注意点…高リスク, so >=4; treat like other 〜に注意 notes (4)"),
    "ふらつきがあり転倒に注意": ("リスク", 4.5, "fall warning with 高リスク flag; 4 or 5"),
    "強い拒否がみられ重点的な対応が必要": ("リスク", 5, "strongest wording; 服薬拒否があり確認が必要 is 5"),
}

FLAG_RE = re.compile(r"^注意点として、(.+)がみられる$")


def split_sentences(text: str) -> list[str]:
    return [s for s in text.split("。") if s]


def build() -> dict:
    rows = list(csv.DictReader(GT_RECORDS.open(encoding="utf-8")))
    votes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        sents = split_sentences(r["記録内容"])
        # positional slots 1..6
        for i, dom in enumerate(DOMAINS, start=1):
            votes[sents[i]][dom][int(r[SCORE_COLS[dom]])] += 1
        for s in sents[7:]:
            if s in BURDEN_SENTENCES:
                votes[s]["介助負担"][int(r["介助負担_score"])] += 1
            elif s in NOTE_SENTENCES:
                votes[s]["リスク"][int(r["リスク_score"])] += 1
            elif FLAG_RE.match(s):
                pass  # handled structurally by the predictor
            else:
                raise ValueError(f"unclassified trailing sentence: {s!r}")

    lexicon: dict[str, dict] = {}
    conflicts = []
    for s, doms in votes.items():
        assert len(doms) == 1, (s, doms)
        dom, cnt = next(iter(doms.items()))
        if len(cnt) > 1:
            conflicts.append((s, dict(cnt)))
        score = sum(k * v for k, v in cnt.items()) / sum(cnt.values())
        lexicon[s] = {"domain": dom, "score": round(score, 3), "n": sum(cnt.values()), "source": "sample_3users"}
    for s, (dom, score, why) in MANUAL_ENTRIES.items():
        assert s not in lexicon, s
        lexicon[s] = {"domain": dom, "score": score, "n": 0, "source": "manual", "why": why}

    if conflicts:
        print("sentences with more than one score in the sample (using mean):")
        for c in conflicts:
            print("  ", c)
    return lexicon


def main() -> None:
    lex = build()
    OUT.write_text(json.dumps(lex, ensure_ascii=False, indent=1), encoding="utf-8")
    by_dom = Counter(v["domain"] for v in lex.values())
    print(f"wrote {OUT} with {len(lex)} sentences: {dict(by_dom)}")


if __name__ == "__main__":
    main()
