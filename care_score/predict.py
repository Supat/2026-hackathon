"""Score daily care records and aggregate them into the 2-week submission.

Usage:
    python -m care_score.predict --records data/test_7users/care_hackathon_student_records_7users_14days.csv \
        --summaries data/test_7users/care_hackathon_2week_summaries_7users.csv --out submission

Outputs (in --out):
    <stem>_records_prediction.csv   daily scores, same columns as the 3-user ground-truth records CSV
    <stem>_2week_prediction.csv     per-user means, same columns as the 3-user ground-truth 2-week CSV
    <stem>_simple.csv               the 「user_id,食事,…,リスク」 layout shown on the hackathon page

How a record is scored
----------------------
1. Split on 「。」.  Sentences 1..6 are looked up in lexicon.json and give the six
   domain scores directly (the sentences are templates; each one has a fixed score).
2. A trailing 介助負担 sentence gives 介助負担 directly; otherwise 介助負担 is
   predicted by a linear regression on the six domain scores, clipped to [0, 3]
   (in the sample, records with 介助負担 >= 4 always carry such a sentence).
3. A trailing リスク note gives リスク directly; otherwise a regression clipped to
   [0, 2] (records with リスク >= 3 always carry a note).
4. A 「注意点として、…がみられる」 sentence lists the flags that fired that day.
   In the sample it appears exactly when リスク >= 4, and each flag is a hard
   threshold on one score, so the flags are used as constraints:
       食事低下 <=> 食事 <= 2      入浴拒否・困難 <=> 入浴 <= 1   活動低下 <=> 運動 <= 1
       睡眠不良 <=> 睡眠 <= 2      意欲低下・混乱 <=> 認知_意欲 <= 2
       介助負担大 <=> 介助負担 >= 4  高リスク <=> リスク >= 4
5. 注意すべき変化 for a day is derived from the rounded scores with the same
   thresholds; 注意日数 is the number of days with at least one flag.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEXICON_PATH = HERE / "lexicon.json"
GT_RECORDS = ROOT / "data/sample_3users/care_hackathon_ground_truth_records_3users.csv"

DOMAINS = ["食事", "入浴", "運動", "排泄", "睡眠", "認知_意欲"]
ALL = DOMAINS + ["介助負担", "リスク"]
SCORE_COL = {k: f"{k}_score" for k in ALL}
MEAN_COL = {k: f"{k}_score_平均" for k in ALL}
SIMPLE_COL = dict(zip(ALL, ["食事", "入浴", "運動・歩行", "排泄自立度", "睡眠状態", "認知・意欲", "介助負担", "リスク"]))

# flag name -> (key, comparator, threshold).  Order = order used in the ground truth label.
FLAGS = [
    ("食事低下", "食事", "<=", 2),
    ("入浴拒否・困難", "入浴", "<=", 1),
    ("活動低下", "運動", "<=", 1),
    ("睡眠不良", "睡眠", "<=", 2),
    ("意欲低下・混乱", "認知_意欲", "<=", 2),
    ("介助負担大", "介助負担", ">=", 4),
    ("高リスク", "リスク", ">=", 4),
]
FLAG_RE = re.compile(r"^注意点として、(.+)がみられる$")

# keyword fallback for sentences missing from the lexicon (never triggered on the
# sample or the 7-user test set; kept so the script degrades gracefully).
KEYWORDS = [
    ("食事", ("食事", "摂取", "食欲", "完食", "主食", "副食", "朝食", "昼食", "夕食")),
    ("入浴", ("入浴", "清拭", "洗体", "浴室")),
    ("運動", ("体操", "歩行", "散歩", "運動", "レクリエーション", "活動")),
    ("排泄", ("排泄", "トイレ", "衣服")),
    ("睡眠", ("睡眠", "眠", "覚醒", "傾眠")),
    ("認知_意欲", ("表情", "会話", "意欲", "混乱", "反応", "交流", "参加")),
    ("介助負担", ("介助", "見守り", "対応")),
    ("リスク", ("注意", "確認", "安定", "特記", "リスク")),
]


def split_sentences(text: str) -> list[str]:
    return [s for s in text.split("。") if s]


def load_lexicon(path: Path | None = None) -> dict[str, dict]:
    return json.loads((path or LEXICON_PATH).read_text(encoding="utf-8"))


def fit_fallback() -> dict[str, np.ndarray]:
    """Linear regression 介助負担 / リスク ~ six domain scores, fit on the sample GT."""
    rows = list(csv.DictReader(GT_RECORDS.open(encoding="utf-8")))
    X = np.array([[float(r[SCORE_COL[d]]) for d in DOMAINS] for r in rows])
    A = np.c_[X, np.ones(len(X))]
    w = {}
    for key in ("介助負担", "リスク"):
        y = np.array([float(r[SCORE_COL[key]]) for r in rows])
        w[key], *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def keyword_domain(sentence: str) -> str | None:
    for dom, kws in KEYWORDS:
        if any(k in sentence for k in kws):
            return dom
    return None


def score_record(text: str, lex: dict, w: dict, warnings: list[str]) -> dict:
    sents = split_sentences(text)
    scores: dict[str, float] = {}
    explicit: set[str] = set()
    flags: set[str] | None = None

    for i, s in enumerate(sents):
        if i == 0:
            continue  # opening sentence (「午前はバイタル確認後…」) carries no score
        m = FLAG_RE.match(s)
        if m:
            flags = {f.strip() for f in m.group(1).split("、")}
            continue
        entry = lex.get(s)
        if entry is None:
            dom = DOMAINS[i - 1] if 1 <= i <= 6 else keyword_domain(s)
            warnings.append(f"unknown sentence (slot {i}, assumed {dom}): {s}")
            if dom is None:
                continue
            # neutral guess: middle of the scale for that domain
            scores[dom] = 3.0 if dom in DOMAINS else (3.0 if dom == "介助負担" else 2.0)
            explicit.add(dom)
            continue
        dom = entry["domain"]
        if dom in scores:
            warnings.append(f"duplicate domain {dom} in record: {s}")
        scores[dom] = float(entry["score"])
        explicit.add(dom)

    for d in DOMAINS:
        if d not in scores:
            warnings.append(f"missing domain {d}, using 3.0: {text[:40]}…")
            scores[d] = 3.0

    x = np.array([scores[d] for d in DOMAINS] + [1.0])
    if "介助負担" not in scores:
        scores["介助負担"] = float(np.clip(x @ w["介助負担"], 0, 3))
    if "リスク" not in scores:
        scores["リスク"] = float(np.clip(x @ w["リスク"], 0, 2))

    if flags is not None:
        # the 注意点 sentence only appears on days with リスク >= 4
        scores["リスク"] = max(scores["リスク"], 4.0)
        for name, key, op, thr in FLAGS:
            if name in flags:
                scores[key] = min(scores[key], thr) if op == "<=" else max(scores[key], thr)
            else:
                scores[key] = max(scores[key], thr + 1) if op == "<=" else min(scores[key], thr - 1)

    return scores


def derive_flags(scores: dict[str, float]) -> str:
    fired = []
    for name, key, op, thr in FLAGS:
        v = round(scores[key])
        if (op == "<=" and v <= thr) or (op == ">=" and v >= thr):
            fired.append(name)
    return " / ".join(fired)


def predict(records: list[dict], lex: dict, w: dict) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    out = []
    for r in records:
        sc = score_record(r["記録内容"], lex, w, warnings)
        row = {"利用者ID": r["利用者ID"], "日付": r["日付"]}
        row.update({SCORE_COL[k]: sc[k] for k in ALL})
        row["注意すべき変化"] = derive_flags(sc)
        out.append(row)
    return out, warnings


def aggregate(daily: list[dict]) -> dict[str, dict]:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for r in daily:
        by_user[r["利用者ID"]].append(r)
    agg = {}
    for uid, rows in by_user.items():
        a = {MEAN_COL[k]: round(float(np.mean([r[SCORE_COL[k]] for r in rows])), 2) for k in ALL}
        a["注意日数"] = sum(1 for r in rows if r["注意すべき変化"])
        a["期間開始"] = min(r["日付"] for r in rows)
        a["期間終了"] = max(r["日付"] for r in rows)
        agg[uid] = a
    return agg


def fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def write_outputs(daily, agg, records, summaries, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    feat = {r["利用者ID"]: r.get("利用者特徴", "") for r in records}
    sum_rows = {r["利用者ID"]: r for r in summaries} if summaries else {}

    # 1) daily, same columns as care_hackathon_ground_truth_records_3users.csv
    p = out_dir / f"{stem}_records_prediction.csv"
    cols = ["利用者ID", "日付", "利用者特徴", "記録内容"] + [SCORE_COL[k] for k in ALL] + ["注意すべき変化_正解"]
    with p.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r, src in zip(daily, records):
            uf = src.get("利用者特徴") or sum_rows.get(r["利用者ID"], {}).get("利用者特徴", "")
            wr.writerow([r["利用者ID"], r["日付"], uf, src["記録内容"]] + [fmt(r[SCORE_COL[k]]) for k in ALL] + [r["注意すべき変化"]])
    written.append(p)

    # 2) 2-week, same columns as care_hackathon_2week_summaries_ground_truth_3users.csv
    p = out_dir / f"{stem}_2week_prediction.csv"
    cols = ["利用者ID", "期間開始", "期間終了", "利用者特徴", "2週間サマリー"] + [MEAN_COL[k] for k in ALL] + ["注意日数"]
    with p.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for uid in sorted(agg):
            a = agg[uid]
            s = sum_rows.get(uid, {})
            wr.writerow([uid, s.get("期間開始", a["期間開始"]), s.get("期間終了", a["期間終了"]),
                         s.get("利用者特徴", feat.get(uid, "")), s.get("2週間サマリー", "")]
                        + [fmt(a[MEAN_COL[k]]) for k in ALL] + [a["注意日数"]])
    written.append(p)

    # 3) simple layout from the hackathon page
    p = out_dir / f"{stem}_simple.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["user_id"] + [SIMPLE_COL[k] for k in ALL])
        for uid in sorted(agg):
            wr.writerow([uid] + [fmt(agg[uid][MEAN_COL[k]]) for k in ALL])
    written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="CSV with 利用者ID,日付,記録内容")
    ap.add_argument("--summaries", help="CSV with 利用者ID,期間開始,期間終了,利用者特徴,2週間サマリー (copied into the output)")
    ap.add_argument("--out", default="submission", help="output directory")
    ap.add_argument("--stem", default=None, help="output file prefix (default: derived from --records)")
    ap.add_argument("--lexicon", type=Path, default=LEXICON_PATH,
                    help="lexicon to use (default: care_score/lexicon.json; e.g. care_score/lexicon_qwen2.5-14b.json)")
    args = ap.parse_args(argv)

    records = list(csv.DictReader(open(args.records, encoding="utf-8")))
    summaries = list(csv.DictReader(open(args.summaries, encoding="utf-8"))) if args.summaries else []
    lex = load_lexicon(args.lexicon)
    w = fit_fallback()
    daily, warnings = predict(records, lex, w)
    agg = aggregate(daily)

    stem = args.stem or re.sub(r"_(student_)?records|_\d+days", "", Path(args.records).stem)
    written = write_outputs(daily, agg, records, summaries, Path(args.out), stem)

    for wmsg in warnings:
        print("WARNING:", wmsg, file=sys.stderr)
    print(f"{len(records)} records, {len(agg)} users")
    print("user_id," + ",".join(SIMPLE_COL[k] for k in ALL) + ",注意日数")
    for uid in sorted(agg):
        print(uid + "," + ",".join(fmt(agg[uid][MEAN_COL[k]]) for k in ALL) + f",{agg[uid]['注意日数']}")
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
