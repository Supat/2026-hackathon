"""Evaluate the scorer on the 3-user sample against its ground truth.

    python -m care_score.evaluate

Reports MAE per item on the daily records (42 x 8) and on the 2-week means
(3 x 8, the metric used on the leaderboard), plus a check that the flag
thresholds reproduce 注意すべき変化_正解 exactly.  Also reports the MAE when the
介助負担 / リスク sentences are hidden, i.e. the pure regression fallback.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .predict import (ALL, FLAGS, GT_RECORDS, MEAN_COL, SCORE_COL, aggregate, derive_flags,
                      fit_fallback, load_lexicon, predict, split_sentences)

ROOT = Path(__file__).resolve().parent.parent
GT_2WEEK = ROOT / "data/sample_3users/care_hackathon_2week_summaries_ground_truth_3users.csv"


def mae_table(pred_rows, gt_rows, col):
    per = {}
    for k in ALL:
        p = np.array([float(r[col[k]]) for r in pred_rows])
        g = np.array([float(r[col[k]]) for r in gt_rows])
        per[k] = float(np.mean(np.abs(p - g)))
    return per


def report(title, per):
    print(f"\n{title}")
    print("  " + "  ".join(f"{k}:{v:.3f}" for k, v in per.items()))
    print(f"  overall MAE = {np.mean(list(per.values())):.4f}")


def main() -> None:
    gt = list(csv.DictReader(GT_RECORDS.open(encoding="utf-8")))
    gt2 = {r["利用者ID"]: r for r in csv.DictReader(GT_2WEEK.open(encoding="utf-8"))}
    lex = load_lexicon()
    w = fit_fallback()

    # 0) do the flag thresholds reproduce the ground-truth label column?
    bad = 0
    for r in gt:
        derived = derive_flags({k: float(r[SCORE_COL[k]]) for k in ALL})
        if derived != r["注意すべき変化_正解"]:
            bad += 1
            print("flag mismatch", r["利用者ID"], r["日付"], repr(derived), "vs", repr(r["注意すべき変化_正解"]))
    print(f"flag thresholds: {len(gt) - bad}/{len(gt)} records reproduced exactly")

    # 1) full pipeline on the sample records (sentences are in the lexicon, so this is a sanity check)
    daily, warnings = predict(gt, lex, w)
    for m in warnings:
        print("WARNING:", m)
    report("daily MAE, full pipeline (42 records)", mae_table(daily, gt, SCORE_COL))
    agg = aggregate(daily)
    agg_rows = [dict(agg[u], **{"注意日数": agg[u]["注意日数"]}) for u in sorted(agg)]
    gt_rows = [gt2[u] for u in sorted(agg)]
    report("2-week mean MAE, full pipeline (3 users x 8)", mae_table(agg_rows, gt_rows, MEAN_COL))
    print("  注意日数 pred/gt:", [(u, agg[u]["注意日数"], gt2[u]["注意日数"]) for u in sorted(agg)])

    # 2) hide the trailing 介助負担 / リスク / 注意点 sentences -> regression fallback only
    hidden = []
    for r in gt:
        s = split_sentences(r["記録内容"])
        keep = [x for i, x in enumerate(s) if i <= 6]
        hidden.append(dict(r, 記録内容="。".join(keep) + "。"))
    daily_h, _ = predict(hidden, lex, w)
    per = mae_table(daily_h, gt, SCORE_COL)
    report("daily MAE with 介助負担/リスク sentences hidden (fallback regression, in-sample)", per)
    agg_h = aggregate(daily_h)
    report("2-week mean MAE, sentences hidden", mae_table([agg_h[u] for u in sorted(agg_h)], gt_rows, MEAN_COL))

    # 3) leave-one-user-out for the fallback regression (honest estimate)
    print("\nleave-one-user-out fallback regression (介助負担 clipped to [0,3], リスク to [0,2] only where no sentence):")
    dom = ["食事", "入浴", "運動", "排泄", "睡眠", "認知_意欲"]
    X = np.array([[float(r[SCORE_COL[d]]) for d in dom] for r in gt])
    A = np.c_[X, np.ones(len(X))]
    users = np.array([r["利用者ID"] for r in gt])
    for key, hi in (("介助負担", 3), ("リスク", 2)):
        y = np.array([float(r[SCORE_COL[key]]) for r in gt])
        has = np.array([any(lex.get(s, {}).get("domain") == key for s in split_sentences(r["記録内容"])) for r in gt])
        errs = []
        for u in np.unique(users):
            tr = users != u
            wk, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
            te = (~tr) & (~has)
            errs += list(np.abs(np.clip(A[te] @ wk, 0, hi) - y[te]))
        print(f"  {key}: MAE {np.mean(errs):.3f} over {len(errs)} sentence-less records")


if __name__ == "__main__":
    main()
