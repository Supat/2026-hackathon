# 介護データ標準化チャレンジ — 第1回 医歯理工連携 医療データハッカソン

Rule-based scorer for the [2026 医療データハッカソン](http://www.bioif.iir.titech.ac.jp/hackathon/)
(国際医工共創研究院, 2026-09-26). It reads 14 days of free-text care records per user and
estimates the 8 standardised scores (食事・入浴・運動・排泄・睡眠・認知/意欲・介助負担・リスク),
then aggregates them to the per-user 2-week means that are scored by MAE.

## Submission files

| file | format |
|---|---|
| `submission/care_hackathon_7users_2week_prediction.csv` | same columns as `care_hackathon_2week_summaries_ground_truth_3users.csv` (7 users × 8 means + 注意日数) — **upload this one** |
| `submission/care_hackathon_7users_records_prediction.csv` | daily scores, same columns as `care_hackathon_ground_truth_records_3users.csv` |
| `submission/care_hackathon_7users_simple.csv` | the `user_id,食事,…,リスク` layout shown on the hackathon page |

## Reproduce

```bash
pip install -r requirements.txt
python -m care_score.build_lexicon        # sample GT -> care_score/lexicon.json
python -m care_score.evaluate             # sanity check on the 3 sample users
python -m care_score.predict \
    --records   data/test_7users/care_hackathon_student_records_7users_14days.csv \
    --summaries data/test_7users/care_hackathon_2week_summaries_7users.csv \
    --out submission
```

## How it works

The records are template-generated. Each day is a 「。」-separated list of sentences whose
first six slots are always 食事 → 入浴 → 運動 → 排泄 → 睡眠 → 認知・意欲, followed by optional
介助負担 / リスク note / 「注意点として、…がみられる」 sentences. In the 3-user sample every
sentence maps to exactly one score, so:

1. **Lexicon lookup** (`care_score/lexicon.json`, 112 sentences). 94 entries are derived
   automatically from the sample ground truth; 18 sentences that only appear in the 7-user
   test set are mapped by hand in `build_lexicon.py` (`MANUAL_ENTRIES`, each with a rationale).
   Genuinely ambiguous ones carry a fractional expected score (e.g. 睡眠がやや浅い → 3.5).
2. **介助負担 / リスク fallback.** Days without an explicit 介助負担 sentence get a linear
   regression on the six domain scores, clipped to 0–3 (in the sample, 介助負担 ≥ 4 always
   comes with a sentence). Days without a リスク note are clipped to 0–2 for the same reason.
3. **注意点 constraints.** The 「注意点として、…」 sentence appears exactly on days with
   リスク ≥ 4 and lists the flags that fired; each flag is a hard threshold
   (食事低下 ⇔ 食事 ≤ 2, 入浴拒否・困難 ⇔ 入浴 ≤ 1, 活動低下 ⇔ 運動 ≤ 1, 睡眠不良 ⇔ 睡眠 ≤ 2,
   意欲低下・混乱 ⇔ 認知 ≤ 2, 介助負担大 ⇔ 介助負担 ≥ 4, 高リスク ⇔ リスク ≥ 4). These thresholds
   reproduce the sample's 注意すべき変化_正解 column 42/42 and are applied as upper/lower bounds.
4. **Aggregation.** Per-user mean of the 14 daily scores; 注意日数 = days with any flag.

On the 3 sample users the pipeline reproduces the 2-week means with MAE 0.010
(the six domain scores exactly; the residual comes from sentence-less 介助負担/リスク days).

## Layout

```
data/sample_3users/   sample input + ground truth (from the hackathon site)
data/test_7users/     event-day input (7 users, no ground truth)
care_score/           build_lexicon.py, lexicon.json, predict.py, evaluate.py
submission/           generated predictions
```
