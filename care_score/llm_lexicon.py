"""Fill lexicon gaps with a local LLM (Ollama) instead of the hand-written MANUAL_ENTRIES.

The LLM is an *offline* step: it scores each unknown sentence once, the answers are
written into lexicon.json, and predict.py / the notebook never call a model.

    # on a Mac with Ollama running (brew install ollama; ollama pull qwen2.5:14b)
    python -m care_score.llm_lexicon --models qwen2.5:14b --validate      # how well does it recover the sample scale?
    python -m care_score.llm_lexicon --models qwen2.5:14b --write         # score the test-only sentences, update lexicon.json

For every unknown sentence the model receives
  * the domain (from its slot position; trailing sentences may be 介助負担 or リスク),
  * the complete tier ladder for that domain taken from the 3-user sample,
  * the 利用者特徴 profile(s) of the user(s) whose records contain the sentence,
  * hard bounds implied by the 注意点 flags on those days,
and must answer JSON {"domain", "score_low", "score_high", "reason"}.  Several votes
(different seeds / models) are averaged; the midpoint of low..high is used, which is
the MAE-optimal choice when two tiers are equally plausible.  Answers are cached in
care_score/llm_cache.json so re-runs are free.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from .build_lexicon import (DOMAINS, FLAG_RE, GT_RECORDS, MANUAL_ENTRIES, OUT as LEXICON_PATH,
                            build as build_sample_lexicon, split_sentences)

ROOT = Path(__file__).resolve().parent.parent
TEST_RECORDS = ROOT / "data/test_7users/care_hackathon_student_records_7users_14days.csv"
TEST_SUMMARIES = ROOT / "data/test_7users/care_hackathon_2week_summaries_7users.csv"
CACHE_PATH = Path(__file__).resolve().parent / "llm_cache.json"

ALL_DOMAINS = DOMAINS + ["介助負担", "リスク"]
DOMAIN_JA = {"食事": "食事（摂取量）", "入浴": "入浴（自立度）", "運動": "運動・歩行（活動量）", "排泄": "排泄（自立度）",
             "睡眠": "睡眠状態", "認知_意欲": "認知・意欲", "介助負担": "介助負担（0=なし … 5=非常に大きい）",
             "リスク": "リスク（0=なし … 5=非常に高い）"}
# flag -> (domain, comparator, threshold)  — identical to predict.FLAGS
FLAGS = {"食事低下": ("食事", "<=", 2), "入浴拒否・困難": ("入浴", "<=", 1), "活動低下": ("運動", "<=", 1),
         "睡眠不良": ("睡眠", "<=", 2), "意欲低下・混乱": ("認知_意欲", "<=", 2),
         "介助負担大": ("介助負担", ">=", 4), "高リスク": ("リスク", ">=", 4)}


# --------------------------------------------------------------------------- clients
class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: int = 300):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def chat(self, system: str, user: str, seed: int, temperature: float) -> str:
        body = {"model": self.model, "stream": False, "format": "json",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "seed": seed, "num_predict": 400}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["message"]["content"]

    def __str__(self) -> str:
        return f"ollama:{self.model}"


class FakeClient:
    """Deterministic stand-in for tests: answers from a {sentence: (domain, lo, hi)} table."""

    def __init__(self, table: dict[str, tuple[str, float, float]], name: str = "fake"):
        self.table, self.name = table, name

    def chat(self, system: str, user: str, seed: int, temperature: float) -> str:
        m = re.search(r"【対象文】\n(.+)\n", user)
        dom, lo, hi = self.table[m.group(1)]
        return json.dumps({"domain": dom, "score_low": lo, "score_high": hi, "reason": "fake"}, ensure_ascii=False)

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------------------- context
def collect_unknown(records: list[dict], lexicon: dict, summaries: dict[str, dict]) -> dict[str, dict]:
    """Return {sentence: {slots, domains, profiles, bounds}} for sentences missing from the lexicon."""
    info: dict[str, dict] = {}
    for r in records:
        sents = split_sentences(r["記録内容"])
        flags = None
        for s in sents:
            m = FLAG_RE.match(s)
            if m:
                flags = {f.strip() for f in m.group(1).split("、")}
        for i, s in enumerate(sents):
            if i == 0 or s in lexicon or FLAG_RE.match(s):
                continue
            d = info.setdefault(s, {"slots": set(), "profiles": set(), "bounds": {}, "n": 0})
            d["n"] += 1
            d["slots"].add(i)
            prof = summaries.get(r["利用者ID"], {}).get("利用者特徴") or r.get("利用者特徴")
            if prof:
                d["profiles"].add(f"{r['利用者ID']}: {prof}")
            if flags is not None:
                # the 注意点 sentence lists every flag that fired that day -> bounds on every domain
                for name, (dom, op, thr) in FLAGS.items():
                    lo, hi = d["bounds"].get(dom, (0, 5))
                    if name in flags:
                        lo, hi = (lo, min(hi, thr)) if op == "<=" else (max(lo, thr), hi)
                    else:
                        lo, hi = (max(lo, thr + 1), hi) if op == "<=" else (lo, min(hi, thr - 1))
                    d["bounds"][dom] = (lo, hi)
    for s, d in info.items():
        d["domains"] = sorted({DOMAINS[i - 1] for i in d["slots"] if 1 <= i <= 6}) or ["介助負担", "リスク"]
        d["bounds"] = {k: v for k, v in d["bounds"].items() if v[0] <= v[1]}   # drop contradictory bounds
    return info


def ladder_text(lexicon: dict, domain: str, hide: str | None = None) -> str:
    tiers: dict[float, list[str]] = defaultdict(list)
    for s, v in lexicon.items():
        if v["domain"] == domain and v.get("source") == "sample_3users" and s != hide:
            tiers[v["score"]].append(s)
    lines = []
    for sc in sorted(tiers, reverse=True):
        lines.append(f"  {int(sc) if float(sc).is_integer() else sc}: " + " ／ ".join(tiers[sc]))
    return "\n".join(lines)


SYSTEM = (
    "あなたは介護記録の標準化を行う専門家です。介護記録の1文を読み、指定された項目のスコアを"
    "サンプルの尺度に照らして判定します。必ずJSONのみで回答してください。"
)


def make_prompt(sentence: str, ctx: dict, lexicon: dict, hide: str | None = None) -> str:
    parts = ["【課題】次の介護記録の1文に対して、該当する項目とスコアを判定してください。", "", "【対象文】", sentence, ""]
    parts.append("【候補となる項目】" + "、".join(DOMAIN_JA[d] for d in ctx["domains"]))
    parts.append("")
    parts.append("【尺度の例（3名分の正解データより。数字はスコア、右はそのスコアが付いた文）】")
    for d in ctx["domains"]:
        parts.append(f"■ {DOMAIN_JA[d]}")
        parts.append(ladder_text(lexicon, d, hide) or "  （例なし）")
    if ctx.get("profiles"):
        parts += ["", "【この文が現れた利用者の特徴】"] + [f"  - {p}" for p in sorted(ctx["profiles"])]
    bounds = {d: b for d, b in ctx.get("bounds", {}).items() if d in ctx["domains"] and b != (0, 5)}
    if bounds:
        parts += ["", "【同日の注意点フラグから確定している範囲】"] + [f"  - {DOMAIN_JA[d]}: {lo}〜{hi}" for d, (lo, hi) in bounds.items()]
    parts += [
        "",
        "【回答形式】以下のキーを持つJSONのみを返してください。",
        '{"domain": "<' + "|".join(ctx["domains"]) + '>", "score_low": <整数>, "score_high": <整数>, "reason": "<日本語で1文>"}',
        "score_low と score_high は同じ値でも構いません。2つの段階のどちらか判断がつかない場合のみ幅を持たせてください。",
        "尺度の例に同じ意味の文があれば、そのスコアに合わせてください。例にない段階（例：入浴の4）も選べます。",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- scoring
def parse_answer(text: str, allowed: list[str]) -> tuple[str, float, float, str]:
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0) if m else text)
    dom = str(obj.get("domain", "")).replace("・", "_").replace("認知意欲", "認知_意欲")
    if dom not in allowed:
        dom = allowed[0] if len(allowed) == 1 else next((a for a in allowed if a in dom), allowed[0])
    lo = float(obj.get("score_low", obj.get("score", 3)))
    hi = float(obj.get("score_high", lo))
    lo, hi = min(lo, hi), max(lo, hi)
    return dom, max(0, min(5, lo)), max(0, min(5, hi)), str(obj.get("reason", ""))


def score_sentence(sentence: str, ctx: dict, lexicon: dict, clients: list, votes: int, cache: dict,
                   hide: str | None = None) -> dict:
    prompt = make_prompt(sentence, ctx, lexicon, hide)
    answers = []
    for client in clients:
        for k in range(votes):
            temp = 0.0 if votes == 1 else 0.5
            key = f"{client}|{k}|{temp}|{hash(prompt) & 0xFFFFFFFF}|{sentence}"
            if key not in cache:
                raw = client.chat(SYSTEM, prompt, seed=k + 1, temperature=temp)
                cache[key] = raw
            try:
                answers.append(parse_answer(cache[key], ctx["domains"]))
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  unparseable answer from {client} for {sentence!r}: {e}", file=sys.stderr)
    if not answers:
        raise RuntimeError(f"no usable answers for {sentence!r}")
    domain = statistics.mode([a[0] for a in answers])
    mids = [(a[1] + a[2]) / 2 for a in answers if a[0] == domain]
    score = statistics.fmean(mids)
    lo, hi = ctx.get("bounds", {}).get(domain, (0, 5))
    score = max(lo, min(hi, score))
    return {"domain": domain, "score": round(score, 2), "n": 0, "source": "llm",
            "models": [str(c) for c in clients], "votes": [(a[0], a[1], a[2]) for a in answers],
            "why": answers[0][3]}


# --------------------------------------------------------------------------- modes
def validate(lexicon: dict, clients: list, votes: int, cache: dict, limit: int | None) -> None:
    """Hide each sample sentence from its own ladder and ask the model to score it."""
    gt = list(csv.DictReader(GT_RECORDS.open(encoding="utf-8")))
    profiles = {r["利用者ID"]: r["利用者特徴"] for r in gt}
    items = [(s, v) for s, v in lexicon.items() if v.get("source") == "sample_3users"]
    if limit:
        items = items[:limit]
    errs, per_dom = [], defaultdict(list)
    print(f"{'sentence':40s} {'dom':8s} truth  llm   |err|")
    for s, v in items:
        users = {r["利用者ID"] for r in gt if s in r["記録内容"]}
        ctx = {"domains": [v["domain"]] if v["domain"] in DOMAINS else ["介助負担", "リスク"],
               "profiles": {f"{u}: {profiles[u]}" for u in users}, "bounds": {}, "n": v["n"]}
        res = score_sentence(s, ctx, lexicon, clients, votes, cache, hide=s)
        err = abs(res["score"] - v["score"]) + (0 if res["domain"] == v["domain"] else 5)
        errs.append(err)
        per_dom[v["domain"]].append(err)
        print(f"{s:40s} {v['domain']:8s} {v['score']:>5} {res['score']:>5}  {err:.2f}")
    print(f"\nvalidation MAE over {len(errs)} sample sentences: {statistics.fmean(errs):.3f}")
    print("  per domain: " + "  ".join(f"{d}:{statistics.fmean(e):.2f}" for d, e in per_dom.items()))
    print("  (a domain misclassification counts as 5)")


def fill(lexicon: dict, records_path: Path, summaries_path: Path | None, clients: list, votes: int,
         cache: dict) -> dict[str, dict]:
    records = list(csv.DictReader(records_path.open(encoding="utf-8")))
    summaries = {r["利用者ID"]: r for r in csv.DictReader(summaries_path.open(encoding="utf-8"))} if summaries_path and summaries_path.exists() else {}
    unknown = collect_unknown(records, lexicon, summaries)
    print(f"{len(unknown)} sentences not in the sample lexicon\n")
    results = {}
    print(f"{'sentence':40s} {'llm':>14s} {'manual':>14s}  Δ")
    for s, ctx in unknown.items():
        res = score_sentence(s, ctx, lexicon, clients, votes, cache)
        results[s] = res
        man = MANUAL_ENTRIES.get(s)
        man_txt = f"{man[0]} {man[1]}" if man else "-"
        delta = f"{res['score'] - man[1]:+.2f}" if man and man[0] == res["domain"] else ("DOMAIN≠" if man else "")
        print(f"{s:40s} {res['domain'] + ' ' + str(res['score']):>14s} {man_txt:>14s}  {delta}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen2.5:14b", help="comma-separated Ollama model names")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--votes", type=int, default=3, help="samples per model (1 = greedy)")
    ap.add_argument("--records", type=Path, default=TEST_RECORDS)
    ap.add_argument("--summaries", type=Path, default=TEST_SUMMARIES)
    ap.add_argument("--validate", action="store_true", help="score the sample's own sentences with their entry hidden")
    ap.add_argument("--limit", type=int, help="with --validate: only the first N sentences")
    ap.add_argument("--write", action="store_true", help="write LLM answers for unknown sentences into lexicon.json")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    clients = [OllamaClient(m.strip(), args.host) for m in args.models.split(",") if m.strip()]
    cache = {} if args.no_cache or not CACHE_PATH.exists() else json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    sample_lex = {s: v for s, v in build_sample_lexicon().items() if v["source"] == "sample_3users"}

    try:
        if args.validate:
            validate(sample_lex, clients, args.votes, cache, args.limit)
        else:
            results = fill(sample_lex, args.records, args.summaries, clients, args.votes, cache)
            if args.write:
                lex = dict(sample_lex)
                for s, (dom, score, why) in MANUAL_ENTRIES.items():      # keep manual as fallback...
                    lex[s] = {"domain": dom, "score": score, "n": 0, "source": "manual", "why": why}
                for s, res in results.items():                           # ...LLM answers take precedence
                    res = dict(res)
                    if s in MANUAL_ENTRIES:
                        res["manual"] = list(MANUAL_ENTRIES[s][:2])
                    lex[s] = res
                LEXICON_PATH.write_text(json.dumps(lex, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"\nwrote {LEXICON_PATH} ({len(lex)} sentences, {len(results)} from LLM)")
            else:
                print("\n(dry run — add --write to update lexicon.json)")
    finally:
        if not args.no_cache:
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
