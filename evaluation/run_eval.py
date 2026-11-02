"""Evaluate the copilot on evaluation/dataset.jsonl.

Metrics
- Retrieval (kb questions): is the gold manual page (±1 page) in the top 1 / 4 / 10 results? MRR@10.
- Path accuracy: did the agent answer from the manuals, the web, or directly, as expected?
- Answer correctness (kb questions): LLM judge compares the answer with the reference answer.
- Faithfulness (kb questions): LLM judge checks every claim is supported by the retrieved text.

Run from the project root:
    python -m evaluation.run_eval --tag baseline
    python -m evaluation.run_eval --tag quick --limit 5
    python -m evaluation.run_eval --resume evaluation/results/<file>.jsonl   # finish a run stopped by rate limits
    python -m evaluation.run_eval --retrieval-only --tag k6                  # search quality only, no LLM tokens
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import PureWindowsPath, Path
from typing import Literal
from groq import RateLimitError
from langchain_groq import ChatGroq
from pydantic import BaseModel
from app.core.config import get_settings
from app.rag.vectorstore import retrieve
from app.rag.workflow import ask
from evaluation.facts import supports

settings = get_settings()
HERE = Path(__file__).parent
JUDGE_MODEL = "qwen/qwen3.8-27b"  # different model from the copilot, so it doesn't grade its own answers
EXPECTED_SOURCE = {"kb": "private_kb", "web": "web_search", "direct": "direct"}
RETRIEVAL_K = 10


class Verdict(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    faithful: bool
    reason: str


def retrieval_ranks(question: dict) -> dict:
    """Where the evidence lands in the app's own retrieval (vector search + reranker), as 1-based ranks.

    gold_rank: first chunk from the gold manual page (±1). content_rank: first chunk that is the gold page
    or contains at least half of the key facts (any manual). file_in_top_k: gold manual among the top_k.
    """
    results = retrieve(question["question"], k=RETRIEVAL_K)
    out = {"gold_rank": None, "content_rank": None, "file_in_top_k": False}
    for i, d in enumerate(results, start=1):
        same_file = PureWindowsPath(d.metadata.get("source", "")).name == question["gold_file"]
        gold_page = same_file and abs(int(d.metadata.get("page", -99)) - question["gold_page"]) <= 1
        if same_file and i <= settings.top_k:
            out["file_in_top_k"] = True
        if gold_page and out["gold_rank"] is None:
            out["gold_rank"] = i
        if (gold_page or supports(d.page_content, question["key_facts"])) and out["content_rank"] is None:
            out["content_rank"] = i
    return out


def judge(judge_llm, question: dict, result: dict) -> Verdict:
    if result["source_used"] == "private_kb":
        context = "\n\n".join(d.page_content for d in result.get("kb_docs", []))
    else:
        context = result.get("web_results", "")
    return judge_llm.invoke(f"""
You are grading a home-appliance support chatbot.
Question: {question['question']}
Reference answer: {question['reference_answer']}
Key facts a correct answer must contain: {question['key_facts']}

Retrieved context the chatbot was given:
{context[:6000]}

Chatbot answer:
{result['answer'][:3000]}

Grade two things:
- correctness: "correct" if the answer contains the key facts and nothing that contradicts the reference;
  "partial" if it has some key facts or is vague; "incorrect" if it is wrong, missing the point, or says it cannot answer.
- faithful: true if every factual claim in the answer is supported by the retrieved context; false if it adds unsupported claims.
Return JSON: {{"correctness": "correct|partial|incorrect", "faithful": true, "reason": "one sentence"}}
""")


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def retrieval_metrics(kb: list[dict]) -> dict:
    k = settings.top_k
    metrics = {}
    for name, field in (("strict", "gold_rank"), ("content", "content_rank")):
        ranks = [r.get(field) for r in kb]
        if all(x is None for x in ranks) and field == "content_rank":
            continue  # older runs did not record content ranks
        metrics.update({
            f"{name}_hit@1_%": pct(sum(1 for x in ranks if x and x <= 1), len(kb)),
            f"{name}_hit@{k}_%": pct(sum(1 for x in ranks if x and x <= k), len(kb)),
            f"{name}_hit@{RETRIEVAL_K}_%": pct(sum(1 for x in ranks if x), len(kb)),
            f"{name}_mrr@{RETRIEVAL_K}": round(sum(1 / x for x in ranks if x) / len(kb), 3) if kb else None,
        })
    metrics[f"correct_manual_in_top{k}_%"] = pct(sum(r["file_in_top_k"] for r in kb), len(kb))
    metrics[f"content_misses@{k}"] = [r["id"] for r in kb if r.get("content_rank") is None or r["content_rank"] > k]
    return metrics


def summarize(rows: list[dict]) -> dict:
    kb = [r for r in rows if r["expected_path"] == "kb"]
    judged = [r for r in kb if r.get("correctness")]
    return {
        "questions": len(rows),
        "path_accuracy_%": pct(sum(r["path_ok"] for r in rows), len(rows)),
        "path_accuracy_by_type_%": {
            t: pct(sum(r["path_ok"] for r in rows if r["expected_path"] == t), sum(r["expected_path"] == t for r in rows))
            for t in ("kb", "web", "direct")
        },
        **retrieval_metrics(kb),
        "answer_correct_%": pct(sum(r["correctness"] == "correct" for r in judged), len(judged)),
        "answer_correct_or_partial_%": pct(sum(r["correctness"] in ("correct", "partial") for r in judged), len(judged)),
        "faithfulness_%": pct(sum(r["faithful"] for r in judged), len(judged)),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 1),
    }


def retrieval_only(questions: list[dict], tag: str):
    """Score search quality with Pinecone alone: no Groq tokens used."""
    rows = []
    for q in questions:
        if q["expected_path"] != "kb":
            continue
        rows.append({"id": q["id"], **retrieval_ranks(q)})
        print(f"{q['id']} gold_rank={rows[-1]['gold_rank']} content_rank={rows[-1]['content_rank']}")
    summary = {"tag": tag, "mode": "retrieval-only", "embedding_model": settings.embedding_model,
               "rerank_model": settings.rerank_model or None, "top_k": settings.top_k,
               "questions": len(rows), **retrieval_metrics(rows)}
    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M}_{tag}_retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", help="results .jsonl of a stopped run; finishes the questions it is missing")
    parser.add_argument("--retrieval-only", action="store_true", help="score retrieval only (no LLM calls)")
    args = parser.parse_args()

    questions = [json.loads(l) for l in open(HERE / "dataset.jsonl", encoding="utf-8") if l.strip()]
    if args.limit:
        questions = questions[: args.limit]
    if args.retrieval_only:
        return retrieval_only(questions, args.tag)
    judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0, api_key=settings.groq_api_key, max_retries=6,
                         reasoning_format="hidden").with_structured_output(Verdict, method="json_mode")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    if args.resume:
        rows_path = Path(args.resume)
        stamp, args.tag = rows_path.stem.split("_", 1)
        rows = [json.loads(l) for l in open(rows_path, encoding="utf-8") if l.strip()]
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        rows_path = out_dir / f"{stamp}_{args.tag}.jsonl"
        rows = []
    done = {r["id"] for r in rows}
    with open(rows_path, "a", encoding="utf-8") as fh:
        for i, q in enumerate(questions, start=1):
            if q["id"] in done:
                continue
            row = {"id": q["id"], "question": q["question"], "expected_path": q["expected_path"]}
            if q["expected_path"] == "kb":
                row.update(retrieval_ranks(q))
            start = time.time()
            try:
                result = ask(q["question"])
            except RateLimitError as exc:
                print(f"Stopped at {q['id']}: Groq rate limit ({exc.body.get('error', {}).get('message', '')[:160]})")
                print(f"Finish later with: python -m evaluation.run_eval --resume {rows_path}")
                break
            row["latency_s"] = round(time.time() - start, 1)
            row["source_used"] = result["source_used"]
            row["path_ok"] = result["source_used"] == EXPECTED_SOURCE[q["expected_path"]]
            row["trace"] = result.get("trace", [])
            row["answer"] = result["answer"]
            if q["expected_path"] == "kb":
                v = judge(judge_llm, q, result)
                row.update(correctness=v.correctness, faithful=v.faithful, judge_reason=v.reason)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(questions)}] {q['id']} path={row['source_used']}{'' if row['path_ok'] else ' (MISS)'}"
                  f" rank={row.get('gold_rank')} {row.get('correctness', '')} {row['latency_s']}s")

    summary = {"tag": args.tag, "timestamp": stamp, "complete": len(rows) == len(questions),
               "copilot_model": settings.groq_model, "decision_model": settings.groq_fast_model, "judge_model": JUDGE_MODEL,
               "embedding_model": settings.embedding_model, "rerank_model": settings.rerank_model or None,
               "top_k": settings.top_k, **summarize(rows)}
    (out_dir / f"{stamp}_{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
