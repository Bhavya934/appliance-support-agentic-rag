"""Test the agent's two decisions (route, grade evidence) with a given model, using the app's own prompts.

Router: kb and web questions should route to "kb"; conversational ones to "direct".
Grader: on the top-k reranked chunks, kb questions should be graded "good" and web questions "weak".
Retrieved chunks are cached so different models grade identical evidence.

Run from the project root:
    python -m evaluation.decision_test --model openai/gpt-oss-20b
    python -m evaluation.decision_test --model openai/gpt-oss-120b --step grader
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from app.core.config import get_settings
from app.rag.state import EvidenceGrade, RouteDecision
from app.rag.vectorstore import retrieve
from app.rag.workflow import grade_kb_prompt, route_prompt
from evaluation.facts import supports

settings = get_settings()
HERE = Path(__file__).parent
CHUNK_CACHE = HERE / "results" / "retrieved_chunks.json"
EXPECTED_ROUTE = {"kb": "kb", "web": "kb", "direct": "direct"}
EXPECTED_GRADE = {"kb": "good", "web": "weak"}


def cached_chunks(questions: list[dict]) -> dict[str, list[Document]]:
    cache = json.loads(CHUNK_CACHE.read_text(encoding="utf-8")) if CHUNK_CACHE.exists() else {}
    for q in questions:
        if q["id"] not in cache:
            cache[q["id"]] = [{"page_content": d.page_content, "metadata": d.metadata} for d in retrieve(q["question"])]
    CHUNK_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {qid: [Document(**d) for d in docs] for qid, docs in cache.items()}


def call(chain, prompt: str):
    out = chain.invoke(prompt)
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    return out["parsed"], usage.get("total_tokens", 0)


def evidence_present(q: dict, docs: list[Document]) -> bool:
    """Did retrieval actually bring back the answer? (gold page ±1, or a chunk with the key facts)"""
    for d in docs:
        same_page = (PureWindowsPath(d.metadata.get("source", "")).name == q["gold_file"]
                     and abs(int(d.metadata.get("page", -99)) - q["gold_page"]) <= 1)
        if same_page or supports(d.page_content, q["key_facts"]):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--step", choices=["router", "grader", "both"], default="both")
    args = parser.parse_args()

    questions = [json.loads(l) for l in open(HERE / "dataset.jsonl", encoding="utf-8") if l.strip()]
    llm = ChatGroq(model=args.model, temperature=0, api_key=settings.groq_api_key, max_retries=6)
    rows, summary = [], {"model": args.model, "timestamp": f"{datetime.now():%Y%m%d-%H%M}"}

    if args.step in ("router", "both"):
        router = llm.with_structured_output(RouteDecision, method="json_mode", include_raw=True)
        ok, tokens = 0, []
        for q in questions:
            decision, used = call(router, route_prompt(q["question"]))
            correct = decision.route == EXPECTED_ROUTE[q["expected_path"]]
            ok += correct
            tokens.append(used)
            rows.append({"step": "router", "id": q["id"], "decision": decision.route, "correct": correct, "tokens": used})
            print(f"router {q['id']}: {decision.route}{'' if correct else '  <-- WRONG'} ({used} tok)")
        summary["router_accuracy_%"] = round(100 * ok / len(questions), 1)
        summary["router_errors"] = [r["id"] for r in rows if r["step"] == "router" and not r["correct"]]
        summary["router_avg_tokens"] = round(sum(tokens) / len(tokens))

    if args.step in ("grader", "both"):
        graded = [q for q in questions if q["expected_path"] in EXPECTED_GRADE]
        chunks = cached_chunks(graded)
        grader = llm.with_structured_output(EvidenceGrade, method="json_mode", include_raw=True)
        tokens = []
        for q in graded:
            docs = chunks[q["id"]]
            grade, used = call(grader, grade_kb_prompt(q["question"], docs))
            row = {"step": "grader", "id": q["id"], "expected": EXPECTED_GRADE[q["expected_path"]], "decision": grade.grade,
                   "correct": grade.grade == EXPECTED_GRADE[q["expected_path"]], "tokens": used}
            if q["expected_path"] == "kb":
                row["evidence_present"] = evidence_present(q, docs)
            rows.append(row)
            tokens.append(used)
            print(f"grader {q['id']}: {grade.grade}{'' if row['correct'] else '  <-- WRONG'} ({used} tok)")
            time.sleep(1)
        g = [r for r in rows if r["step"] == "grader"]
        kb_rows = [r for r in g if r["expected"] == "good"]
        kb_with_evidence = [r for r in kb_rows if r["evidence_present"]]
        web_rows = [r for r in g if r["expected"] == "weak"]
        pct = lambda rs: round(100 * sum(r["correct"] for r in rs) / len(rs), 1) if rs else None
        summary.update({
            "grader_accuracy_%": pct(g),
            "grader_good_on_kb_%": pct(kb_rows),
            "grader_good_on_kb_when_evidence_retrieved_%": pct(kb_with_evidence),
            "grader_weak_on_web_%": pct(web_rows),
            "grader_errors": [r["id"] for r in g if not r["correct"]],
            "grader_avg_tokens": round(sum(tokens) / len(tokens)),
        })

    tag = args.model.split("/")[-1]
    out = HERE / "results" / f"{summary['timestamp']}_decisions_{tag}"
    out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out.with_name(out.name + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
