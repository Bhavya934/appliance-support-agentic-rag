"""Compare retrieval setups on the kb questions of evaluation/dataset.jsonl (no Groq tokens used).

Setups: dense vector search, BM25 keyword search, hybrid (reciprocal rank fusion of both),
and each followed by Pinecone's hosted reranker.

Two hit definitions are reported:
- strict:  a retrieved chunk comes from the gold manual page (±1 page)
- content: a retrieved chunk (from any manual) contains at least half of the question's key facts

Run from the project root:  python -m evaluation.retrieval_experiments
"""
import json
from pathlib import Path, PureWindowsPath
from pinecone import Pinecone
from app.core.config import get_settings
from app.rag.hybrid import BM25Index, chunk_key, rrf
from app.rag.vectorstore import get_vectorstore
from app.services.ingestion import chunk_documents, load_file
from evaluation.facts import supports

settings = get_settings()
HERE = Path(__file__).parent
CANDIDATES = 20
RERANK_MODEL = "bge-reranker-v2-m3"
CACHE = HERE / "results" / "rerank_cache.json"


def main():
    questions = [json.loads(l) for l in open(HERE / "dataset.jsonl", encoding="utf-8") if l.strip()]
    kb = [q for q in questions if q["expected_path"] == "kb"]

    kb_dir = Path(settings.kb_dir)
    chunks = chunk_documents(d for p in sorted(kb_dir.glob("*.pdf")) for d in load_file(p))
    bm25 = BM25Index(chunks)
    by_key = {chunk_key(c): c for c in chunks}
    store = get_vectorstore()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    def rerank(question: str, keys: list, tag: str) -> list:
        cache_id = f"{tag}|{question}"
        if cache_id not in cache:
            docs = [by_key[k].page_content for k in keys]
            res = pc.inference.rerank(model=RERANK_MODEL, query=question, documents=docs, top_n=len(docs), return_documents=False)
            cache[cache_id] = [list(keys[r.index]) for r in res.data]
        return [tuple(k) for k in cache[cache_id]]

    def strict_hit(q, key) -> bool:
        return PureWindowsPath(key[0]).name == q["gold_file"] and abs(key[1] - q["gold_page"]) <= 1

    results = {}
    for q in kb:
        dense = [chunk_key(d) for d in store.similarity_search(q["question"], k=CANDIDATES)]
        dense = [k for k in dense if k in by_key]
        sparse = bm25.search(q["question"], CANDIDATES)
        hybrid = rrf([dense, sparse])[:CANDIDATES]
        rankings = {
            "dense": dense,
            "bm25": sparse,
            "hybrid": hybrid,
            "dense+rerank": rerank(q["question"], dense, "dense"),
            "hybrid+rerank": rerank(q["question"], hybrid, "hybrid"),
        }
        for name, keys in rankings.items():
            strict = next((i for i, k in enumerate(keys, 1) if strict_hit(q, k)), None)
            content = next((i for i, k in enumerate(keys, 1) if strict_hit(q, k) or supports(by_key[k].page_content, q["key_facts"])), None)
            results.setdefault(name, []).append((q["id"], strict, content))
        print(q["id"], {n: r[-1][1:] for n, r in results.items()})

    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    n = len(kb)
    pct = lambda hits: round(100 * hits / n, 1)
    table = []
    for name, rows in results.items():
        for k in (4, 6):
            table.append({
                "setup": f"{name}@{k}",
                "strict_hit_%": pct(sum(1 for _, s, _ in rows if s and s <= k)),
                "content_hit_%": pct(sum(1 for _, _, c in rows if c and c <= k)),
                "strict_hit@1_%": pct(sum(1 for _, s, _ in rows if s == 1)),
                "content_mrr": round(sum(1 / c for _, _, c in rows if c and c <= 10) / n, 3),
                "content_misses": [qid for qid, _, c in rows if not c or c > k],
            })
    print(f"\n{'setup':20} {'strict':>7} {'content':>8} {'strict@1':>9} {'MRR':>6}  content misses")
    for t in table:
        print(f"{t['setup']:20} {t['strict_hit_%']:>7} {t['content_hit_%']:>8} {t['strict_hit@1_%']:>9} {t['content_mrr']:>6}  {t['content_misses']}")
    (HERE / "results" / "retrieval_experiments.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
