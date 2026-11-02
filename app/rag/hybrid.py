"""Keyword (BM25) search over the manual chunks, fused with vector search."""
import re
from pathlib import PureWindowsPath
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

STOPWORDS = set("""a an the and or of to in on for with is are be can i my me how what do does should
you your it this that if at by from as not no yes when which why will""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def chunk_key(doc: Document) -> tuple:
    """Identifies a chunk the same way in Pinecone results and in the local corpus."""
    m = doc.metadata
    return PureWindowsPath(m.get("source", "")).name, int(m.get("page", -1)), int(m.get("start_index", -1))


class BM25Index:
    def __init__(self, chunks: list[Document]):
        self.keys = [chunk_key(c) for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple]:
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.keys[i] for i in best]


def rrf(rankings: list[list], k: int = 60) -> list:
    """Reciprocal rank fusion: items ranked high in any list rise to the top."""
    scores: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
