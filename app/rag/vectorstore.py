import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()

# Output sizes of the embedding models hosted by Pinecone's inference API
EMBEDDING_DIMENSIONS = {
    "multilingual-e5-large": 1024,
    "llama-text-embed-v2": 1024,
}

_client = None
_embeddings = None
_vectorstore = None


def pinecone_client() -> Pinecone:
    global _client
    if _client is None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is missing")
        _client = Pinecone(api_key=settings.pinecone_api_key)
    return _client


def get_embedding_dimension() -> int:
    model = settings.embedding_model.strip().lower()
    if model not in EMBEDDING_DIMENSIONS:
        raise ValueError(f"Unknown embedding model '{settings.embedding_model}'. Add its dimension to EMBEDDING_DIMENSIONS.")
    return EMBEDDING_DIMENSIONS[model]


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Pinecone's hosted inference: free on the Starter plan, no extra key
        _embeddings = PineconeEmbeddings(model=settings.embedding_model, pinecone_api_key=settings.pinecone_api_key)
    return _embeddings


def ensure_index():
    """Return the index, creating it on first use. Refuses to touch an existing index of the wrong size."""
    pc = pinecone_client()
    dimension = get_embedding_dimension()
    name = settings.pinecone_index_name

    if name in [x["name"] for x in pc.list_indexes()]:
        existing = pc.describe_index(name).dimension
        if existing != dimension:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {existing} but '{settings.embedding_model}' produces {dimension}. "
                "Use a new PINECONE_INDEX_NAME and re-run ingest_kb.py."
            )
    else:
        pc.create_index(name=name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)

    return pc.Index(name)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(index=ensure_index(), embedding=get_embeddings(), namespace=settings.pinecone_namespace)
    return _vectorstore


def retrieve(query: str, k: int | None = None):
    """Vector search for rerank_candidates chunks, then Pinecone's reranker keeps the best k."""
    k = k or settings.top_k
    if not settings.rerank_model:
        return get_vectorstore().similarity_search(query, k=k)
    candidates = get_vectorstore().similarity_search(query, k=max(settings.rerank_candidates, k))
    if not candidates:
        return []
    ranked = pinecone_client().inference.rerank(
        model=settings.rerank_model,
        query=query,
        documents=[d.page_content for d in candidates],
        top_n=k,
        return_documents=False,
    )
    return [candidates[r.index] for r in ranked.data]


def add_documents(chunks):
    return get_vectorstore().add_documents(chunks)
