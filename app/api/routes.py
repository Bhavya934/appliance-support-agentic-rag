import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Request
from groq import RateLimitError
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit
from app.services.rate_limit import SlidingWindowLimiter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
settings = get_settings()

per_ip_limiter = SlidingWindowLimiter(settings.demo_questions_per_ip_per_hour, 3600)
daily_limiter = SlidingWindowLimiter(settings.demo_questions_per_day, 86400)

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def client_ip(request: Request) -> str:
    # Behind the hosting proxy the visitor's address is the first X-Forwarded-For entry
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def enforce_demo_limits(request: Request, question: str):
    if len(question) > settings.demo_max_question_chars:
        raise HTTPException(status_code=413, detail=f"Please keep questions under {settings.demo_max_question_chars} characters in the demo.")
    wait = per_ip_limiter.check(client_ip(request))
    if wait is not None:
        raise HTTPException(status_code=429, detail=f"Demo limit: {settings.demo_questions_per_ip_per_hour} questions per hour. Please try again in {max(1, wait // 60)} min.")
    if daily_limiter.check("all") is not None:
        raise HTTPException(status_code=429, detail="The demo has reached today's question limit. Please try again tomorrow.")


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


def key_shape(value: str, prefix: str) -> dict:
    """Describe an API key without revealing it: catches blank keys and copy-paste mistakes."""
    return {"set": bool(value), "expected_prefix": value.strip().startswith(prefix),
            "extra_whitespace": value != value.strip(), "length": len(value)}


@router.get("/health/dependencies")
def dependencies():
    """Checks each outside service the agent needs. Never returns key values."""
    from groq import Groq
    from app.rag.vectorstore import ensure_index, get_embeddings

    def run(check):
        try:
            return {"ok": True, **(check() or {})}
        except Exception as exc:
            message = str(exc)
            for secret in (settings.groq_api_key, settings.pinecone_api_key, settings.tavily_api_key):
                if secret:
                    message = message.replace(secret, "***")
            return {"ok": False, "error": f"{type(exc).__name__}: {message[:300]}"}

    def pinecone_check():
        stats = ensure_index().describe_index_stats()
        vectors = stats.namespaces.get(settings.pinecone_namespace)
        get_embeddings().embed_query("test")
        return {"index": settings.pinecone_index_name, "namespace_vectors": vectors.vector_count if vectors else 0}

    return {
        "groq": {"key": key_shape(settings.groq_api_key, "gsk_"),
                 **run(lambda: {"models": len(Groq(api_key=settings.groq_api_key).models.list().data)})},
        "pinecone": {"key": key_shape(settings.pinecone_api_key, "pcsk_"), **run(pinecone_check)},
        "tavily": {"key": key_shape(settings.tavily_api_key, "tvly-")},
    }


@router.post("/chat")
def chat(payload: ChatRequest, request: Request):
    if settings.public_demo:
        enforce_demo_limits(request, payload.question)
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except RateLimitError as exc:
        logger.warning("Groq rate limit: %s", exc)
        raise HTTPException(status_code=429, detail="The free AI quota is used up for the moment. Please try again in a few minutes.") from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        detail = "Something went wrong while answering. Please try again." if settings.public_demo else str(exc)
        raise HTTPException(status_code=500, detail=detail) from exc



@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if settings.public_demo:
        raise HTTPException(status_code=403, detail="Uploading documents is disabled in the public demo.")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
