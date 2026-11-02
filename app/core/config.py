from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Home Appliance Support Copilot"
    app_env: str = "development"
    groq_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "appliance-support-kb"
    pinecone_namespace: str = "appliance-kb"
    embedding_model: str = "multilingual-e5-large"
    groq_model: str = "openai/gpt-oss-120b"       # writes the answers
    groq_fast_model: str = "openai/gpt-oss-20b"   # routing, grading, rewriting, small talk
    top_k: int = 4
    rerank_model: str = "bge-reranker-v2-m3"  # empty string = plain vector search
    rerank_candidates: int = 20
    max_retries: int = 1
    admin_api_key: str = "change-me"
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    kb_dir: str = str(BASE_DIR / "data" / "appliance_kb")

    # Public deployment: uploads off, usage capped so visitors can't drain the free API quotas
    public_demo: bool = False
    demo_questions_per_ip_per_hour: int = 10
    demo_questions_per_day: int = 60   # ~2.5K gpt-oss-120b tokens per answer vs a 200K/day free quota
    demo_max_question_chars: int = 500

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")




@lru_cache
def get_settings() -> Settings:
    return Settings()
