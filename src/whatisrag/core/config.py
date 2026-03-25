"""
pydantic-settings 기반 설정.
.env 파일 또는 환경변수로 오버라이드 가능.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ── Embedding ─────────────────────────────────────
    embedding_provider: str = "bge_m3"   # "bge_m3" | "openai"
    embedding_model: str = "BAAI/bge-m3"

    # ── Qdrant ────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "whatisrag"

    # ── Reranker (Stage 2) ────────────────────────────
    reranker_provider: str = "cross_encoder"  # "cross_encoder" | "cohere"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_api_key: str = ""

    # ── Neo4j (Stage 5) ───────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── API ───────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
