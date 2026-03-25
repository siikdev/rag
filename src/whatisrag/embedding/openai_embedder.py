"""
Stage 0용 OpenAI Embedder — dense vector만 생성.
Stage 1에서 BGE-M3 (dense + sparse)로 교체 예정.
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from whatisrag.core.config import settings
from whatisrag.core.interfaces import BaseEmbedder
from whatisrag.core.schema import Document


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small/large 기반 dense embedder."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        batch_size: int = 128,
    ) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key or settings.openai_api_key,
        )
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[Document]:
        logger.info(f"Embedding {len(texts)} documents (OpenAI)")
        vectors = self._embeddings.embed_documents(texts)
        return [
            Document(content=text, dense_vector=vector)
            for text, vector in zip(texts, vectors)
        ]

    def embed_query(self, text: str) -> Document:
        vector = self._embeddings.embed_query(text)
        return Document(content=text, dense_vector=vector)
