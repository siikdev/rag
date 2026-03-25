"""
Stage 0 — Naive RAG Pipeline (Dense-only).

베이스라인으로 사용. Stage 1 이후 모든 Pipeline의 성능 비교 기준점.

흐름:
  query → Qdrant dense search → LLM 생성
"""
from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient, models

from whatisrag.core.config import settings
from whatisrag.core.schema import Document, GenerationResult, RetrievalResult
from whatisrag.pipeline.base_pipeline import RAGBasePipeline


class NaivePipeline(RAGBasePipeline):
    """Dense vector 검색만 사용하는 베이스라인 파이프라인."""

    def __init__(
        self,
        embedder,  # BaseEmbedder — 순환 임포트 방지로 타입 힌트 생략
        collection_name: str = settings.qdrant_collection,
        qdrant_url: str = settings.qdrant_url,
        top_k: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._embedder = embedder
        self._collection = collection_name
        self._client = QdrantClient(url=qdrant_url)
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "NaivePipeline"

    def _retrieve(self, query: str) -> list[RetrievalResult]:
        query_doc = self._embedder.embed_query(query)
        if query_doc.dense_vector is None:
            raise ValueError("Embedder did not return a dense vector.")

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_doc.dense_vector,
            limit=self._top_k,
        )
        return [
            RetrievalResult(
                document=Document(
                    content=hit.payload.get("content", ""),
                    metadata={k: v for k, v in hit.payload.items() if k != "content"},
                ),
                score=hit.score,
                retriever_name=self.name,
            )
            for hit in hits
        ]

    def run(self, query: str) -> GenerationResult:
        logger.debug(f"[{self.name}] query={query!r}")

        retrieval_results = self._retrieve(query)
        source_docs = [r.document for r in retrieval_results]

        answer = self._generate(query, source_docs)

        return GenerationResult(
            answer=answer,
            source_documents=source_docs,
            intermediate_steps={
                "retrieval_results": [
                    {"score": r.score, "content": r.document.content[:100]}
                    for r in retrieval_results
                ]
            },
        )
