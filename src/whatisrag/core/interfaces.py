"""
시스템 전체 컴포넌트의 추상 인터페이스(계약).

핵심 원칙:
- 모든 구체 구현체는 이 인터페이스를 상속
- Pipeline은 인터페이스에만 의존 → 런타임에 구현체 교체 가능 (전략 패턴)
- 이 파일은 Stage가 바뀌어도 수정하지 않는 것이 목표
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from whatisrag.core.schema import (
    Document,
    GenerationResult,
    RankedResult,
    RetrievalResult,
)


class BaseEmbedder(ABC):
    """Dense / Sparse 임베딩 생성 인터페이스."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Document]:
        """텍스트 리스트 → dense_vector, sparse_vector가 채워진 Document 리스트."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> Document:
        """쿼리 텍스트 → 임베딩된 Document (content=text)."""
        ...


class BaseRetriever(ABC):
    """모든 Retriever의 공통 인터페이스."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline 비교 / 로깅에 사용할 식별자."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """쿼리 → 관련 문서 + 점수 리스트 반환."""
        ...


class BaseQueryTransformer(ABC):
    """HyDE, RAG-Fusion 등 쿼리 변환 인터페이스."""

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        """원본 쿼리 → 확장된 쿼리 리스트.
        변환 없이 그대로 쓸 경우 [query] 반환.
        """
        ...


class BaseReranker(ABC):
    """Cross-encoder, Cohere 등 Reranker 인터페이스."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RankedResult]:
        """Retriever 결과를 받아 재정렬 후 top_k 반환."""
        ...


class BasePipeline(ABC):
    """Stage별 RAG 파이프라인 인터페이스."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, query: str) -> GenerationResult:
        """쿼리 → 최종 답변 + 소스 문서 + 중간 단계 반환."""
        ...
