"""
시스템 전체에서 사용하는 공통 데이터 모델.
한 번 정의 후 수정을 최소화하는 것이 원칙 — 모든 컴포넌트가 이 스키마에 의존.
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """단일 문서 청크를 나타내는 핵심 단위."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Embedding vectors — ingestion 시 채워짐
    dense_vector: Optional[list[float]] = None
    # sparse: {token_id(int): weight(float)} — BGE-M3 lexical_weights 포맷
    sparse_vector: Optional[dict[int, float]] = None


class RetrievalResult(BaseModel):
    """Retriever가 반환하는 단일 결과."""

    document: Document
    score: float
    retriever_name: str


class RankedResult(BaseModel):
    """Reranker가 재정렬한 결과."""

    document: Document
    rerank_score: float
    original_score: float
    retriever_name: str


class GenerationResult(BaseModel):
    """Pipeline.run()의 최종 출력."""

    answer: str
    source_documents: list[Document]
    # 각 단계별 중간 결과 — 디버깅 및 RAGAS 평가에 활용
    intermediate_steps: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """RAGAS 평가 결과."""

    pipeline_name: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
