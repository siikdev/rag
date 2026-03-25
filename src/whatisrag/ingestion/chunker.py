"""
청킹 전략 모음.

Stage 0: RecursiveCharacterTextSplitter (Fixed)
Stage 1+: SemanticChunker, ParentChildChunker 추가 예정

모든 전략은 list[Document] → list[Document] 시그니처를 따름.
"""
from __future__ import annotations

from enum import Enum

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from whatisrag.core.schema import Document


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


def chunk_fixed(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """RecursiveCharacterTextSplitter — 기본 전략."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    result: list[Document] = []
    for doc in documents:
        splits = splitter.split_text(doc.content)
        for i, chunk in enumerate(splits):
            result.append(
                Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_strategy": ChunkStrategy.FIXED,
                    },
                )
            )
    logger.info(f"Fixed chunking: {len(documents)} docs → {len(result)} chunks")
    return result


def chunk_semantic(
    documents: list[Document],
    openai_api_key: str = "",
) -> list[Document]:
    """SemanticChunker — 문장 임베딩 유사도 기반 분할 (OpenAI 임베딩 사용).
    Stage 1 이후 Hybrid Search와 함께 사용 권장.
    """
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    splitter = SemanticChunker(embeddings)
    result: list[Document] = []
    for doc in documents:
        splits = splitter.split_text(doc.content)
        for i, chunk in enumerate(splits):
            result.append(
                Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_strategy": ChunkStrategy.SEMANTIC,
                    },
                )
            )
    logger.info(f"Semantic chunking: {len(documents)} docs → {len(result)} chunks")
    return result


def chunk(
    documents: list[Document],
    strategy: ChunkStrategy = ChunkStrategy.FIXED,
    **kwargs,
) -> list[Document]:
    """전략에 따라 청킹 함수 디스패치."""
    if strategy == ChunkStrategy.FIXED:
        return chunk_fixed(documents, **kwargs)
    if strategy == ChunkStrategy.SEMANTIC:
        return chunk_semantic(documents, **kwargs)
    raise ValueError(f"Unknown strategy: {strategy}")
