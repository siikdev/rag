"""
샘플 데이터 수집 스크립트 — Stage 0.

사용법:
  poetry run python scripts/ingest_sample_data.py --path data/raw/sample.pdf

기본 동작:
  data/raw/ 디렉토리의 모든 PDF를 로드 → 청킹 → OpenAI 임베딩 → Qdrant 업서트
"""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from whatisrag.embedding.openai_embedder import OpenAIEmbedder
from whatisrag.ingestion.chunker import ChunkStrategy, chunk
from whatisrag.ingestion.indexer import init_collection_dense, upsert_documents
from whatisrag.ingestion.loader import load


def main(path: str, chunk_size: int = 512, recreate: bool = False) -> None:
    # 1. 로드
    documents = load(path)
    logger.info(f"Loaded {len(documents)} documents")

    # 2. 청킹
    chunks = chunk(documents, strategy=ChunkStrategy.FIXED, chunk_size=chunk_size)
    logger.info(f"Chunked into {len(chunks)} pieces")

    # 3. 임베딩
    embedder = OpenAIEmbedder()
    texts = [doc.content for doc in chunks]
    embedded_docs = embedder.embed_documents(texts)

    # metadata 병합
    for chunk_doc, embedded in zip(chunks, embedded_docs):
        chunk_doc.dense_vector = embedded.dense_vector

    # 4. 인덱싱
    init_collection_dense(recreate=recreate)
    upsert_documents(chunks)
    logger.success("Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/raw", help="File or directory path")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()
    main(args.path, args.chunk_size, args.recreate)
