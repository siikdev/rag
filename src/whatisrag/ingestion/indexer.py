"""
Qdrant 인덱서 — 컬렉션 초기화 및 Document 업서트.

Stage 0: dense vector 전용 컬렉션
Stage 1: sparse vector 필드 추가 (BGE-M3)
"""
from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient, models

from whatisrag.core.config import settings
from whatisrag.core.schema import Document


# BGE-M3 dense 차원
DENSE_DIM = 1024


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def init_collection_dense(
    collection_name: str = settings.qdrant_collection,
    client: QdrantClient | None = None,
    recreate: bool = False,
) -> None:
    """Stage 0용 dense-only 컬렉션 초기화."""
    client = client or get_client()

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created dense collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")


def init_collection_hybrid(
    collection_name: str = settings.qdrant_collection,
    client: QdrantClient | None = None,
    recreate: bool = False,
) -> None:
    """Stage 1용 dense + sparse 컬렉션 초기화."""
    client = client or get_client()

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info(f"Created hybrid collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")


def upsert_documents(
    documents: list[Document],
    collection_name: str = settings.qdrant_collection,
    client: QdrantClient | None = None,
    batch_size: int = 64,
) -> None:
    """Document 리스트를 Qdrant에 업서트.
    dense_vector만 있으면 Stage 0 포맷, sparse_vector까지 있으면 Stage 1 포맷으로 자동 분기.
    """
    client = client or get_client()

    is_hybrid = any(doc.sparse_vector is not None for doc in documents)

    points: list[models.PointStruct] = []
    for doc in documents:
        if doc.dense_vector is None:
            raise ValueError(f"Document {doc.id} has no dense_vector. Run embedder first.")

        if is_hybrid and doc.sparse_vector is not None:
            sparse_indices = list(doc.sparse_vector.keys())
            sparse_values = list(doc.sparse_vector.values())
            vector = {
                "dense": doc.dense_vector,
                "sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            }
        else:
            vector = doc.dense_vector

        points.append(
            models.PointStruct(
                id=str(doc.id),
                vector=vector,
                payload={
                    "content": doc.content,
                    **doc.metadata,
                },
            )
        )

    # 배치 업서트
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

    logger.info(f"Upserted {len(points)} documents → {collection_name}")
