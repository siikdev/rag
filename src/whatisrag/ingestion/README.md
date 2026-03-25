# ingestion — 문서 수집 → 청킹 → 인덱싱

> 외부 문서를 RAG 시스템에서 사용할 수 있는 형태로 변환하는 파이프라인.
> `loader → chunker → (embedder) → indexer` 순서로 실행됩니다.

---

## 역할

- 다양한 형식(PDF, TXT, URL)의 문서를 로드
- 청킹 전략에 따라 Document 단위로 분할
- Qdrant에 dense / hybrid 벡터로 저장

---

## 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `loader.py` | PDF, TXT, 디렉토리 로드 → `Document` 리스트 | ✅ Stage 0 완료 |
| `chunker.py` | Fixed / Semantic 청킹 전략 | ✅ Stage 0 완료 |
| `indexer.py` | Qdrant 컬렉션 초기화 + Document 업서트 | ✅ Stage 0 완료 |

---

## 사용 흐름

```python
from whatisrag.ingestion import load, chunk, ChunkStrategy
from whatisrag.ingestion import init_collection_dense, upsert_documents
from whatisrag.embedding import OpenAIEmbedder

# 1. 로드
docs = load("data/raw/paper.pdf")

# 2. 청킹
chunks = chunk(docs, strategy=ChunkStrategy.FIXED, chunk_size=512)

# 3. 임베딩 (embedder는 ingestion 외부에서 주입)
embedder = OpenAIEmbedder()
embedded = embedder.embed_documents([c.content for c in chunks])
for chunk_doc, emb in zip(chunks, embedded):
    chunk_doc.dense_vector = emb.dense_vector

# 4. 인덱싱
init_collection_dense(recreate=False)
upsert_documents(chunks)
```

---

## Stage별 구현 계획

### Stage 0 — 기본 구조 (완료)
- `PyPDFLoader`, `TextLoader` 지원
- `RecursiveCharacterTextSplitter` (Fixed 청킹)
- Qdrant dense-only 컬렉션 (`vectors_config=VectorParams(size=1024)`)

### Stage 1 — Hybrid 인덱싱 (예정)
- `init_collection_hybrid()` 활성화
- BGE-M3 sparse vector 저장 지원
- `indexer.py`의 `upsert_documents`는 이미 dense/hybrid 자동 분기 구현됨

### Stage 1+ — Semantic Chunking (예정)
- `chunker.py`의 `SemanticChunker` 활성화
- 문장 임베딩 유사도 기반으로 의미 단위 분할
- 일반적으로 Fixed 청킹보다 검색 품질 향상

### Stage 4 — RAPTOR용 계층 청킹 (예정)
- 청크 → 클러스터링 → 요약 → 요약도 인덱싱
- `raptor_indexer.py` 별도 추가 예정

---

## 청킹 전략 비교

| 전략 | 장점 | 단점 | 사용 Stage |
|------|------|------|-----------|
| Fixed | 빠름, 간단 | 문장 중간 절단 가능 | 0 (베이스라인) |
| Semantic | 의미 단위 보존 | OpenAI 임베딩 비용 발생 | 1+ |
| Parent-Child | 검색 정확도 + 컨텍스트 보존 | 구현 복잡 | 2+ 예정 |
| RAPTOR | 계층적 요약으로 글로벌 질문 대응 | 비용 높음 | 4 |

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | `loader`, `chunker` (Fixed), `indexer` (dense) 초기 구현 |
| 1 | `indexer.hybrid` 활성화, Semantic Chunker 추가 예정 |
