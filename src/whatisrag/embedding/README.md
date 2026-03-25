# embedding — Embedder 구현체

> 텍스트를 벡터로 변환하는 레이어.
> `BaseEmbedder` 인터페이스를 구현하며, Stage에 따라 dense-only → dense+sparse로 확장됩니다.

---

## 역할

- 문서 청크와 쿼리를 벡터로 변환
- Stage 0은 OpenAI dense embedding 사용
- Stage 1부터 BGE-M3로 교체 → dense + sparse 벡터를 동시에 생성하여 Hybrid Search 지원

---

## 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `openai_embedder.py` | OpenAI `text-embedding-3-small` 기반 dense embedder | ✅ Stage 0 완료 |
| `bge_m3.py` | BGE-M3 dense + sparse 동시 생성 (FlagEmbedding) | 🔲 Stage 1 예정 |

---

## Stage별 구현 계획

### Stage 0 — OpenAI Embedder (완료)
- `text-embedding-3-small` (1536-dim) 사용
- dense vector만 생성
- Qdrant에 단일 vector로 저장

### Stage 1 — BGE-M3 Embedder (예정)
- `BAAI/bge-m3` 모델: 단일 모델로 dense + sparse + ColBERT 지원
- `FlagEmbedding` 라이브러리의 `BGEM3FlagModel` 사용
- dense: 1024-dim cosine similarity
- sparse: lexical weights `{token_id: weight}` — BM25-like 키워드 매칭
- Qdrant `named vectors` 포맷으로 저장: `{"dense": [...], "sparse": SparseVector(...)}`

```python
# BGE-M3 핵심 사용 패턴 (구현 예정)
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,  # Stage 1에서는 불필요
)
# output["dense_vecs"]      → list[list[float]]  (1024-dim)
# output["lexical_weights"] → list[dict[str, float]]  (token → weight)
```

---

## 인터페이스

```python
class BaseEmbedder(ABC):
    def embed_documents(self, texts: list[str]) -> list[Document]: ...
    def embed_query(self, text: str) -> Document: ...
```

`embed_documents`는 content와 vector가 채워진 `Document` 리스트를 반환합니다.
호출자(indexer)는 반환된 Document를 그대로 Qdrant에 업서트합니다.

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | `openai_embedder.py` — dense only |
| 1 | `bge_m3.py` 추가 — dense + sparse |
