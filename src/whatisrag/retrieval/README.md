# retrieval — Retriever 구현체

> 쿼리에 관련된 문서를 찾아오는 레이어.
> 모든 구현체는 `BaseRetriever.retrieve(query, top_k) → list[RetrievalResult]` 인터페이스를 따릅니다.
> Stage가 올라갈수록 새 Retriever 클래스가 추가되며, 기존 클래스는 수정하지 않습니다.

---

## 파일 목록

| 파일 | 클래스 | Stage | 상태 |
|------|--------|-------|------|
| `dense_retriever.py` | `DenseRetriever` | 1 | 🔲 예정 |
| `sparse_retriever.py` | `SparseRetriever` | 1 | 🔲 예정 |
| `hybrid_retriever.py` | `HybridRetriever` | 1 ★ | 🔲 예정 |
| `query_transformer.py` | `HyDETransformer`, `RAGFusionTransformer` | 1 | 🔲 예정 |
| `raptor_retriever.py` | `RAPTORRetriever` | 4 | 🔲 예정 |
| `self_rag_retriever.py` | `SelfRAGRetriever` | 4 | 🔲 예정 |
| `graph_retriever.py` | `GraphRetriever` | 5 | 🔲 예정 |

> ★ `HybridRetriever`는 Stage 1부터 6까지 모든 파이프라인의 **first-stage retriever**로 재사용됩니다.

---

## Stage별 구현 계획

### Stage 1 — Hybrid Search

**핵심 개념: Dense + Sparse + RRF**

| 방식 | 모델 | 특징 |
|------|------|------|
| Dense | BGE-M3 (1024-dim) | 의미 유사도 강함 |
| Sparse | BGE-M3 lexical weights | 키워드 정확도 강함 |
| Hybrid | RRF 융합 | 두 방식의 장점 결합 |

**RRF (Reciprocal Rank Fusion) 공식:**
```
RRF_score(d) = Σ 1 / (k + rank_i(d))   k=60 (표준값)
```

**Qdrant 네이티브 Hybrid Search:**
```python
results = client.query_points(
    collection_name=collection_name,
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
        Prefetch(query=SparseVector(...), using="sparse", limit=top_k * 2),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
)
```
Qdrant의 내장 RRF를 사용하므로 별도 Python 구현 불필요.

**Query Transformation:**

- **HyDE (Hypothetical Document Embeddings)**: 쿼리로 가상 답변 문서 생성 → 그 임베딩으로 검색
  - 논문: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- **RAG-Fusion**: 쿼리 변형 N개 생성 → 병렬 검색 → RRF 병합
  - 논문: [RAG-Fusion](https://arxiv.org/abs/2402.03367)

### Stage 4 — RAPTOR Retriever

**핵심 개념: 계층적 문서 트리**
```
원본 청크
  → UMAP 차원 축소 + GMM 클러스터링
  → 클러스터별 요약 생성 (LLM)
  → 요약도 동일 Qdrant에 인덱싱
  → 쿼리 시 전체 트리에서 검색
```
논문: [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

**Self-RAG:**
- Special tokens(`[Retrieve]`, `[ISREL]`, `[ISSUP]`, `[ISUSE]`)로 모델이 스스로 검색 필요 여부와 결과 품질을 판단
- 논문: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

### Stage 5 — Graph Retriever

Knowledge Graph 기반 검색. Local query(특정 엔티티)와 Global query(전체 요약) 분기.
자세한 내용은 `graph/README.md` 참고.

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 1 | `dense_retriever`, `sparse_retriever`, `hybrid_retriever`, `query_transformer` 추가 예정 |
| 4 | `raptor_retriever`, `self_rag_retriever` 추가 예정 |
| 5 | `graph_retriever` 추가 예정 |
