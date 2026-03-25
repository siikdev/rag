# pipeline — Stage별 RAG 파이프라인

> 각 Stage의 컴포넌트를 조합하여 완전한 RAG 파이프라인을 구성하는 레이어.
> **이전 Stage의 Pipeline은 절대 수정하지 않고** 새 Pipeline 클래스를 추가하는 방식으로 확장합니다.
> 모든 Pipeline은 `BasePipeline.run(query) → GenerationResult` 인터페이스를 따릅니다.

---

## 파일 목록

| 파일 | Pipeline 이름 | Stage | 상태 |
|------|--------------|-------|------|
| `base_pipeline.py` | `RAGBasePipeline` | 공통 | ✅ 완료 |
| `naive_pipeline.py` | `NaivePipeline` | 0 | ✅ 완료 |
| `hybrid_pipeline.py` | `HybridPipeline` | 1 | 🔲 예정 |
| `rerank_pipeline.py` | `RerankPipeline` | 2 | 🔲 예정 |
| `advanced_pipeline.py` | `RAPTORPipeline`, `SelfRAGPipeline` | 4 | 🔲 예정 |
| `graph_pipeline.py` | `GraphRAGPipeline` | 5 | 🔲 예정 |
| `agentic_pipeline.py` | `AgenticRAGPipeline` | 6 | 🔲 예정 |

---

## Stage별 파이프라인 설계

### Stage 0 — NaivePipeline (완료)
```
query → Qdrant dense search (top_k=5) → GPT-4o-mini
```

### Stage 1 — HybridPipeline (예정)
```
query
  → QueryTransformer (HyDE or RAG-Fusion)
  → HybridRetriever (Dense + Sparse + RRF, top_k=10)
  → GPT-4o-mini
```

### Stage 2 — RerankPipeline (예정)
```
query
  → QueryTransformer
  → HybridRetriever (top_k=50)   ← first-stage
  → CrossEncoderReranker (top_k=5)  ← second-stage
  → GPT-4o-mini
```

### Stage 4 — SelfRAGPipeline (예정)
```
query
  → [Retrieve 필요?] ─No→ 직접 생성
        │ Yes
        ▼
  → HybridRetriever
  → [결과 관련성 평가 (ISREL)]
  → [생성 + 지지 근거 평가 (ISSUP)]
  → [최종 답변 유용성 평가 (ISUSE)]
```

### Stage 6 — AgenticRAGPipeline (예정)
```
query
  → LangGraph Router
      ├─ HybridRetriever Tool
      ├─ GraphRetriever Tool
      └─ RAPTORRetriever Tool
  → Reranker
  → Generator
  → Answer Grader
  → (Self-Reflection loop if quality < threshold)
```

---

## 컴포지션 원칙

Stage 2의 `RerankPipeline`은 Stage 1의 `HybridRetriever`를 직접 인스턴스로 받아 사용합니다.
새 Pipeline을 만들 때 기존 컴포넌트를 재사용하는 방식:

```python
# Stage 2 구현 예시
class RerankPipeline(RAGBasePipeline):
    def __init__(self, retriever: BaseRetriever, reranker: BaseReranker, ...):
        self._retriever = retriever  # Stage 1의 HybridRetriever 주입
        self._reranker = reranker
```

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | `base_pipeline.py` (LLM 공통 로직), `naive_pipeline.py` (Dense-only 베이스라인) |
| 1 | `hybrid_pipeline.py` 추가 예정 |
| 2 | `rerank_pipeline.py` 추가 예정 |
