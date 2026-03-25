# reranking — Reranker 구현체

> Retriever가 가져온 후보 문서를 **쿼리-문서 쌍**으로 함께 보고 정밀 재정렬하는 레이어 (Stage 2).
> 검색 속도(bi-encoder)와 정확도(cross-encoder)의 트레이드오프를 Two-stage로 해결합니다.

---

## Two-Stage Retrieval 패턴

```
1단계 (Recall 극대화)
  HybridRetriever → top_k=50  (빠름, 전체 코퍼스 커버)

2단계 (Precision 극대화)
  Reranker → top_k=5  (느리지만 정확, 후보 50개만 처리)
```

---

## 파일 목록

| 파일 | 클래스 | 방식 | 상태 |
|------|--------|------|------|
| `cross_encoder.py` | `CrossEncoderReranker` | 로컬 BGE-Reranker-v2-m3 | 🔲 Stage 2 예정 |
| `cohere_reranker.py` | `CohereReranker` | Cohere Rerank API | 🔲 Stage 2 예정 |

---

## Stage 2 구현 계획

### Cross-encoder vs Bi-encoder 차이

| | Bi-encoder (Retriever) | Cross-encoder (Reranker) |
|--|------------------------|--------------------------|
| 입력 | 쿼리, 문서를 각각 인코딩 | 쿼리+문서를 함께 인코딩 |
| 속도 | 빠름 (사전 계산 가능) | 느림 (쌍마다 추론 필요) |
| 정확도 | 상대적으로 낮음 | 높음 (상호작용 포착) |
| 역할 | 전체 코퍼스 검색 | 후보군 재정렬 |

### CrossEncoderReranker
- 모델: `BAAI/bge-reranker-v2-m3`
- `sentence-transformers`의 `CrossEncoder` 활용
- CPU/GPU 자동 감지

### CohereReranker
- Cohere `rerank-english-v3.0` API
- 로컬 GPU 없을 때 대안

---

## 인터페이스

```python
class BaseReranker(ABC):
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RankedResult]: ...
```

`RankedResult`는 `rerank_score`와 `original_score`를 모두 보존하여
점수 변화를 분석하거나 RAGAS 평가에 활용할 수 있습니다.

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 2 | `cross_encoder.py`, `cohere_reranker.py` 추가 예정 |
