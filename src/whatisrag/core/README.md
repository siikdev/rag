# core — 공통 인터페이스 & 스키마

> **변경 최소화 원칙**: 이 패키지는 시스템 전체의 계약(Contract)을 정의합니다.
> Stage가 바뀌어도 이 파일들은 수정하지 않는 것이 목표입니다.
> 수정이 필요하다면 반드시 하위 호환성을 유지하세요.

---

## 역할

모든 컴포넌트(Embedder, Retriever, Reranker, Pipeline)가 의존하는 **공통 레이어**.
구체 구현체는 여기 정의된 추상 클래스를 상속하고, Pipeline은 인터페이스에만 의존합니다.

```
core/interfaces.py  ←  모든 구현체가 이 계약을 따름
core/schema.py      ←  시스템 전체 공통 데이터 모델
core/config.py      ←  환경변수 기반 설정
```

---

## 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `schema.py` | `Document`, `RetrievalResult`, `RankedResult`, `GenerationResult`, `EvaluationResult` | ✅ 완료 |
| `interfaces.py` | `BaseEmbedder`, `BaseRetriever`, `BaseQueryTransformer`, `BaseReranker`, `BasePipeline` | ✅ 완료 |
| `config.py` | pydantic-settings 기반 전역 설정 (`settings` 싱글턴) | ✅ 완료 |

---

## 핵심 설계

### 전략 패턴 (Strategy Pattern)

```python
# Pipeline은 인터페이스에만 의존
pipeline = RerankPipeline(
    retriever=HybridRetriever(...),   # Stage 1 구현체
    reranker=CrossEncoderReranker(),  # Stage 2 구현체
)

# 나중에 Retriever만 교체하면 나머지는 그대로
pipeline = RerankPipeline(
    retriever=GraphRetriever(...),    # Stage 5 구현체로 교체
    reranker=CrossEncoderReranker(),  # 그대로 재사용
)
```

### 데이터 흐름

```
Document
  └─ content: str
  └─ metadata: dict
  └─ dense_vector: list[float]   ← Stage 0+: OpenAI / BGE-M3
  └─ sparse_vector: dict[int, float]  ← Stage 1+: BGE-M3 lexical

RetrievalResult
  └─ document: Document
  └─ score: float
  └─ retriever_name: str

RankedResult          ← Stage 2+
  └─ rerank_score: float
  └─ original_score: float

GenerationResult      ← Pipeline.run() 반환값
  └─ answer: str
  └─ source_documents: list[Document]
  └─ intermediate_steps: dict   ← RAGAS 평가 입력으로 활용
```

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | 초기 정의 — schema, interfaces, config |
