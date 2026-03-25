# evaluation — RAGAS 평가 시스템

> 모든 Stage의 Pipeline을 **동일한 QA 데이터셋**으로 평가하여
> 각 기법의 성능 향상을 수치로 검증하는 레이어 (Stage 3).

---

## 핵심 철학

"측정할 수 없으면 개선할 수 없다."
새 컴포넌트를 추가할 때마다 `benchmark.py`에 등록하고
수치로 효과를 검증합니다.

---

## 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `ragas_evaluator.py` | RAGAS 지표 계산 래퍼 | 🔲 Stage 3 예정 |
| `benchmark.py` | 여러 Pipeline을 동일 QA셋으로 비교 실행 | 🔲 Stage 3 예정 |
| `report.py` | 결과 DataFrame + 시각화 | 🔲 Stage 3 예정 |

---

## RAGAS 지표 설명

| 지표 | 측정 대상 | 이상적인 값 |
|------|-----------|------------|
| **Faithfulness** | 답변이 context에 근거하는가 (할루시네이션 탐지) | 높을수록 좋음 |
| **Answer Relevancy** | 답변이 질문과 얼마나 관련 있는가 | 높을수록 좋음 |
| **Context Precision** | 가져온 context 중 실제로 필요한 비율 | 높을수록 좋음 |
| **Context Recall** | 필요한 context를 빠짐없이 가져왔는가 | 높을수록 좋음 |

---

## Stage 3 구현 계획

### QA 데이터셋 구성 (`data/evaluation/`)
```json
[
  {
    "question": "...",
    "ground_truth": "...",
    "contexts": ["..."]   // 선택적, RAGAS가 자동 생성 가능
  }
]
```
초기에는 수동으로 10~20개 QA 쌍 작성 → 이후 LLM으로 자동 생성.

### Benchmark 실행 구조
```python
# scripts/run_benchmark.py 에서 호출
pipelines = [
    NaivePipeline(embedder=...),       # Stage 0
    HybridPipeline(retriever=...),     # Stage 1
    RerankPipeline(retriever=..., reranker=...),  # Stage 2
]
benchmark = Benchmark(pipelines=pipelines, qa_dataset=qa_data)
results = benchmark.run()
report.print_table(results)
```

### 기대 출력
```
Pipeline          Faithfulness  Answer Rel.  Context Prec.  Context Recall
──────────────────────────────────────────────────────────────────────────
NaivePipeline         0.72         0.68          0.61           0.55
HybridPipeline        0.81         0.75          0.73           0.68
RerankPipeline        0.88         0.82          0.85           0.74
```

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 3 | `ragas_evaluator`, `benchmark`, `report` 추가 예정 |
