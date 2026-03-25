# tests — 테스트

> `make test` 또는 `poetry run pytest tests/ -v`로 실행.

---

## 구조

```
tests/
├── unit/          # 개별 컴포넌트 단위 테스트
│   └── test_schema.py
└── integration/   # 파이프라인 end-to-end 테스트 (Qdrant 필요)
```

---

## 파일 목록

| 파일 | 테스트 대상 | 상태 |
|------|------------|------|
| `unit/test_schema.py` | `core/schema.py` 기본 동작 | ✅ Stage 0 완료 |
| `unit/test_chunker.py` | 청킹 전략 | 🔲 Stage 1 예정 |
| `unit/test_hybrid_retriever.py` | RRF 병합 로직 | 🔲 Stage 1 예정 |
| `unit/test_reranker.py` | Reranker 인터페이스 | 🔲 Stage 2 예정 |
| `integration/test_pipeline_stage0.py` | NaivePipeline end-to-end | 🔲 예정 |
| `integration/test_pipeline_stage1.py` | HybridPipeline end-to-end | 🔲 Stage 1 예정 |

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | `unit/test_schema.py` 추가 |
