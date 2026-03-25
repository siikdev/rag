# api — FastAPI 서빙 레이어

> RAG 파이프라인을 HTTP API로 제공하는 레이어.
> Stage 0부터 기본 구조를 갖추고, Stage가 올라갈수록 엔드포인트가 추가됩니다.

---

## 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `main.py` | FastAPI 앱 초기화 및 라우터 등록 | 🔲 예정 |
| `routers/ingest.py` | `POST /ingest` — 문서 수집 트리거 | 🔲 예정 |
| `routers/query.py` | `POST /query` — 질문 → 답변 | 🔲 예정 |
| `routers/evaluate.py` | `POST /evaluate` — RAGAS 평가 실행 | 🔲 Stage 3 예정 |

---

## API 설계

### `POST /query`
```json
// Request
{
  "question": "RAG에서 Hybrid Search가 왜 중요한가요?",
  "pipeline": "rerank",   // "naive" | "hybrid" | "rerank" | "graph" | "agentic"
  "top_k": 5
}

// Response
{
  "answer": "...",
  "source_documents": [...],
  "pipeline_used": "rerank",
  "latency_ms": 1234
}
```

### `POST /ingest`
```json
// Request
{
  "file_path": "data/raw/paper.pdf",
  "chunk_size": 512,
  "recreate": false
}

// Response
{
  "status": "success",
  "documents_indexed": 142
}
```

### `POST /evaluate` (Stage 3)
```json
// Request
{
  "pipelines": ["naive", "hybrid", "rerank"],
  "qa_dataset_path": "data/evaluation/qa.json"
}

// Response
{
  "results": [
    {"pipeline": "naive", "faithfulness": 0.72, ...},
    {"pipeline": "hybrid", "faithfulness": 0.81, ...}
  ]
}
```

---

## 실행

```bash
# 개발
make api

# Docker (예정)
docker compose up
```

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | 디렉토리 구조 및 인터페이스 설계 |
| 1 | `main.py`, `routers/ingest.py`, `routers/query.py` 구현 예정 |
| 3 | `routers/evaluate.py` 추가 예정 |
