# scripts — 실행 스크립트

> `make` 명령어로 호출하는 CLI 스크립트 모음.

---

## 파일 목록

| 파일 | 명령어 | 역할 | 상태 |
|------|--------|------|------|
| `ingest_sample_data.py` | `make ingest` | 문서 수집 → 청킹 → Qdrant 인덱싱 | ✅ Stage 0 완료 |
| `run_benchmark.py` | `make benchmark` | Stage별 RAGAS 성능 비교 | 🔲 Stage 3 예정 |

---

## 사용법

```bash
# 기본 (data/raw/ 디렉토리 전체)
make ingest

# 특정 파일
poetry run python scripts/ingest_sample_data.py --path data/raw/paper.pdf

# 컬렉션 초기화 후 재수집
poetry run python scripts/ingest_sample_data.py --recreate

# 청크 크기 조정
poetry run python scripts/ingest_sample_data.py --chunk-size 256
```

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 0 | `ingest_sample_data.py` 구현 |
| 3 | `run_benchmark.py` 구현 예정 |
