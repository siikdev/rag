# data — 데이터 디렉토리

> `.gitignore`에 의해 `raw/`와 `processed/`는 Git에서 제외됩니다.
> `evaluation/` 폴더의 QA 데이터셋은 버전 관리됩니다.

---

## 구조

```
data/
├── raw/           # 원본 문서 (PDF, TXT 등) — git 제외
├── processed/     # 청킹 결과 캐시 — git 제외
└── evaluation/    # RAGAS 평가용 QA 데이터셋 — git 포함
    └── qa.json    # Stage 3에서 추가 예정
```

---

## 평가 데이터셋 형식 (`evaluation/qa.json`)

Stage 3에서 생성 예정:
```json
[
  {
    "question": "질문 텍스트",
    "ground_truth": "정답 텍스트",
    "source": "출처 파일명 (선택)"
  }
]
```

RAGAS는 `ground_truth`와 Pipeline이 생성한 `answer`를 비교하여 지표를 계산합니다.

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 3 | `evaluation/qa.json` 추가 예정 |
