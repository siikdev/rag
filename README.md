# What is RAG? — Big Tech-grade RAG System

> LLM Engineer 포트폴리오 프로젝트.
> Naive RAG에서 시작해 Hybrid Search, Reranking, Graph RAG, Agentic RAG까지
> **논문 기반으로 단계적으로 구현**하며 각 기법의 원리를 직접 검증합니다.

---

## 목표

- RAG의 각 구성 요소를 **모듈화**하여 Stage별로 누적 확장
- 이전 Stage의 컴포넌트를 **재사용**하며 성능 향상 과정을 RAGAS로 수치화
- FastAPI + Docker로 **프로덕션 수준의 서빙** 구조 유지

---

## 구현 현황

| Stage | 주제 | 핵심 기법 | 상태 |
|-------|------|-----------|------|
| 0 | 기반 구조 + Naive RAG | Dense Search, OpenAI Embedding | ✅ 완료 |
| 1 | Hybrid Search + Query Transformation | BGE-M3, RRF, HyDE, RAG-Fusion | 🔲 예정 |
| 2 | Reranking | Cross-encoder, Cohere Rerank | 🔲 예정 |
| 3 | 평가 시스템 | RAGAS (Faithfulness, Relevancy, Precision, Recall) | 🔲 예정 |
| 4 | Advanced RAG | RAPTOR, Self-RAG | 🔲 예정 |
| 5 | Graph RAG | LightRAG, Neo4j, Knowledge Graph | 🔲 예정 |
| 6 | Agentic RAG | LangGraph, Multi-Retriever Tool Use | 🔲 예정 |

---

## 아키텍처 개요

```
[Query Input]
      │
      ▼
[Query Transformer]   ← HyDE / RAG-Fusion / Step-back  (Stage 1)
      │
      ▼
[Retriever]           ← Dense / Hybrid / RAPTOR / Graph  (Stage 1~5)
      │
      ▼
[Reranker]            ← Cross-encoder / Cohere           (Stage 2)
      │
      ▼
[Generator]           ← OpenAI GPT-4o-mini
      │
      ▼
[Evaluator]           ← RAGAS                            (Stage 3)
```

각 레이어는 `core/interfaces.py`의 추상 인터페이스를 구현합니다.
Pipeline은 인터페이스에만 의존하므로 구현체를 런타임에 교체할 수 있습니다.

---

## 프로젝트 구조

```
whatisrag/
├── src/whatisrag/
│   ├── core/          # 인터페이스 & 공통 스키마 (변경 최소화 원칙)
│   ├── embedding/     # Embedder 구현체
│   ├── ingestion/     # 문서 로드 → 청킹 → Qdrant 인덱싱
│   ├── retrieval/     # Retriever 구현체 (Dense, Hybrid, RAPTOR, Graph)
│   ├── reranking/     # Reranker 구현체 (Cross-encoder, Cohere)
│   ├── graph/         # Knowledge Graph 구성 (Stage 5)
│   ├── pipeline/      # Stage별 RAG 파이프라인 조합
│   ├── evaluation/    # RAGAS 평가 및 Stage 간 벤치마크
│   └── api/           # FastAPI 서빙
├── data/
│   ├── raw/           # 원본 문서
│   ├── processed/     # 청킹 결과
│   └── evaluation/    # RAGAS QA 데이터셋
├── notebooks/         # Stage별 탐색 및 실험
├── scripts/           # ingest, benchmark 실행 스크립트
└── tests/
```

각 디렉토리의 상세 역할과 구현 계획은 해당 폴더의 `README.md`를 참고하세요.

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Orchestration | LangChain, LangGraph |
| Embedding | BGE-M3 (FlagEmbedding), OpenAI Embeddings |
| Vector Store | Qdrant (Hybrid Search 네이티브 지원) |
| Reranker | BGE-Reranker-v2-m3, Cohere Rerank |
| Graph DB | Neo4j, LightRAG |
| Evaluation | RAGAS, DeepEval |
| Serving | FastAPI, Docker |
| LLM | OpenAI GPT-4o-mini |

---

## 빠른 시작

```bash
# 1. 의존성 설치
poetry install

# 2. 환경변수 설정
cp .env.example .env
# .env에 OPENAI_API_KEY 입력

# 3. Qdrant 실행
make up

# 4. 문서 수집 (data/raw/ 에 PDF 넣고)
make ingest

# 5. API 서버 실행
make api
```

---

## Stage별 상세 노트

각 Stage 구현 완료 시 해당 섹션을 업데이트합니다.

### Stage 0 — Naive RAG
- Dense vector (OpenAI `text-embedding-3-small`) + Qdrant cosine 검색
- 베이스라인으로 이후 모든 Stage의 성능 비교 기준점

### Stage 1 — Hybrid Search _(예정)_
### Stage 2 — Reranking _(예정)_
### Stage 3 — RAGAS 평가 _(예정)_
### Stage 4 — Advanced RAG _(예정)_
### Stage 5 — Graph RAG _(예정)_
### Stage 6 — Agentic RAG _(예정)_

---

## 학습 문서

각 Stage의 개념과 원리는 [`docs/`](docs/) 폴더에 정리되어 있습니다.

| 문서 | 주제 |
|------|------|
| [Stage 0 — Naive RAG](docs/stage0_naive_rag.md) | RAG 개요와 한계 |
| [Stage 1 — Hybrid Search](docs/stage1_hybrid_search.md) | BGE-M3, Dense/Sparse, RRF, HyDE, RAG-Fusion |

---

## 참고 논문

| 논문 | Stage |
|------|-------|
| [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) | 1 |
| [RAG-Fusion](https://arxiv.org/abs/2402.03367) | 1 |
| [HyDE](https://arxiv.org/abs/2212.10496) | 1 |
| [RAPTOR](https://arxiv.org/abs/2401.18059) | 4 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 4 |
| [GraphRAG (Microsoft)](https://arxiv.org/abs/2404.16130) | 5 |
| [LightRAG](https://arxiv.org/abs/2410.05779) | 5 |
