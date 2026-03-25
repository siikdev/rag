# Stage 1 — Hybrid Search & Query Transformation

## 이 Stage에서 구현하는 것

Naive RAG의 두 가지 핵심 약점을 동시에 해결합니다.

| 약점 | 해결 기법 |
|------|-----------|
| "키워드는 맞지만 의미가 다른 문서"가 검색됨 | **Hybrid Search** (Dense + Sparse + RRF) |
| "질문이 짧거나 모호해서" 관련 문서를 못 찾음 | **Query Transformation** (HyDE, RAG-Fusion) |

---

## 1. BGE-M3

### 왜 BGE-M3인가?

일반적으로 Dense 임베딩과 Sparse 임베딩은 **별도의 모델**이 필요합니다.
(예: OpenAI Embedding + BM25)

**BGE-M3** (BAAI General Embedding, Multi-Lingual, Multi-Granularity, Multi-Functionality)는
단일 모델로 세 가지 벡터를 동시에 생성합니다.

| 출력 타입 | 차원 | 용도 |
|-----------|------|------|
| **Dense vector** | 1024-dim float | 의미 유사도 검색 |
| **Sparse vector** | 가변 (token → weight) | 키워드 매칭 검색 |
| **ColBERT vector** | token-level multi-vector | 정밀 재정렬 (Stage 2+) |

### 동작 원리

```
"오늘 날씨가 어때요?" 입력
         ↓
    BGE-M3 인코더
         ↓
Dense:  [0.12, -0.34, 0.89, ...]  (1024개 숫자 — 문장 전체 의미)
Sparse: {"오늘": 0.8, "날씨": 1.2, "어때요": 0.3}  (중요 단어와 가중치)
```

Dense는 문장 전체의 **의미**를 하나의 벡터로 압축합니다.
Sparse는 문장에서 중요한 **단어**들을 추출하고 가중치를 부여합니다.

### 사용 라이브러리

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
output = model.encode(
    ["검색할 텍스트"],
    return_dense=True,
    return_sparse=True,
)
# output["dense_vecs"]      → [[0.12, -0.34, ...]]  (1024-dim)
# output["lexical_weights"] → [{"검색": 0.9, "텍스트": 0.7}]
```

---

## 2. Dense vs Sparse 검색

두 방식은 서로 다른 종류의 쿼리에 강합니다.

### Dense Search (의미 기반)

임베딩 모델이 텍스트를 고차원 벡터로 변환하고,
**Cosine Similarity**로 가장 가까운 벡터를 찾습니다.

```
쿼리: "심장 발작 증상"
      ↓ 임베딩
      [0.23, 0.87, -0.12, ...]

문서A: "myocardial infarction symptoms"  → 벡터 거리: 0.05  ← 가까움 (의미 유사)
문서B: "심근경색의 주요 징후"              → 벡터 거리: 0.08  ← 가까움 (의미 유사)
문서C: "심장 박동 측정법"                 → 벡터 거리: 0.45  ← 멈
```

**강점:** 다른 언어, 동의어, 의역도 잘 찾음
**약점:** 특정 키워드(고유명사, 코드, 모델명)가 중요한 경우 약함

### Sparse Search (키워드 기반)

BM25 알고리즘이 기반. 문서에 **실제로 등장하는 단어**의 빈도와 중요도로 점수를 매깁니다.
BGE-M3의 sparse vector는 BM25와 유사하지만 신경망 기반으로 가중치를 더 정교하게 계산합니다.

```
쿼리: "GPT-4o 파라미터 수"

문서A: "GPT-4o는 약 200B 파라미터로..."  → 키워드 매칭: GPT-4o ✓, 파라미터 ✓  → 높은 점수
문서B: "대규모 언어 모델의 규모에 대해..."  → 키워드 매칭: 없음  → 낮은 점수
       (Dense에서는 의미상 관련있어 높은 점수 받을 수 있음)
```

**강점:** 고유명사, 모델명, 코드, 전문 용어에 강함
**약점:** 동의어, 다른 언어 표현을 인식 못함

### 두 방식의 보완 관계

```
Dense만 사용:
  쿼리 "LLaMA 3 context window"
  → "대형 언어 모델의 긴 문맥 처리" 같은 문서를 반환 (의미는 비슷하지만 원하는 문서가 아님)

Sparse만 사용:
  쿼리 "심장 마비 예방법"
  → "cardiac arrest prevention" 문서를 못 찾음 (키워드가 다름)

Hybrid:
  → 두 결과를 RRF로 합쳐서 두 약점을 상호 보완
```

---

## 3. RRF (Reciprocal Rank Fusion)

Dense 검색 결과와 Sparse 검색 결과를 **어떻게 합칠** 것인가?

단순히 점수를 더하면 안 됩니다. Dense 점수는 0~1 범위 cosine similarity이고,
Sparse 점수는 BM25 기반으로 범위가 다릅니다. 스케일이 다른 점수를 더하면 한쪽이 지배합니다.

### RRF 공식

```
RRF_score(문서 d) = Σ [ 1 / (k + rank_i(d)) ]

k = 60  (표준값, 상위 랭크의 영향력을 완화)
rank_i(d) = i번째 검색 결과 리스트에서 문서 d의 순위
```

### 예시

Dense 검색 결과 (top 5):
```
1위: 문서A (score: 0.92)
2위: 문서B (score: 0.88)
3위: 문서C (score: 0.71)
```

Sparse 검색 결과 (top 5):
```
1위: 문서C (score: 14.2)
2위: 문서D (score: 11.8)
3위: 문서A (score: 9.3)
```

RRF 계산 (k=60):
```
문서A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01563 = 0.03202
문서B: 1/(60+2) + 0        = 0.01613
문서C: 1/(60+3) + 1/(60+1) = 0.01563 + 0.01639 = 0.03202
문서D: 0        + 1/(60+2) = 0.01613

최종 순위:
1위: 문서A = 문서C (동점) ← 두 검색에서 모두 상위권
3위: 문서B = 문서D
```

**핵심:** RRF는 점수의 절대값이 아니라 **순위**만 사용하므로,
스케일이 다른 검색 결과를 공정하게 합칠 수 있습니다.

### Qdrant에서의 구현

Qdrant는 RRF를 내장 기능으로 지원합니다. 직접 구현할 필요가 없습니다.

```python
results = client.query_points(
    collection_name="whatisrag",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=50),
        Prefetch(query=SparseVector(...), using="sparse", limit=50),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # ← Qdrant 내장 RRF
    limit=10,
)
```

---

## 4. HyDE (Hypothetical Document Embeddings)

논문: [Precise Zero-Shot Dense Retrieval without Relevance Labels (2022)](https://arxiv.org/abs/2212.10496)

### 문제: 쿼리와 문서의 벡터 공간 불일치

질문과 답변은 텍스트 패턴이 다릅니다.

```
질문: "RAG에서 청킹 크기는 어떻게 정하나요?"
       → 짧고, 의문문, 구체적인 답 없음

관련 문서: "청킹 크기 선택 가이드: chunk_size=512일 때 recall이 높고..."
           → 길고, 평서문, 구체적인 내용 포함
```

이 두 텍스트는 의미상 관련 있지만,
**벡터 공간에서 패턴이 달라서** 거리가 예상보다 멀 수 있습니다.

### HyDE 아이디어

"질문의 임베딩 대신, **질문에 대한 가상 답변**의 임베딩으로 검색하면 어떨까?"

```
원본 쿼리: "RAG에서 청킹 크기는 어떻게 정하나요?"
                    ↓  LLM으로 가상 답변 생성
가상 문서: "RAG 시스템에서 청킹 크기는 일반적으로 256~1024 토큰이 권장됩니다.
           chunk_size=512는 검색 정확도와 컨텍스트 보존의 균형점으로..."
                    ↓  가상 문서를 임베딩
가상 임베딩: [0.45, -0.23, ...]  ← 이걸로 Qdrant 검색
```

가상 문서는 틀려도 됩니다. 중요한 것은 **실제 관련 문서와 비슷한 패턴**을 가진다는 것입니다.

### 효과

```
일반 검색:  질문 임베딩  ---거리: 0.4---  관련 문서
HyDE 검색:  가상 문서 임베딩  ---거리: 0.1---  관련 문서
```

실제 문서와 패턴이 비슷한 가상 문서를 통해 더 가까운 거리로 검색됩니다.

### 구현 개요

```python
class HyDETransformer(BaseQueryTransformer):
    def transform(self, query: str) -> list[str]:
        # LLM으로 가상 답변 생성
        hypothetical_doc = self._llm.invoke(
            f"다음 질문에 대한 답변 문서를 작성하세요:\n{query}"
        )
        return [hypothetical_doc]  # 원본 쿼리 대신 가상 문서 반환
```

### 주의점

- LLM 호출 비용 발생 (질문마다 1회 추가 호출)
- 생성된 가상 문서가 전혀 엉뚱한 방향이면 오히려 성능 저하 가능
- 팩트가 중요한 도메인보다 **개념 설명** 타입의 질문에 효과적

---

## 5. RAG-Fusion

논문: [RAG-Fusion: a New Take on Retrieval-Augmented Generation (2024)](https://arxiv.org/abs/2402.03367)

### 문제: 단일 쿼리의 한계

사용자의 질문은 하나의 표현이지만, 관련 정보는 다양한 표현으로 문서에 존재합니다.

```
질문: "transformer 어텐션 메커니즘 설명해줘"

관련 문서들이 사용하는 표현:
- "self-attention mechanism in transformers"
- "query-key-value attention"
- "멀티헤드 어텐션의 원리"
- "scaled dot-product attention 수식"
```

하나의 쿼리 임베딩이 이 모든 변형을 커버하기 어렵습니다.

### RAG-Fusion 아이디어

"하나의 쿼리를 **여러 관점의 변형 쿼리**로 확장하고, 각각 검색 후 RRF로 병합하자"

```
원본 쿼리: "transformer 어텐션 메커니즘 설명해줘"
                    ↓  LLM으로 변형 쿼리 N개 생성
변형 쿼리 1: "self-attention이 어떻게 작동하나요?"
변형 쿼리 2: "transformer에서 query, key, value의 역할은?"
변형 쿼리 3: "scaled dot-product attention 수식 설명"
변형 쿼리 4: "멀티헤드 어텐션의 장점"
                    ↓  각 쿼리로 병렬 검색
검색 결과 1: [문서A, 문서C, 문서E, ...]
검색 결과 2: [문서B, 문서A, 문서F, ...]
검색 결과 3: [문서C, 문서G, 문서A, ...]
검색 결과 4: [문서D, 문서C, 문서B, ...]
                    ↓  RRF로 최종 병합
최종 결과: [문서A, 문서C, 문서B, ...]  ← 여러 쿼리에서 상위권인 문서가 최상위
```

### HyDE와의 차이

| | HyDE | RAG-Fusion |
|--|------|------------|
| 접근 방식 | 가상 답변 문서 생성 | 다양한 쿼리 변형 생성 |
| LLM 호출 | 1회 | N회 (쿼리 수만큼) |
| 적합한 상황 | 질문과 문서 패턴이 다를 때 | 쿼리가 모호하거나 다양한 표현이 필요할 때 |
| 검색 횟수 | 1회 | N회 (병렬) |

### 구현 개요

```python
class RAGFusionTransformer(BaseQueryTransformer):
    def transform(self, query: str) -> list[str]:
        # LLM으로 변형 쿼리 4개 생성
        queries = self._llm.invoke(
            f"다음 질문의 변형 버전 4개를 생성하세요:\n{query}"
        )
        return [query] + queries  # 원본 포함 총 5개

# HybridRetriever에서:
all_results = []
for q in transformer.transform(original_query):
    results = hybrid_search(q, top_k=10)
    all_results.append(results)

final = rrf_merge(all_results)  # N개 결과 리스트를 RRF로 병합
```

---

## 전체 Stage 1 파이프라인 흐름

```
사용자 입력: "transformer의 attention이 왜 중요한가요?"
                    │
                    ▼
         [QueryTransformer]
          HyDE 또는 RAG-Fusion으로
          쿼리 1~5개로 확장
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   변형 쿼리 1            변형 쿼리 2 ...
         │                     │
         ▼                     ▼
   [HybridRetriever]     [HybridRetriever]
   Dense(BGE-M3) ──┐     Dense(BGE-M3) ──┐
   Sparse(BGE-M3)──┘ RRF Sparse(BGE-M3)──┘ RRF
         │                     │
         └──────────┬──────────┘
                    ▼
              [최종 RRF 병합]
              (쿼리 간 결과 통합)
                    │
                    ▼
              상위 k개 문서
                    │
                    ▼
              [GPT-4o-mini]
                    │
                    ▼
                 최종 답변
```

---

## 성능 향상 기대치

RAGAS 기준 (Stage 3에서 실측 예정):

| 지표 | Naive RAG | Hybrid Search | +Query Transform |
|------|-----------|---------------|-----------------|
| Context Recall | ~0.55 | ~0.68 | ~0.75 |
| Context Precision | ~0.61 | ~0.73 | ~0.70 |
| Faithfulness | ~0.72 | ~0.78 | ~0.80 |

> Hybrid Search는 Recall을 크게 높이고,
> Query Transformation은 모호한 질문에서 Recall을 추가로 높입니다.
> (실측값은 데이터셋에 따라 다를 수 있음)

---

## 참고 자료

- [BGE-M3 논문](https://arxiv.org/abs/2402.03016)
- [HyDE 논문](https://arxiv.org/abs/2212.10496)
- [RAG-Fusion 논문](https://arxiv.org/abs/2402.03367)
- [Qdrant Hybrid Search 공식 문서](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [RRF 원본 논문 (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114)
