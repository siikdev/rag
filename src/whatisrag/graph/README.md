# graph — Knowledge Graph 구성 및 Graph RAG

> 문서에서 Entity/Relation을 추출해 Knowledge Graph를 구성하고,
> 그래프 탐색을 통한 검색을 지원하는 레이어 (Stage 5).

---

## 핵심 개념

일반 RAG는 텍스트 유사도 기반으로 문서를 검색합니다.
Graph RAG는 개념 간 **관계(Relation)**를 활용하여 전통적 검색이 놓치는
글로벌 질문("전체적으로 이 분야의 핵심 논쟁은 무엇인가?")에 강합니다.

**Microsoft GraphRAG (2024):**
```
문서
  → Entity/Relation 추출 (LLM)
  → Knowledge Graph 구성
  → Community Detection (Leiden algorithm)
  → Community별 요약 생성
  → Local query: 특정 엔티티 중심 탐색
  → Global query: Community 요약 기반 응답
```
논문: [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130)

**LightRAG (2024):** GraphRAG의 경량화 버전. 구현 이해가 쉬워 학습 목적으로 먼저 구현.
논문: [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779)

---

## 파일 목록

| 파일 | 역할 | 상태 |
|------|------|------|
| `builder.py` | Entity/Relation 추출 → NetworkX/Neo4j 저장 | 🔲 Stage 5 예정 |
| `lightrag_adapter.py` | LightRAG 라이브러리 연동 | 🔲 Stage 5 예정 |

---

## Stage 5 구현 계획

### 1단계: Entity/Relation 추출
```python
# builder.py 핵심 로직 (예정)
prompt = """
텍스트에서 모든 Entity와 Relation을 JSON으로 추출하세요.
형식: {"entities": [...], "relations": [{"from": ..., "to": ..., "type": ...}]}
"""
```

### 2단계: 그래프 저장
- 초기: NetworkX (인메모리, 구현 단순)
- 이후: Neo4j (영속성, Cypher 쿼리 지원)

### 3단계: LightRAG 연동
- LightRAG는 자체 그래프 구축 + 검색 파이프라인 제공
- 먼저 LightRAG로 전체 흐름 이해 → 직접 구현으로 심화

### 4단계: GraphRetriever
```python
# retrieval/graph_retriever.py 에서 이 모듈 활용
class GraphRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        # 1. 쿼리에서 Entity 추출
        # 2. 그래프에서 관련 서브그래프 탐색
        # 3. 연결된 청크 반환
```

---

## 인프라

Stage 5에서 `docker-compose.yml`에 Neo4j 서비스 추가 예정:
```yaml
neo4j:
  image: neo4j:5
  ports:
    - "7474:7474"  # Browser UI
    - "7687:7687"  # Bolt
```

---

## 변경 이력

| Stage | 변경 내용 |
|-------|-----------|
| 5 | `builder.py`, `lightrag_adapter.py` 추가 예정 |
