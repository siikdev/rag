"""Core schema 기본 동작 테스트."""
from whatisrag.core.schema import Document, GenerationResult, RetrievalResult


def test_document_defaults():
    doc = Document(content="hello world")
    assert doc.content == "hello world"
    assert doc.id is not None
    assert doc.dense_vector is None
    assert doc.sparse_vector is None


def test_retrieval_result():
    doc = Document(content="test")
    result = RetrievalResult(document=doc, score=0.9, retriever_name="naive")
    assert result.score == 0.9


def test_generation_result():
    doc = Document(content="context")
    result = GenerationResult(answer="answer", source_documents=[doc])
    assert result.answer == "answer"
    assert len(result.source_documents) == 1
