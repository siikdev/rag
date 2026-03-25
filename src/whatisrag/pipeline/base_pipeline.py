"""
BasePipeline 구체 베이스 클래스.
공통 유틸리티(LLM 호출, 컨텍스트 포매팅)를 제공.
각 Stage의 Pipeline은 이를 상속해서 run()만 구현.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from whatisrag.core.config import settings
from whatisrag.core.interfaces import BasePipeline
from whatisrag.core.schema import Document, GenerationResult

_DEFAULT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question based only on the following context.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


class RAGBasePipeline(BasePipeline):
    """공통 LLM 호출 로직을 포함한 베이스 구현체."""

    def __init__(
        self,
        openai_api_key: str = "",
        model: str = "",
        prompt: ChatPromptTemplate | None = None,
    ) -> None:
        api_key = openai_api_key or settings.openai_api_key
        self._llm = ChatOpenAI(
            api_key=api_key,
            model=model or settings.openai_model,
            temperature=0,
        )
        self._prompt = prompt or _DEFAULT_PROMPT
        self._chain = self._prompt | self._llm | StrOutputParser()

    def _format_context(self, documents: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[{i+1}] {doc.content}" for i, doc in enumerate(documents)
        )

    def _generate(self, query: str, documents: list[Document]) -> str:
        context = self._format_context(documents)
        return self._chain.invoke({"context": context, "question": query})

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, query: str) -> GenerationResult:
        raise NotImplementedError
