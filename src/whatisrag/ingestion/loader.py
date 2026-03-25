"""
문서 로더 — PDF, TXT, 디렉토리 지원.
LangChain DocumentLoader를 래핑해서 내부 Document 스키마로 변환.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from loguru import logger

from whatisrag.core.schema import Document


def load_pdf(path: str | Path) -> list[Document]:
    """PDF 파일 한 개 로드 → Document 리스트 (페이지 단위)."""
    path = Path(path)
    logger.info(f"Loading PDF: {path}")
    loader = PyPDFLoader(str(path))
    lc_docs = loader.load()
    return [
        Document(
            content=doc.page_content,
            metadata={**doc.metadata, "source": str(path)},
        )
        for doc in lc_docs
        if doc.page_content.strip()
    ]


def load_text(path: str | Path) -> list[Document]:
    """텍스트 파일 한 개 로드."""
    path = Path(path)
    logger.info(f"Loading text: {path}")
    loader = TextLoader(str(path), encoding="utf-8")
    lc_docs = loader.load()
    return [
        Document(
            content=doc.page_content,
            metadata={**doc.metadata, "source": str(path)},
        )
        for doc in lc_docs
        if doc.page_content.strip()
    ]


def load_directory(dir_path: str | Path, glob: str = "**/*.pdf") -> list[Document]:
    """디렉토리 내 모든 파일 로드."""
    dir_path = Path(dir_path)
    logger.info(f"Loading directory: {dir_path} (glob={glob})")
    loader = DirectoryLoader(str(dir_path), glob=glob, loader_cls=PyPDFLoader)
    lc_docs = loader.load()
    return [
        Document(
            content=doc.page_content,
            metadata={**doc.metadata},
        )
        for doc in lc_docs
        if doc.page_content.strip()
    ]


def load(path: str | Path) -> list[Document]:
    """확장자에 따라 자동 로더 선택."""
    path = Path(path)
    if path.is_dir():
        return load_directory(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".txt", ".md"):
        return load_text(path)
    raise ValueError(f"Unsupported file type: {ext}")
