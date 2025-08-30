"""
index_documents.py — Table indexer for RAG

Goal
-------
Ingest a PDF/DOCX file, split it into chunks (one of: fixed|sentence|paragraph),
create Gemini embeddings for each chunk, and persist rows into a single
PostgreSQL table with pgvector.

Table: indexed_chunks
---------------------
- id BIGSERIAL PRIMARY KEY
- chunk_text TEXT
- embedding VECTOR(768)
- file_name TEXT
- split_strategy TEXT
- created_at TIMESTAMPTZ DEFAULT NOW()

Usage
-----
python index_documents.py --file <something.pdf|.docx> --strategy <fixed|sentence|paragraph>
"""
import argparse
import logging
import os
import re
import sys
import time
from typing import Iterable, List

from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from gemini_handling import embed_chunks
from db_handling import get_db_connection, ensure_schema, insert_chunks

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800                            # characters per chunk for packing
CHUNK_OVERLAP = 200                         # overlap in characters between adjacent chunks


def _detect_file_type(path: str) -> str:
    """Detect file type based on file extension (pdf or docx)."""
    lowercase_file_path = path.lower()
    if lowercase_file_path.endswith(".pdf"):
        return "pdf"
    if lowercase_file_path.endswith(".docx"):
        return "docx"
    raise ValueError("Unsupported file type. Use .pdf or .docx")


def _extract_text_pdf(path: str) -> str:
    """Extract and normalize text content from a PDF file."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    return _normalize_text(text)


def _extract_text_docx(path: str) -> str:
    """Extract and normalize text content from a DOCX file."""
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    text = "\n".join(paragraphs)
    return _normalize_text(text)


def _normalize_text(text: str) -> str:
    """Clean up text formatting while preserving paragraph structure."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # Limit consecutive newlines
    text = re.sub(r"[\t\x0b\x0c]", " ", text)  # Replace tabs/form feeds with spaces
    text = re.sub(r" +", " ", text)  # Collapse multiple spaces
    text = re.sub(r" *\n *", "\n", text)  # Clean up spaces around newlines
    return text.strip()


def extract_text(file_path: str) -> str:
    """Extract and normalize text from PDF or DOCX files."""
    file_type = _detect_file_type(file_path)
    if file_type == "pdf":
        return _extract_text_pdf(file_path)
    elif file_type == "docx":
        return _extract_text_docx(file_path)
    raise AssertionError("unreachable")


def split_text(text: str, strategy: str) -> List[str]:
    """Split text into chunks using the specified strategy (fixed/sentence/paragraph)."""
    strategy = strategy.lower()

    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]

    if strategy == "sentence":
        SENTENCE_CHUNK_SIZE = 100
        SENTENCE_CHUNK_OVERLAP = 25
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SENTENCE_CHUNK_SIZE,
            chunk_overlap=SENTENCE_CHUNK_OVERLAP,
            separators=[". ", "! ", "? ", "\n"]
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    if strategy == "paragraph":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def run_indexing(file_path: str, strategy: str) -> None:
    """Main pipeline to extract, chunk, embed, and store document content."""
    start_time = time.time()
    file_name = os.path.basename(file_path)
    logging.info("Reading: %s", file_name)
    raw_text = extract_text(file_path)
    if not raw_text:
        logging.warning("No text extracted from %s", file_name)
        return

    logging.info("Splitting text using strategy=%s", strategy)
    chunks = split_text(raw_text, strategy)
    if not chunks:
        logging.warning("No chunks produced; nothing to index.")
        return

    logging.info("Embedding %d chunks with Gemini", len(chunks))
    embedding_vectors = embed_chunks(chunks)
    if len(embedding_vectors) != len(chunks):
        chunk_embedding_pairs = [(chunk, vector) for chunk, vector in zip(chunks, embedding_vectors)]  # Handle skipped chunks
    else:
        chunk_embedding_pairs = list(zip(chunks, embedding_vectors))

    db_connection = get_db_connection()
    try:
        ensure_schema(db_connection)
        inserted_count = insert_chunks(db_connection, file_name, strategy, chunk_embedding_pairs)
    finally:
        db_connection.close()

    elapsed_time = time.time() - start_time
    logging.info("✅ Indexed %d chunks from %s (strategy=%s) in %.2fs", inserted_count, file_name, strategy, elapsed_time)


def setup_logger(level: str = "INFO") -> None:
    """Configure logging with specified level and format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for file path and chunking strategy."""
    argument_parser = argparse.ArgumentParser(description="Index a document into pgvector (single-table)")
    argument_parser.add_argument("--file", required=True, help="Path to .pdf or .docx file")
    argument_parser.add_argument("--strategy", required=True, choices=["fixed", "sentence", "paragraph"], help="Chunking strategy")
    return argument_parser.parse_args(argv)


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    try:
        run_indexing(args.file, args.strategy)
    except Exception as e:
        logging.exception("Indexing failed: %s", e)
        sys.exit(1)