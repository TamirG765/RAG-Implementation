"""
index_documents.py — Table indexer for RAG

Gaol
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

from dotenv import load_dotenv 
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from gemini_handling import embed_chunks
from db_handling import get_db_connection, ensure_schema, insert_chunks

load_dotenv()

CHUNK_SIZE = 800                            # characters per chunk for packing
CHUNK_OVERLAP = 200                         # overlap in characters between adjacent chunks


def detect_file_type(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return "pdf"
    if path_lower.endswith(".docx"):
        return "docx"
    raise ValueError("Unsupported file type. Use .pdf or .docx")


def extract_text_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    return normalize_text(text)


def extract_text_docx(path: str) -> str:
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    text = "\n".join(paras)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    """Clean up text formatting while preserving paragraph structure."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # Limit consecutive newlines
    text = re.sub(r"[\t\x0b\x0c]", " ", text)  # Replace tabs/form feeds with spaces
    text = re.sub(r" +", " ", text)  # Collapse multiple spaces
    text = re.sub(r" *\n *", "\n", text)  # Clean up spaces around newlines
    return text.strip()


def extract_text(file_path: str) -> str:
    """Extract and normalize text from PDF or DOCX files."""
    ftype = detect_file_type(file_path)
    if ftype == "pdf":
        return extract_text_pdf(file_path)
    elif ftype == "docx":
        return extract_text_docx(file_path)
    raise AssertionError("unreachable")


def split_text(text: str, strategy: str) -> List[str]:
    """Split text into chunks using LangChain splitters ('fixed', 'sentence', or 'paragraph')."""
    strategy = strategy.lower()

    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        return [c for c in splitter.split_text(text) if c.strip()]

    if strategy == "sentence":
        SENTENCE_CHUNK_SIZE = 100
        SENTENCE_CHUNK_OVERLAP = 25
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SENTENCE_CHUNK_SIZE,
            chunk_overlap=SENTENCE_CHUNK_OVERLAP,
            separators=[". ", "! ", "? ", "\n"]
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]

    if strategy == "paragraph":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        return [c for c in splitter.split_text(text) if c.strip()]


def run_indexing(file_path: str, strategy: str) -> None:
    """Main pipeline: extract text, chunk it, generate embeddings, and store in database."""
    start = time.time()
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
    vectors = embed_chunks(chunks)
    if len(vectors) != len(chunks):
        pairs = [(c, v) for c, v in zip(chunks, vectors)]  # Handle skipped chunks
    else:
        pairs = list(zip(chunks, vectors))

    conn = get_db_connection()
    try:
        ensure_schema(conn)
        inserted = insert_chunks(conn, file_name, strategy, pairs)
    finally:
        conn.close()

    elapsed = time.time() - start
    logging.info("✅ Indexed %d chunks from %s (strategy=%s) in %.2fs", inserted, file_name, strategy, elapsed)


def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index a document into pgvector (single-table)")
    p.add_argument("--file", required=True, help="Path to .pdf or .docx file")
    p.add_argument("--strategy", required=True, choices=["fixed", "sentence", "paragraph"], help="Chunking strategy")
    return p.parse_args(argv)


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    try:
        run_indexing(args.file, args.strategy)
    except Exception as e:
        logging.exception("Indexing failed: %s", e)
        sys.exit(1)