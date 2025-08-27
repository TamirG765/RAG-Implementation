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

try:
    from dotenv import load_dotenv 
    load_dotenv()
except Exception:
    raise RuntimeError("dotenv is not installed. `pip install python-dotenv`")

CHUNK_SIZE = 800                            # characters per chunk for packing
CHUNK_OVERLAP = 200                         # overlap in characters between adjacent chunks
EMBED_MODEL = "models/text-embedding-004"   # Gemini embedding model
EMBED_DIM = 768
BATCH_SIZE = 64
EMBED_SLEEP = 7

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Setup logger

def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Init database connection and schema

try:
    import psycopg
    from psycopg import sql
    from pgvector.psycopg import register_vector  
except Exception as e:  # pragma: no cover
    print("ERROR: psycopg and pgvector are required. `pip install psycopg pgvector`", file=sys.stderr)
    raise


def get_db_conn():
    """Open a new psycopg connection and register pgvector."""
    conn = psycopg.connect(POSTGRES_URL)
    register_vector(conn)
    return conn


def ensure_schema(conn) -> None:
    """Create the single table if it doesn't exist. Sets VECTOR(EMBED_DIM)."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS indexed_chunks (
                id BIGSERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding VECTOR({EMBED_DIM}),
                file_name TEXT NOT NULL,
                split_strategy TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_indexed_chunks_strategy
            ON indexed_chunks (split_strategy, created_at);
            """
        )
    conn.commit()


def insert_rows(conn, file_name: str, strategy: str, rows: List[tuple[str, list[float]]]) -> int:
    """Bulk insert chunk rows. Each row is (chunk_text, embedding). Returns number inserted."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        query = sql.SQL(
            "INSERT INTO indexed_chunks (chunk_text, embedding, file_name, split_strategy) "
            "VALUES (%s, %s, %s, %s)"
        )
        data = [(chunk_text, embedding, file_name, strategy) for (chunk_text, embedding) in rows]
        cur.executemany(query.as_string(cur), data)
    conn.commit()
    return len(rows)


# Reading files & normalizing text

# PDF & DOCX readers
try:
    from pypdf import PdfReader
except Exception as e:
    PdfReader = None

try:
    from docx import Document
except Exception as e:
    Document = None


def detect_file_type(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return "pdf"
    if path_lower.endswith(".docx"):
        return "docx"
    raise ValueError("Unsupported file type. Use .pdf or .docx")


def extract_text_pdf(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. `pip install pypdf`")
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
    if Document is None:
        raise RuntimeError("python-docx is not installed. `pip install python-docx`")
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    text = "\n".join(paras)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    # Collapse space but preserve paragraph breaks
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\x0b\x0c]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def extract_text(file_path: str) -> str:
    ftype = detect_file_type(file_path)
    if ftype == "pdf":
        return extract_text_pdf(file_path)
    elif ftype == "docx":
        return extract_text_docx(file_path)
    raise AssertionError("unreachable")

# LangChain Text Splitters (fixed | sentence | paragraph)
try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    _HAS_LANGCHAIN = True
except Exception:
    raise RuntimeError("langchain-text-splitters is not installed. `pip install langchain-text-splitters`")


def split_text(text: str, strategy: str) -> List[str]:
    """Returns a list of chunk strings according to strategy.
    Prefers LangChain splitters when available; otherwise falls back to local implementations.
    """
    strategy = strategy.lower()

    if _HAS_LANGCHAIN:
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

# Embeddings chunks using Gemini API

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

try:
    from tenacity import retry, wait_exponential_jitter, stop_after_attempt
except Exception:
    # Provide light fallback if tenacity isn't available
    def retry(*args, **kwargs):
        return lambda fn: fn
    wait_exponential_jitter = stop_after_attempt = lambda *args, **kwargs: None


def _require_gemini():
    if not _HAS_GEMINI:
        raise RuntimeError(
            "google-generativeai is not installed. `pip install google-generativeai`"
        )
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export GEMINI_API_KEY or place it in a .env file."
        )


def _configure_gemini():
    _require_gemini()
    genai.configure(api_key=GEMINI_API_KEY)


@retry(wait=wait_exponential_jitter(1, 4, 0.1), stop=stop_after_attempt(5))
def _embed_one(text: str) -> List[float]:
    """Embed a single chunk with retries. Returns a list of floats."""
    _configure_gemini()
    # The SDK typically expects model names prefixed with "models/"
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    # Some SDK versions return dict; others return object with .embedding
    vec = getattr(resp, "embedding", None) or resp.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Unexpected embedding response format from Gemini")
    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(vec)}")
    return [float(x) for x in vec]


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embed each chunk sequentially with retries. Keeps it simple and robust."""
    embeddings: List[List[float]] = []
    for idx, ch in enumerate(chunks, 1):
        ch = ch.strip()
        if not ch:
            # Keep alignment: still push an empty vector (or skip). We'll skip.
            continue
        vec = _embed_one(ch)
        embeddings.append(vec)
        time.sleep(EMBED_SLEEP)
        if idx % 10 == 0:
            logging.info("Embedded %d/%d chunks", idx, len(chunks))
    return embeddings


# Pipeline with logging

def run_indexing(file_path: str, strategy: str) -> None:
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

    logging.info("Embedding %d chunks with Gemini (%s)", len(chunks), EMBED_MODEL)
    vectors = embed_chunks(chunks)
    if len(vectors) != len(chunks):
        # In case of skips, keep only aligned pairs
        pairs = [(c, v) for c, v in zip(chunks, vectors)]
    else:
        pairs = list(zip(chunks, vectors))

    conn = get_db_conn()
    try:
        ensure_schema(conn)
        inserted = insert_rows(conn, file_name, strategy, pairs)
    finally:
        conn.close()

    elapsed = time.time() - start
    logging.info("✅ Indexed %d chunks from %s (strategy=%s) in %.2fs", inserted, file_name, strategy, elapsed)


# CLI args

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
