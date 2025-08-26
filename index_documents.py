"""
index_documents.py — Single-table indexer for RAG

Purpose
-------
Ingest a PDF/DOCX file, split it into chunks (one of: fixed|sentence|paragraph),
create Gemini embeddings for each chunk, and persist rows into a single
PostgreSQL table with pgvector.

Only two runtime inputs are configurable: `file_path` and `strategy`.
Everything else is fixed as constants below (DB URL, model name, chunk sizes, etc.).

Table: indexed_chunks
---------------------
- id BIGSERIAL PRIMARY KEY
- chunk_text TEXT NOT NULL
- embedding VECTOR(768)
- file_name TEXT NOT NULL
- split_strategy TEXT NOT NULL
- created_at TIMESTAMPTZ DEFAULT NOW()

Usage
-----
python index_documents.py --file ./sample.pdf --strategy sentence

Notes
-----
- `POSTGRES_URL` and `GEMINI_API_KEY` are read from environment variables (or fall back
  to placeholders defined in constants). You only pass `--file` and `--strategy`.
- The code creates the schema if it doesn't exist.
- If `nltk` isn't installed, we use a simple regex sentence splitter.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import re
import sys
import time
from typing import Iterable, List
import time

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Fixed configuration (only file_path & strategy are CLI-configurable)
CHUNK_SIZE = 800          # approximate characters per chunk for packing
CHUNK_OVERLAP = 200       # overlap in characters between adjacent chunks
EMBED_MODEL = "models/text-embedding-004"  # Gemini embedding model
EMBED_DIM = 768
BATCH_SIZE = 64
EMBED_SLEEP = 7

# Secrets / connection strings: read from env, fallback to placeholders
POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Logging -----------------------------------------------------------------

def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --- Database layer (psycopg + pgvector adapter) ------------------------------

try:
    import psycopg
    from psycopg import sql
    from pgvector.psycopg import register_vector  # type: ignore
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
        # Optional: build a vector index later when you have enough data
        # cur.execute(
        #     "CREATE INDEX IF NOT EXISTS idx_indexed_chunks_embedding "
        #     "ON indexed_chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);"
        # )
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


# --- File reading & normalization --------------------------------------------

# PDF & DOCX readers
try:
    from pypdf import PdfReader  # type: ignore
except Exception as e:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from docx import Document  # type: ignore
except Exception as e:  # pragma: no cover
    Document = None  # type: ignore


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


# --- Chunking strategies ------------------------------------------------------

@dataclasses.dataclass
class ChunkingConfig:
    chunk_size: int = CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP


def split_text(text: str, strategy: str, cfg: ChunkingConfig | None = None) -> List[str]:
    """Returns a list of chunk strings according to strategy."""
    cfg = cfg or ChunkingConfig()
    strategy = strategy.lower()
    if strategy == "fixed":
        return chunk_fixed(text, cfg.chunk_size, cfg.overlap)
    if strategy == "sentence":
        return chunk_by_sentences(text, cfg.chunk_size, cfg.overlap)
    if strategy == "paragraph":
        return chunk_by_paragraphs(text, cfg.chunk_size, cfg.overlap)
    raise ValueError("strategy must be one of: fixed | sentence | paragraph")


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-based sliding window with overlap."""
    chunks: List[str] = []
    n = len(text)
    if n == 0:
        return chunks
    i = 0
    while i < n:
        chunk = text[i : i + chunk_size]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        # Move by chunk_size - overlap, but at least 1
        step = max(1, chunk_size - overlap)
        i += step
    return chunks


# Optional: try NLTK sentence split; fallback to regex
try:
    import nltk  # type: ignore
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def _split_sentences(text: str) -> List[str]:
    if _HAS_NLTK:
        from nltk.tokenize import sent_tokenize  # type: ignore
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    # naive regex-based sentence segmentation
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def pack_units(units: List[str], target_len: int, overlap: int, sep: str = " ") -> List[str]:
    """Greedily packs units (sentences/paragraphs) into ~target_len char windows with char overlap.
    Overlap is applied on the final character stream, approximated by carrying the last `overlap`
    characters of the previous chunk as a prefix to the next one.
    """
    if not units:
        return []
    joined = []
    current = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if not current:
            return None
        chunk = sep.join(current).strip()
        current = []
        current_len = 0
        return chunk

    for u in units:
        ulen = len(u) + (0 if not current else len(sep))
        if current_len + ulen <= target_len:
            current.append(u)
            current_len += ulen
        else:
            chunk = flush()
            if chunk:
                joined.append(chunk)
            # start new with overlap prefix from previous chunk
            if joined:
                overlap_src = joined[-1][-overlap:] if overlap > 0 else ""
                current = [overlap_src, u] if overlap_src else [u]
                # compute length again
                current_len = len(overlap_src) + (len(sep) if overlap_src and u else 0) + len(u)
            else:
                current = [u]
                current_len = len(u)
    last = flush()
    if last:
        joined.append(last)
    # Final trim & filter
    return [c.strip() for c in joined if c and c.strip()]


def chunk_by_sentences(text: str, target_len: int, overlap: int) -> List[str]:
    sentences = _split_sentences(text)
    return pack_units(sentences, target_len, overlap, sep=" ")


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_paragraphs(text: str, target_len: int, overlap: int) -> List[str]:
    paras = _split_paragraphs(text)
    return pack_units(paras, target_len, overlap, sep="\n\n")


# --- Embeddings via Gemini ----------------------------------------------------

# We use google-generativeai's embed_content API. We do a simple per-chunk loop with
# retries to keep the code portable.

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

try:
    from tenacity import retry, wait_exponential_jitter, stop_after_attempt  # type: ignore
except Exception:  # pragma: no cover
    # Provide light fallback if tenacity isn't available
    def retry(*args, **kwargs):
        def deco(fn):
            return fn
        return deco
    def wait_exponential_jitter(*args, **kwargs):  # type: ignore
        return None
    def stop_after_attempt(*args, **kwargs):  # type: ignore
        return None


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
    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore


@retry(wait=wait_exponential_jitter(1, 4, 0.1), stop=stop_after_attempt(5))
def _embed_one(text: str) -> List[float]:
    """Embed a single chunk with retries. Returns a list of floats."""
    _configure_gemini()
    # The SDK typically expects model names prefixed with "models/"
    resp = genai.embed_content(model=EMBED_MODEL, content=text)  # type: ignore
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


# --- Orchestration ------------------------------------------------------------

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


# --- CLI ----------------------------------------------------------------------

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
