"""
search_documents.py — Interactive cosine search over pgvector

Goal
----
The script will prompt the user for a query, embed it
with Gemini (`text-embedding-004`), and return the top-5 most similar
`chunk_text` rows from the `indexed_chunks` table using cosine similarity.

Usage
-----
python search_documents.py
# then type your question at the prompt (blank line to exit)

"""
import logging
import os
import sys
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    raise RuntimeError("dotenv is not installed. `pip install python-dotenv`")

EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768
TOP_K = 5

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Setup logger

def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Database connection

try:
    import psycopg
    from pgvector.psycopg import register_vector, Vector
except Exception as e:  # pragma: no cover
    print("ERROR: psycopg and pgvector are required. `pip install psycopg pgvector`", file=sys.stderr)
    raise


def get_db_conn():
    """Open a new psycopg connection and register pgvector."""
    conn = psycopg.connect(POSTGRES_URL)
    register_vector(conn)
    return conn


# Embeddings using Gemini API with retry logic

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

try:
    from tenacity import retry, wait_exponential_jitter, stop_after_attempt
except Exception:
    def retry(*args, **kwargs):
        def deco(fn):
            return fn
        return deco
    def wait_exponential_jitter(*args, **kwargs):
        return None
    def stop_after_attempt(*args, **kwargs):
        return None


def _require_gemini():
    if not _HAS_GEMINI:
        raise RuntimeError("google-generativeai is not installed. `pip install google-generativeai`")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")


def _configure_gemini():
    _require_gemini()
    genai.configure(api_key=GEMINI_API_KEY)


def embed_query(text: str) -> List[float]:
    """Generate embedding for search query using Gemini API."""
    _configure_gemini()
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    vec = getattr(resp, "embedding", None) or resp.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Unexpected embedding response format from Gemini")
    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(vec)}")
    return [float(x) for x in vec]


# Search using cosine similarity

def search_top_k(conn, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Find top-k most similar chunks using cosine similarity, returns list of dicts with chunk data and scores."""
    sql_query = (
        "SELECT id, file_name, split_strategy, created_at, chunk_text, "
        "1 - (embedding <=> %s) AS similarity "
        "FROM indexed_chunks "
        "ORDER BY embedding <=> %s "
        "LIMIT %s"
    )
    params = [Vector(query_embedding), Vector(query_embedding), top_k]

    results: List[Dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(sql_query, params)
        for row in cur.fetchall():
            rid, fname, strat, created_at, text, sim = row
            results.append(
                {
                    "id": rid,
                    "file_name": fname,
                    "split_strategy": strat,
                    "created_at": created_at,
                    "similarity": float(sim),
                    "chunk_text": text,
                }
            )
    return results

# Main loop for interactive search

def interactive_loop() -> None:
    """Main interactive search loop - prompts for queries and displays results."""
    print("\nType your question and press Enter. Empty input exits.\n")
    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye for now.")
            return
        if not q:
            print("Bye for now.")
            return
        try:
            qvec = embed_query(q)  # Convert query to embedding
            conn = get_db_conn()
            try:
                rows = search_top_k(conn, qvec, TOP_K)
            finally:
                conn.close()
            # Display search results
            print("\nTop results:")
            for i, r in enumerate(rows, 1):
                print(f"\n#{i} | sim={r['similarity']:.3f} | file={r['file_name']} | strategy={r['split_strategy']}")
                print(r["chunk_text"])
            print("\n— End of results —\n")
        except Exception as e:
            logging.exception("Search failed: %s", e)
            print("(An error occurred; see logs.)")


# Main

def main() -> None:
    setup_logger()
    interactive_loop()


if __name__ == "__main__":
    main()
