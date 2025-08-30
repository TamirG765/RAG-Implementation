"""
db_handling.py — PostgreSQL database operations with pgvector

Centralized module for database operations used across the RAG implementation
for both document indexing and search operations.
"""
import os
from typing import List, Dict, Any

import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector, Vector

from dotenv import load_dotenv

load_dotenv()

EMBED_DIM = 768
TOP_K = 5

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/ragdb")


def get_db_connection():
    """Create and return a PostgreSQL connection with pgvector support."""
    conn = psycopg.connect(POSTGRES_URL)
    register_vector(conn)
    return conn


def ensure_schema(conn) -> None:
    """Create the indexed_chunks table and database indexes if needed."""
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


def insert_chunks(conn, file_name: str, strategy: str, chunk_data: List[tuple[str, list[float]]]) -> int:
    """Bulk insert chunk data into database and return count of inserted rows."""
    if not chunk_data:
        return 0
    with conn.cursor() as cur:
        query = sql.SQL(
            "INSERT INTO indexed_chunks (chunk_text, embedding, file_name, split_strategy) "
            "VALUES (%s, %s, %s, %s)"
        )
        data = [(chunk_text, embedding, file_name, strategy) for (chunk_text, embedding) in chunk_data]
        cur.executemany(query.as_string(cur), data)
    conn.commit()
    return len(chunk_data)


def search_similar_chunks(conn, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Find top-k most similar chunks using cosine similarity search."""
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