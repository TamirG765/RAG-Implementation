"""
gemini_handling.py — Gemini API embedding utilities

Centralized module for handling Gemini API embedding operations used across
the RAG implementation for both document indexing and query processing.
"""
import logging
import os
import time
from typing import List

import google.generativeai as genai
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768
EMBED_SLEEP = 7  # Rate limiting delay in seconds

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def validate_gemini_connection():
    """Validate that Gemini API key is available."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export GEMINI_API_KEY or place it in a .env file."
        )


def configure_gemini():
    """Configure Gemini API with the provided API key."""
    validate_gemini_connection()
    genai.configure(api_key=GEMINI_API_KEY)


@retry(wait=wait_exponential_jitter(1, 4, 0.1), stop=stop_after_attempt(5))
def embed_single_text(text: str) -> List[float]:
    """Embed a single text chunk with retries and return embedding vector."""
    configure_gemini()
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    vec = getattr(resp, "embedding", None) or resp.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Unexpected embedding response format from Gemini")
    if len(vec) != EMBED_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: expected {EMBED_DIM}, got {len(vec)}")
    return [float(x) for x in vec]


def embed_query(text: str) -> List[float]:
    """Generate embedding vector for a search query text."""
    return embed_single_text(text)


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embedding vectors for multiple text chunks with rate limiting."""
    embeddings: List[List[float]] = []
    for idx, ch in enumerate(chunks, 1):
        ch = ch.strip()
        if not ch:
            continue  # Skip empty chunks
        vec = embed_single_text(ch)
        embeddings.append(vec)
        time.sleep(EMBED_SLEEP)  # Rate limiting
        if idx % 10 == 0:
            logging.info("Embedded %d/%d chunks", idx, len(chunks))
    return embeddings