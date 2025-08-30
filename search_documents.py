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

from dotenv import load_dotenv

from gemini_handling import embed_query
from db_handling import get_db_connection, search_similar_chunks, TOP_K

load_dotenv()

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
            conn = get_db_connection()
            try:
                rows = search_similar_chunks(conn, qvec, TOP_K)
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


def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logger()
    interactive_loop()


if __name__ == "__main__":
    main()