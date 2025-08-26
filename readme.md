# RAG Document Search System

A Retrieval-Augmented Generation (RAG) system that converts PDF and DOCX documents into vector embeddings and enables semantic search over the content using PostgreSQL with pgvector.

## How It Works

1. **Index Documents** (`index_documents.py`) - Extracts text from PDF/DOCX files, splits into chunks, generates embeddings using Gemini, and stores in PostgreSQL
2. **Search Documents** (`search_documents.py`) - Performs semantic similarity search using cosine distance to find relevant content

## Features

- **Document Support**: PDF and DOCX files
- **Chunking Strategies**: Fixed-size, sentence-based, or paragraph-based splitting
- **Vector Embeddings**: Gemini `text-embedding-004` model
- **Similarity Search**: Cosine similarity using PostgreSQL pgvector extension
- **Interactive Search**: Command-line interface for querying indexed documents

---

## Quick Setup

### 0. Install Docker on your machine

### 1. Database (PostgreSQL + pgvector)
```bash
# Run PostgreSQL with pgvector extension
docker run --name rag-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  -v rag_pg_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16

# Enable vector extension (one-time)
docker exec -it rag-pg psql -U postgres -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Environment Variables
Create a `.env` file:
```bash
POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/ragdb"
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Install Dependencies
```bash
# Create environment
conda create -n rag python=3.11
conda activate rag
```

```bash
# Install packages
pip install -r requirements.txt
```

## Usage

### Index Documents
```bash
# Index a PDF with sentence-based chunking
python index_documents.py --file "document.pdf" --strategy sentence

# Index a DOCX with fixed-size chunks  
python index_documents.py --file "report.docx" --strategy fixed
```

**Chunking strategies:**
- `fixed` - 800 chars with 200 char overlap using LangChain CharacterTextSplitter
- `sentence` - 100 chars with 25 char overlap using LangChain RecursiveCharacterTextSplitter with sentence separators
- `paragraph` - 800 chars with 200 char overlap using LangChain CharacterTextSplitter with paragraph separators

### Search Documents
```bash
# Start interactive search
python search_documents.py
```
Enter queries to find semantically similar content from indexed documents. Returns top-5 matches with similarity scores.