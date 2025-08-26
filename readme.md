# RAG Indexing Setup Guide

This guide explains how to **run `index_documents.py` end-to-end**: installing dependencies, setting up PostgreSQL with `pgvector` (via Docker), configuring environment variables, and executing the script.

---

## 🐘 PostgreSQL via Docker

We’ll run PostgreSQL with the `pgvector` extension preinstalled using Docker.

### Step 1: Install Docker locally (if not already installed)

### Step 2: Run the container
Execute this in **Terminal (macOS)** or **PowerShell (Windows)**:

```bash
docker run --name rag-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  -v rag_pg_data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```

- Exposes Postgres on `localhost:5432`
- User: `postgres`
- Password: `postgres`
- Database: `ragdb`
- Volume: `rag_pg_data` (persistent storage)


### Step 3: Verify DB is running (optional)
```bash
docker exec -it rag-pg psql -U postgres -d ragdb -c "SELECT version();"
```

### Step 4: Enable the vector extension
Once connected to the database, enable the pgvector extension (one-time setup):

```bash
docker exec -it rag-pg psql -U postgres -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> This creates the `vector` type inside your database so embeddings can be stored.

---

## 🔑 Environment Variables

### macOS/Linux (zsh/bash):
```bash
export POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/ragdb"
export GEMINI_API_KEY="<your_gemini_api_key_here>"
```

### Windows (PowerShell):
```powershell
$env:POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/ragdb"
$env:GEMINI_API_KEY = "<your_gemini_api_key_here>"
```

### Or just create a `.env` file in the root directory with the following content:

```bash
POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/ragdb"
GEMINI_API_KEY="<your_gemini_api_key_here>"
```

---

## ▶️ Run the Script

create a virtual environment and activate it:

```bash
conda create -n rag
conda activate rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

```bash
python index_documents.py --file "/path/to/your/document.pdf" --strategy fixed
```

Valid strategies:
- `fixed`
- `sentence`
- `paragraph`

Example:
```bash
python index_documents.py --file "/test.docx" --strategy sentence
```

Now that we have the Database ready, we can run the search script:

```bash
python search_documents.py
```
---