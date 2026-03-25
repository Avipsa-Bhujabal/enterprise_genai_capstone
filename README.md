# Enterprise Document Q&A with Agentic RAG

A capstone-ready Generative AI application that lets users upload enterprise documents and ask natural-language questions. The system ingests PDF, TXT, CSV, and Excel files, chunks content for semantic search, stores embeddings in a vector database, retrieves relevant context, and uses an LLM to generate grounded answers.

The project also includes a lightweight agentic workflow with four stages:
- **Planner agent**: interprets the question and creates a retrieval plan
- **Retriever agent**: fetches the most relevant chunks from the vector store
- **Reasoner agent**: drafts an answer grounded in retrieved evidence
- **Validator agent**: checks the draft for missing citations, unsupported claims, and empty-context cases

## Architecture

```text
Streamlit UI / REST Client
        |
        v
     FastAPI API
        |
        +--> Document Loader (PDF/TXT/CSV/XLSX)
        +--> Chunker
        +--> Embedding Model
        +--> Chroma Vector Store
        +--> Agentic RAG Orchestrator
                |- Planner
                |- Retriever
                |- Reasoner
                |- Validator
```

## Features
- Upload PDF, TXT, CSV, XLSX, XLS files
- Parse structured and unstructured documents
- Chunk documents with metadata
- Create embeddings and persist them in Chroma
- Similarity-based retrieval for semantic search
- LLM-based grounded answering with citations
- Agent-style reasoning flow for task planning and validation
- Safety controls for empty retrieval, unsupported file types, and prompt injection cues
- Streamlit UI for quick demos
- Docker support for local deployment

## Tech Stack
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Vector DB:** Chroma
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM:** OpenAI-compatible chat model via the `openai` Python SDK
- **Parsing:** PyPDF, pandas, openpyxl

## Project Structure

```text
enterprise_genai_capstone/
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes.py
│       ├── core/
│       │   ├── config.py
│       │   └── logging.py
│       ├── models/
│       │   └── schemas.py
│       ├── services/
│       │   ├── agents.py
│       │   ├── chunking.py
│       │   ├── ingestion.py
│       │   ├── llm.py
│       │   ├── rag.py
│       │   └── vector_store.py
│       └── main.py
├── frontend/
│   └── streamlit_app.py
├── data/
│   ├── chroma_db/
│   └── uploads/
├── tests/
│   └── test_smoke.py
├── .env.example
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── requirements.txt
└── README.md
```

## Setup

### 1) Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Copy `.env.example` to `.env` and fill in your values.

```bash
cp .env.example .env
```

Required:
- `OPENAI_API_KEY`

Optional:
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` for OpenAI-compatible providers
- `CHROMA_PERSIST_DIR`
- `TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

## Run Locally

### Start the API
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the UI
```bash
streamlit run frontend/streamlit_app.py
```

UI default URL:
- `http://localhost:8501`

API docs:
- `http://localhost:8000/docs`

## API Endpoints

### Health check
```http
GET /health
```

### Ingest documents
```http
POST /api/v1/ingest
Content-Type: multipart/form-data
files: [file1, file2, ...]
```

### Ask a question
```http
POST /api/v1/ask
{
  "question": "What does the policy say about remote work approvals?",
  "top_k": 4
}
```

## Reliability & Safety Controls
- Reject unsupported file types
- Handle empty files and empty retrieval results gracefully
- Add provenance metadata to every chunk
- Instruct the LLM to answer only from retrieved context
- Return `I don't know` style responses when evidence is insufficient
- Basic prompt-injection defense by isolating system instructions from user content

## Deployment

### Docker Compose
```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Limitations
- Baseline version uses local semantic search, not hybrid reranking
- Agent workflow is intentionally lightweight for capstone clarity
- OCR is not included for scanned PDFs
- Fine-grained auth and multi-tenant isolation are not included in this starter version

## Suggested Enhancements
- Add authentication and per-user document spaces
- Add hybrid retrieval with BM25 + vector search
- Add reranking
- Add evaluation dataset and RAG metrics
- Add LangGraph for richer agent state transitions
- Add citations with page-level references for PDFs
- Add guardrails library integration and moderation

## Resume-Friendly Summary
Built an agentic RAG application using FastAPI, Streamlit, Chroma, SentenceTransformers, and OpenAI APIs to ingest enterprise documents, retrieve relevant context, and generate grounded answers with autonomous planning, retrieval, reasoning, and validation steps.
