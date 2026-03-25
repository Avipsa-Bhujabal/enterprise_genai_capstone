import os
import io
import json
import uuid
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ============================================================
# Single-File Gemini RAG + Agentic Document Q&A App
# ------------------------------------------------------------
# Features:
# - Upload PDF, TXT, CSV, XLSX
# - Chunk documents
# - Create embeddings with Gemini
# - Store vectors in memory for semantic search
# - Agent-style workflow: planner -> retriever -> reasoner -> verifier
# - Streamlit UI
#
# Run:
#   pip install streamlit google-genai pypdf pandas openpyxl numpy python-dotenv
#   set GEMINI_API_KEY=your_key_here   # Windows PowerShell: $env:GEMINI_API_KEY="..."
#   streamlit run app.py
# ============================================================

APP_TITLE = "Enterprise Document Q&A with Gemini + Agentic RAG"
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_CHARS_FOR_CONTEXT = 12000


# -----------------------------
# Data Models
# -----------------------------
@dataclass
class ChunkRecord:
    chunk_id: str
    source_name: str
    page_or_sheet: str
    chunk_index: int
    text: str
    embedding: List[float]


# -----------------------------
# Gemini Client
# -----------------------------
def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


# -----------------------------
# File Parsers
# -----------------------------
def read_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "source_name": filename,
                "page_or_sheet": f"Page {i}",
                "text": text.strip(),
            })
    return pages


def read_txt(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"source_name": filename, "page_or_sheet": "Text", "text": text}]


def read_csv(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    text = df.to_csv(index=False)
    return [{"source_name": filename, "page_or_sheet": "CSV", "text": text}]


def read_excel(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    workbook = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    docs = []
    for sheet_name, df in workbook.items():
        text = df.to_csv(index=False)
        docs.append({
            "source_name": filename,
            "page_or_sheet": f"Sheet: {sheet_name}",
            "text": text,
        })
    return docs


def parse_uploaded_file(uploaded_file) -> List[Dict[str, str]]:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return read_pdf(file_bytes, filename)
    if lower.endswith(".txt"):
        return read_txt(file_bytes, filename)
    if lower.endswith(".csv"):
        return read_csv(file_bytes, filename)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return read_excel(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {filename}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if end < n:
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind(" "))
            if last_break > int(chunk_size * 0.6):
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# -----------------------------
# Embeddings + Vector Search
# -----------------------------
def embed_texts(client: genai.Client, texts: List[str]) -> List[List[float]]:
    vectors = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        vectors.append(response.embeddings[0].values)
    return vectors



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)



def search_similar_chunks(client: genai.Client, query: str, records: List[ChunkRecord], top_k: int = TOP_K):
    if not records:
        return []

    query_vec = np.array(embed_texts(client, [query])[0], dtype=np.float32)
    scored = []
    for record in records:
        chunk_vec = np.array(record.embedding, dtype=np.float32)
        score = cosine_similarity(query_vec, chunk_vec)
        scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# -----------------------------
# Agentic Workflow
# -----------------------------
def planner_agent(question: str) -> Dict[str, Any]:
    q = question.lower()
    intent = "factual_qa"
    if any(word in q for word in ["compare", "difference", "vs", "versus"]):
        intent = "comparison"
    elif any(word in q for word in ["summarize", "summary", "overview"]):
        intent = "summarization"
    elif any(word in q for word in ["decision", "recommend", "should"]):
        intent = "decision_support"

    return {
        "intent": intent,
        "needs_retrieval": True,
        "top_k": TOP_K,
        "answer_style": "grounded and concise",
    }



def retriever_agent(client: genai.Client, question: str, records: List[ChunkRecord], top_k: int):
    hits = search_similar_chunks(client, question, records, top_k=top_k)
    context_items = []
    for score, record in hits:
        context_items.append({
            "score": round(score, 4),
            "source_name": record.source_name,
            "page_or_sheet": record.page_or_sheet,
            "chunk_index": record.chunk_index,
            "text": record.text,
        })
    return context_items



def build_context(context_items: List[Dict[str, Any]]) -> str:
    blocks = []
    total_chars = 0
    for item in context_items:
        block = (
            f"[Source: {item['source_name']} | {item['page_or_sheet']} | Chunk {item['chunk_index']} | "
            f"Score: {item['score']}]\n{item['text']}"
        )
        if total_chars + len(block) > MAX_CHARS_FOR_CONTEXT:
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n".join(blocks)



def reasoning_agent(client: genai.Client, question: str, plan: Dict[str, Any], context_items: List[Dict[str, Any]]) -> str:
    context = build_context(context_items)

    prompt = f"""
You are a grounded enterprise knowledge assistant.

Task plan:
{json.dumps(plan, indent=2)}

User question:
{question}

Retrieved context:
{context}

Instructions:
1. Answer ONLY from the retrieved context.
2. If the context is insufficient, say exactly what is missing.
3. Cite supporting sources inline using this format:
   [filename | page/sheet | chunk #]
4. Be accurate, concise, and business-friendly.
5. Do not invent policies, numbers, or facts.
""".strip()

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
    )
    return response.text



def verifier_agent(client: genai.Client, question: str, draft_answer: str, context_items: List[Dict[str, Any]]) -> str:
    context = build_context(context_items)
    prompt = f"""
You are a response verifier.

Question:
{question}

Draft answer:
{draft_answer}

Grounding context:
{context}

Check the answer for:
- unsupported claims
- overstatements
- missing uncertainty
- missing citations

Return a final improved answer only.
If evidence is weak, explicitly say the answer is based on limited retrieved context.
""".strip()

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
    )
    return response.text


# -----------------------------
# Index Builder
# -----------------------------
def build_index(client: genai.Client, parsed_docs: List[Dict[str, str]]) -> List[ChunkRecord]:
    all_records: List[ChunkRecord] = []

    texts_to_embed = []
    metadata = []
    for doc in parsed_docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            texts_to_embed.append(chunk)
            metadata.append((doc["source_name"], doc["page_or_sheet"], idx, chunk))

    if not texts_to_embed:
        return []

    embeddings = embed_texts(client, texts_to_embed)
    for emb, meta in zip(embeddings, metadata):
        source_name, page_or_sheet, idx, chunk = meta
        all_records.append(
            ChunkRecord(
                chunk_id=str(uuid.uuid4()),
                source_name=source_name,
                page_or_sheet=page_or_sheet,
                chunk_index=idx,
                text=chunk,
                embedding=emb,
            )
        )
    return all_records


# -----------------------------
# Streamlit App
# -----------------------------
def init_state():
    if "records" not in st.session_state:
        st.session_state.records = []
    if "files_indexed" not in st.session_state:
        st.session_state.files_indexed = []
    if "history" not in st.session_state:
        st.session_state.history = []



def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Single-file capstone app using Gemini embeddings + Gemini generation + agent-style RAG workflow")

    with st.sidebar:
        st.header("Setup")
        st.write("1. Add your Gemini API key in a .env file")
        st.code('GEMINI_API_KEY=your_key_here')
        st.write("2. Upload one or more files")
        st.write("3. Click **Process Documents**")
        st.write("4. Ask grounded questions")

        uploaded_files = st.file_uploader(
            "Upload enterprise documents",
            type=["pdf", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        process_clicked = st.button("Process Documents", use_container_width=True)
        clear_clicked = st.button("Clear Session", use_container_width=True)

        if clear_clicked:
            st.session_state.records = []
            st.session_state.files_indexed = []
            st.session_state.history = []
            st.success("Session cleared.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Indexed Documents")
        if st.session_state.files_indexed:
            for name in st.session_state.files_indexed:
                st.write(f"- {name}")
            st.info(f"Total chunks indexed: {len(st.session_state.records)}")
        else:
            st.write("No documents indexed yet.")

    with col2:
        st.subheader("How this works")
        st.markdown(
            """
- **Planner agent** determines the query intent.
- **Retriever agent** finds the most relevant chunks.
- **Reasoning agent** drafts a grounded answer.
- **Verifier agent** checks for unsupported claims.
            """
        )

    if process_clicked:
        try:
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                client = get_client()
                parsed_docs = []
                for file in uploaded_files:
                    parsed_docs.extend(parse_uploaded_file(file))

                records = build_index(client, parsed_docs)
                st.session_state.records = records
                st.session_state.files_indexed = sorted({d["source_name"] for d in parsed_docs})
                st.success(f"Indexed {len(records)} chunks from {len(st.session_state.files_indexed)} file(s).")
        except Exception as e:
            st.error(f"Document processing failed: {e}")

    st.divider()
    st.subheader("Ask Questions")
    question = st.text_area(
        "Ask a natural language question about the uploaded documents",
        placeholder="Example: What are the refund approval rules in the policy documents?",
        height=120,
    )

    ask_clicked = st.button("Get Answer")

    if ask_clicked:
        try:
            if not question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.records:
                st.warning("Please upload and process documents first.")
            else:
                client = get_client()
                plan = planner_agent(question)
                context_items = retriever_agent(client, question, st.session_state.records, top_k=plan["top_k"])
                draft_answer = reasoning_agent(client, question, plan, context_items)
                final_answer = verifier_agent(client, question, draft_answer, context_items)

                st.markdown("### Final Answer")
                st.write(final_answer)

                with st.expander("Retrieved Evidence"):
                    for item in context_items:
                        st.markdown(
                            f"**{item['source_name']} | {item['page_or_sheet']} | Chunk {item['chunk_index']} | Similarity: {item['score']}**"
                        )
                        st.write(item["text"])
                        st.divider()

                st.session_state.history.append({
                    "question": question,
                    "answer": final_answer,
                    "plan": plan,
                    "retrieved": context_items,
                })
        except Exception as e:
            st.error(f"Answer generation failed: {e}")

    if st.session_state.history:
        st.divider()
        st.subheader("Session History")
        for i, item in enumerate(reversed(st.session_state.history), start=1):
            with st.expander(f"Q{i}: {item['question'][:120]}"):
                st.write(item["answer"])


if __name__ == "__main__":
    main()
