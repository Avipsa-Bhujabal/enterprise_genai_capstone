# 🧠 Enterprise Document Q&A using Gemini + Agentic RAG

## 🚀 Overview
This project is a **Generative AI-powered knowledge and decision support system** that allows users to query enterprise documents using natural language.

It combines:
- **Google Gemini (LLMs + Embeddings)**
- **Retrieval-Augmented Generation (RAG)**
- **Agentic AI Workflow (Planner → Retriever → Reasoner → Verifier)**

The system enables users to upload documents (PDF, TXT, CSV, Excel) and get **accurate, grounded, and context-aware answers**.

---

## ✨ Features
- 📄 Multi-format document ingestion (PDF, TXT, CSV, Excel)
- 🔍 Semantic search using embeddings
- 🧩 Intelligent chunking for better retrieval
- 🤖 Agent-based reasoning pipeline:
  - Planner Agent
  - Retriever Agent
  - Reasoning Agent
  - Verifier Agent
- 📊 Streamlit UI for interaction
- 📚 Source-grounded answers with citations

---

## 🏗️ Architecture

```
User Query
   ↓
Planner Agent → Defines intent
   ↓
Retriever Agent → Fetches relevant chunks
   ↓
Reasoning Agent → Generates grounded answer
   ↓
Verifier Agent → Validates and refines output
   ↓
Final Answer (with citations)
```

---

## ⚙️ Tech Stack
- **LLM & Embeddings:** Google Gemini
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Document Parsing:** PyPDF, CSV, Excel
- **Environment Management:** python-dotenv

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install streamlit google-genai pypdf pandas openpyxl numpy python-dotenv
```

### 3. Setup environment variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application
```bash
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

---

## 🧪 How It Works

### 1. Upload Documents
Users upload enterprise files (PDF, CSV, Excel, TXT).

### 2. Document Processing
- Extract text
- Clean and chunk data
- Generate embeddings using Gemini

### 3. Query Processing
- User asks a question
- Planner agent determines intent
- Retriever finds relevant chunks

### 4. Response Generation
- LLM generates answer using retrieved context
- Verifier ensures accuracy and grounding

---

## 📌 Example Use Cases
- 📊 Business policy Q&A
- 📄 Contract analysis
- 📈 Financial document insights
- 🏢 Enterprise knowledge assistant

---

## ⚠️ Limitations
- In-memory vector store (not persistent)
- Performance depends on document size
- Limited context window for large documents
- Requires API key for Gemini

---

## 🔮 Future Improvements
- Add persistent vector database (Chroma / FAISS)
- Implement chat memory
- Add authentication (OTP / OAuth)
- Deploy using Docker or cloud (AWS/GCP)
- Add evaluation metrics for RAG quality

---

## 💼 Resume Description (ATS Ready)

Developed a Generative AI application using Google Gemini, RAG, and agent-based architecture to enable semantic search and intelligent Q&A over enterprise documents, improving information retrieval accuracy and decision support.

---

## 🙌 Author
Built as a capstone project for demonstrating **Generative AI, RAG, and Agentic AI systems**.

---

## ⭐ If you like this project
Give it a star ⭐ and use it in your portfolio!

