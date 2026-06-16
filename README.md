# TruthGate RAG: FastAPI Documentation Expert

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system for the official FastAPI documentation. Optimized for accuracy, citation tracking, and false-premise detection.

## 🚀 Key Features
- **100% Offline:** Uses local embeddings and a local LLM. No API keys or external quotas required.
- **Robust RAG Logic:** Features context sufficiency grading and false premise detection.
- **Strict Citations:** Every answer includes direct links to the source documentation.
- **Fast Execution:** Optimized for Mac (Metal/MPS) and CUDA.

## 🛠️ Architecture
- **Workflow Engine:** **LangGraph** (State-based execution with conditional branching)
- **LLM:** `Qwen2.5-1.5B-Instruct` (Running locally via `transformers`)
- **Embeddings:** `all-MiniLM-L6-v2` (Running locally via `sentence-transformers`)
- **Vector Store:** ChromaDB
- **Backend:** FastAPI
- **Frontend:** Streamlit

### LangGraph Workflow
The system uses a state machine to handle complex technical queries:
1. **Detect False Premise:** Checks if the query makes factually incorrect assumptions about FastAPI.
2. **Retrieval:** Fetches relevant snippets from ChromaDB.
3. **Context Grading:** Validates if retrieved context is sufficient to answer.
4. **Generation:** Generates a precise answer with citations.

## 📦 Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd TruthGate-RAG
   ```

2. **Setup Environment:**
   This project uses the `him_langchain` environment.
   ```bash
   source him_langchain/bin/activate
   pip install -r requirements.txt
   ```

3. **Ingest Documentation:**
   ```bash
   make ingest
   ```
   *Note: This will download the LLM (~3GB) and the embedding model (~100MB) on the first run.*

4. **Run the Application:**
   ```bash
   # Terminal 1: Start Backend
   make run-backend
   
   # Terminal 2: Start Frontend
   make run-frontend
   ```

## 🧠 Model Details
- **Model Size:** ~3 GB (Qwen2.5-1.5B-Instruct)
- **RAM Requirement:** Recommended 8GB+ RAM.
- **Performance:** State-of-the-art for its size, excellent for technical Q&A.
