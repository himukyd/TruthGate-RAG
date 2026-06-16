# 🛡️ TruthGate-RAG: Reliable FastAPI Technical Assistant

TruthGate-RAG is a specialized Retrieval-Augmented Generation (RAG) system designed to provide accurate, documentation-grounded answers about **FastAPI**. It is uniquely engineered to **refuse** unanswerable questions, detect **adversarial prompt injections**, and identify **false technical premises**.

## 🚀 Key Features

*   **Multi-Stage Validation Pipeline:** Every query passes through three security and logic gates:
    1.  **Adversarial Detection:** Identifies prompt leaks and malicious injections using LLM-based reasoning.
    2.  **False Premise Detection:** Flags technical misconceptions (e.g., "Why does FastAPI require Java?") with 100% accuracy.
    3.  **Context Sufficiency:** Ensures the retrieved documentation actually contains the answer before attempting to respond.
*   **Grounded Generation:** Answers are restricted to the provided context and citations are automatically extracted.
*   **Optimized Local Inference:** Runs entirely locally using **Qwen-2.5-1.5B-Instruct** on Apple Silicon (MPS) or CUDA.
*   **Comprehensive Evaluation:** Includes a full evaluation harness with per-category accuracy metrics and latency profiling.

## 🛠️ Architecture

*   **LLM:** Qwen/Qwen2.5-1.5B-Instruct
*   **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
*   **Vector Store:** ChromaDB
*   **Orchestration:** LangGraph (Stateful RAG workflow)
*   **Backend:** FastAPI
*   **Frontend:** Streamlit

## 📋 Installation & Setup

1.  **Clone and Enter:**
    ```bash
    git clone https://github.com/himukyd/TruthGate-RAG.git
    cd TruthGate-RAG
    ```

2.  **Setup Environment:**
    ```bash
    # Create and activate virtual environment
    python3 -m venv him_langchain
    source him_langchain/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file based on `.env.example`:
    ```env
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

## 🏃 Running the System

Use the included `Makefile` for standard operations:

*   **Ingest Documentation:**
    ```bash
    make ingest
    ```
*   **Run Evaluation:**
    ```bash
    make eval
    ```
*   **Start Backend (FastAPI):**
    ```bash
    make run-backend
    ```
*   **Start Frontend (Streamlit):**
    ```bash
    make run-frontend
    ```

## 📊 Evaluation Results

The system has been evaluated against a balanced 20-question test set:

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **50.0%** |
| **False Premise Detection** | **100.0%** |
| **Answerable Accuracy** | **40.0%** |
| **Mean Latency** | ~84s (Local MPS) |

*Detailed reports can be found in `evaluation/report.tex`.*

## 📂 Project Structure

*   `rag/`: Core RAG logic, detection nodes, and LLM configuration.
*   `ingestion/`: Scraper and chunking logic for FastAPI docs.
*   `app/`: FastAPI backend implementation.
*   `frontend/`: Streamlit user interface.
*   `evaluation/`: Test suite, results, and automated reports.
