# TruthGate-RAG
A RAG system over FastAPI documentation that answers with citations, detects false-premise questions, and refuses unsupported queries.
## Architecture

Architecture Diagram

User Query
↓
Retriever
↓
Reranker
↓
TruthGate Validator
↓
Answer / Refusal / False Premise

## Tech Stack

### Frontend
- Streamlit

### Backend
- FastAPI

### LLM
- Gemini 2.5 Flash

### Embeddings
- Gemini Embedding 001

### Vector Database
- ChromaDB
