import time
import logging
from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse, Citation
from rag.graph import process_query
from rag.llm import get_local_llm
import threading

app = FastAPI(title="TruthGate RAG API")


@app.on_event("startup")
def preload_model():
    """Preload the local LLM in a background thread to avoid cold-start timeouts."""
    def _load():
        try:
            get_local_llm()
        except Exception as e:
            logger.error(f"Error preloading LLM: {e}")
    threading.Thread(target=_load, daemon=True).start()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        response = process_query(request.query)
        
        # Log requirements
        logger.info(f"Query: {request.query} | Status: {response.status} | Latency: {response.latency:.4f}s | Cost: ${response.cost:.6f}")
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
