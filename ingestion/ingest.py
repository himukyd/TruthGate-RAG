import os
import sys
import time
from dotenv import load_dotenv

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.scraper import main as run_scraper
from ingestion.chunker import load_documents, get_chunks
from rag.vectorstore import get_vectorstore

load_dotenv()

def main():
    # 1. Scrape
    print("--- Phase 1: Scraping ---")
    if not os.path.exists("data/raw_docs") or not os.listdir("data/raw_docs"):
        run_scraper()
    else:
        print("Data already scraped. Skipping phase 1.")

    # 2. Chunk
    print("\n--- Phase 2: Chunking ---")
    docs = load_documents("data/raw_docs")
    if len(docs) < 200:
        print(f"Warning: Only found {len(docs)} pages. The requirement is at least 200.")
    
    chunks = get_chunks(docs)
    print(f"Total chunks: {len(chunks)}")

    # 3. Vectorize
    print("\n--- Phase 3: Indexing ---")
    vectorstore = get_vectorstore()
    
    # Since we are using a local model, we don't need small batches or waits
    batch_size = 200 
    num_chunks = len(chunks)
    num_batches = (num_chunks + batch_size - 1) // batch_size
    
    for i in range(0, num_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        batch_idx = i // batch_size + 1
        
        try:
            vectorstore.add_documents(batch)
            print(f"Indexed batch {batch_idx}/{num_batches}")
        except Exception as e:
            print(f"Error indexing batch {batch_idx}: {e}")
            raise e

    print("\nIngestion complete!")

if __name__ == "__main__":
    main()
