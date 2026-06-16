import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        # Using a small, local embedding model for faster vector creation.
        # You can override this with EMBEDDINGS_MODEL_NAME in .env.
        model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
        _embeddings_instance = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings_instance

def get_vectorstore():
    embeddings = get_embeddings()
    persist_directory = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="fastapi_docs_local"
    )
    return vectorstore

def get_retriever(search_kwargs=None):
    if search_kwargs is None:
        search_kwargs = {"k": int(os.getenv("RETRIEVER_K", "3"))}
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
