import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Split source/title from content
                lines = content.split("\n")
                source = lines[0].replace("Source: ", "").strip()
                title = lines[1].replace("Title: ", "").strip()
                body = "\n".join(lines[3:])
                
                doc = Document(
                    page_content=body,
                    metadata={"source": source, "title": title}
                )
                documents.append(doc)
    return documents

def get_chunks(documents):
    # Using RecursiveCharacterTextSplitter which is generally preferred for docs
    # Chunk size and overlap are key hyperparameters to justify in DECISIONS.md
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    docs = load_documents("data/raw_docs")
    chunks = get_chunks(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents.")
