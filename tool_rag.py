# Please install chromadb first: `pip install chromadb sentence-transformers`
import os
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")

default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=default_ef
)

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def load_documents(folder_path):
    if not os.path.exists(folder_path):
        print(f"folder {folder_path} not exist")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            chunks = chunk_text(content)
            for id, chunk in enumerate(chunks):
                doc_id = f"{filename}_{id}"
                collection.upsert(
                    documents=[chunk],
                    ids=[doc_id],
                    metadatas=[{"source": filename}]
                )

def search_documents(query, n_results=2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    context = "\n".join(results['documents'][0])
    return context

def search_docs(query):
    return search_documents(query)

Knowledge_FOLDER = "knowledge"

if __name__ == "__main__":
    load_documents(Knowledge_FOLDER)