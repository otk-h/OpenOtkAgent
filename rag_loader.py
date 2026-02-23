import os
from rag_engine import rag_instance

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def load_documents(folder_path):
    print(f"Loading documents from folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"folder {folder_path} not exist")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"Processing file: {filename}")
            content = file.read()
            chunks = chunk_text(content)
            for id, chunk in enumerate(chunks):
                doc_id = f"{filename}_{id}"
                rag_instance.add_doc(chunk, doc_id)
                print(f"Added document {doc_id} to RAG database.")

KNOWLEDGE_FOLDER = "knowledge"

if __name__ == "__main__":
    load_documents(KNOWLEDGE_FOLDER)
