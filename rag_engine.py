# Please install chromadb first: `pip install chromadb sentence-transformers`
import chromadb
from chromadb.utils import embedding_functions

DBPath = "./chroma_db"

class RAGEngine:
    def __init__(self, path=DBPath):
        self.client = chromadb.PersistentClient(path=path)
        
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="knowledge",
            embedding_function=self.ef
        )
    
    def add_doc(self, content: str, doc_id: str):
        self.collection.add(
            documents=[content],
            ids=[doc_id],
        )

    def query(self, text: str):
        results = self.collection.query(
            query_texts=[text],
            n_results=2
        )
        context = "\n".join(results['documents'][0])
        return context

rag_instance = RAGEngine()
