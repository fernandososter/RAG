import chromadb
from chromadb.config import Settings

class ChromaClient:
    def __init__(self, collection_name="meu_rag"):
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb", persist_directory="chroma_db"))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, documents, metadatas, ids):
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_text, n_results=3):
        return self.collection.query(query_texts=[query_text], n_results=n_results)
