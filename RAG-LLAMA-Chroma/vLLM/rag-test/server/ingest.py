import os
import logging
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# embeddings (768 dims)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

chroma = Chroma(
    collection_name="rag-vllm-test1",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

def load_text_documents(folder_path: str) -> Tuple[List[str], List[dict], List[str]]:
    documents = []
    metadatas = []
    ids = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, folder_path)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    chunks = text_splitter.split_text(content)

                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "source": filename,
                            "file_path": relative_path,
                            "chunk": i,
                        })
                        ids.append(f"{relative_path}_{i}")

                    print(f"Processado: {relative_path} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"Erro ao processar {relative_path}: {e}")

    return documents, metadatas, ids


documents, metadatas, ids = load_text_documents(
    "/Users/fernando/Workspace/AI/RAG/RAG/RAG-LLAMA-Chroma/vLLM/rag-test/server/documents/"
)

if not documents:
    print("Nenhum documento foi carregado.")
else:
    batch_size = 5000
    total_documents = len(documents)

    for i in range(0, total_documents, batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        try:
            chroma.add_texts(
                texts=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            print(f"Indexados {len(batch_docs)} documentos (batch {i//batch_size + 1})")
        except Exception as e:
            print(f"Erro ao indexar batch {i//batch_size + 1}: {e}")

    print(f"Concluído! Total de {total_documents} chunks indexados.")
