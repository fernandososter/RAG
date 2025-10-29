import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ------------------- CONFIG -------------------
DATA_DIR    = os.getenv("DATA_DIR", "data")
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
EMB_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
LLM_BASEURL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
#LLM_NAME    = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_NAME = "Qwen2.5-7B-Instruct"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# LLM via vLLM server (OpenAI-compatible)
llm = ChatOpenAI(
    model=LLM_NAME,
    base_url=LLM_BASEURL,
    api_key="dummy",
    temperature=TEMPERATURE,
)

# Embeddings + VectorStore (persistente)
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vs = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIR)
retriever = vs.as_retriever(search_type="mmr",
                            search_kwargs={"k": 4, "fetch_k": 20})

# Prompt com "system" (idioma da pergunta + grounded)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente multilíngue. Responda no idioma da pergunta. "
               "Use apenas o CONTEXTO a seguir; se não houver resposta no contexto, diga que não encontrou."),
    ("human",  "Pergunta: {question}\n\nContexto:\n{context}\n\nResposta:")
])

def join_docs(docs):
    # Inclua fontes/páginas se desejar:
    # f"{d.metadata.get('source')} p.{d.metadata.get('page')}\n{d.page_content}"
    return "\n\n---\n\n".join(d.page_content for d in docs)

# Chain LCEL server-side
rag_chain = (
    {
        "context": retriever | join_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------- FASTAPI ----------------
app = FastAPI(title="RAG Server (FastAPI + vLLM + Chroma)")

class HistoryTurn(BaseModel):
    user: str
    assistant: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[HistoryTurn]] = None
    k: Optional[int] = 4

class QueryResponse(BaseModel):
    answer: str
    # opcionalmente, adicione "sources": List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # (Opcional) condense question com histórico:
    # Aqui, simples: você pode criar um chain "condense" e plugar antes do retriever.
    # Por ora, ignoramos e usamos a pergunta direta.
    if req.k and req.k != 4:
        # reconfigurar retriever com k desejado
        local_retriever = vs.as_retriever(search_type="mmr",
                                          search_kwargs={"k": req.k, "fetch_k": max(12, req.k*3)})
        local_chain = (
            {
                "context": local_retriever | join_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = local_chain.invoke(req.question)
    else:
        answer = rag_chain.invoke(req.question)

    return QueryResponse(answer=answer)

@app.post("/ingest")
async def ingest(file: UploadFile = File(...),
                 chunk_size: int = Form(1000),
                 chunk_overlap: int = Form(150)):
    # Salva arquivo recebido
    dest_path = os.path.join(DATA_DIR, file.filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    # Carrega documento
    if file.filename.lower().endswith(".pdf"):
        docs = PyPDFLoader(dest_path).load()
    else:
        docs = TextLoader(dest_path, encoding="utf-8").load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Upsert no Chroma
    Chroma.from_documents(chunks, embedding=embeddings,
                          persist_directory=PERSIST_DIR)
    # reabrir para garantir persistência e reutilização
    global vs, retriever, rag_chain
    vs = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIR)
    retriever = vs.as_retriever(search_type="mmr",
                                search_kwargs={"k": 4, "fetch_k": 20})
    rag_chain = (
        {"context": retriever | join_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return {"status": "ingested", "file": file.filename, "chunks": len(chunks)}
