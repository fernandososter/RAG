import os
import argparse
from typing import List, Tuple

# LangChain 0.1.x
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

DATA_DIR = os.getenv("DATA_DIR", "data")
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
#LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_MODEL = os.getenv("LLM_MODEL", "hf.co/reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B:Q8_0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

def load_documents(path: str):
    docs = []
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Pasta de dados não encontrada: {path}")
    docs += DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, show_progress=True).load()
    docs += DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True, show_progress=True).load()
    if not docs:
        print("⚠️ Nenhum documento encontrado.")
    else:
        print(f"✅ Carregados {len(docs)} documentos")
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Gerados {len(chunks)} chunks")
    return chunks

def build_or_load_vectorstore(persist_dir=PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print(f"📦 Abrindo índice existente em {persist_dir}")
        return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    else:
        print("⚙️ Construindo um novo índice (primeira execução).")
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
        vs.persist()
        return vs

def build_crc(retriever):
    # ChatOllama funciona bem como LLM para o CRC
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

    # return_source_documents=True => devolve trechos usados (útil para “citações”)
    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        # opcional: paraphrase follow-up into a standalone question (default True nessa versão)
        condense_question_llm=llm,
        verbose=False,
    )
    return crc

def pretty_sources(sources):
    uniq = []
    for d in sources:
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "desconhecido"
        if src not in uniq:
            uniq.append(src)
    if not uniq:
        return "—"
    return "\n".join(f"• {s}" for s in uniq)

def run_chat(k=4):
    vs = build_or_load_vectorstore(PERSIST_DIR)
    retriever = vs.as_retriever(
        search_type="mmr",                    # reduz redundância
        search_kwargs={"k": k, "fetch_k": max(k*3, 12)}
    )
    chain = build_crc(retriever)

    chat_history: List[Tuple[str, str]] = []
    print("\n💬 Chat RAG (sair: Ctrl+C). Faça perguntas sobre seus documentos.\n")
    while True:
        try:
            q = input("Você: ").strip()
            if not q:
                continue
            res = chain.invoke({"question": q, "chat_history": chat_history})
            answer = res["answer"]
            sources = res.get("source_documents", [])
            print("\nAssistente:\n" + answer.strip() + "\n")
            if sources:
                print("Fontes:\n" + pretty_sources(sources) + "\n")
            # atualiza histórico para próximos follow-ups
            chat_history.append((q, answer))
        except KeyboardInterrupt:
            print("\n👋 Encerrando. Até mais!")
            break

def ingest_only(chunk_size=1000, chunk_overlap=150):
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    vs.persist()
    print(f"💾 Índice persistido em: {PERSIST_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG + ConversationalRetrievalChain (Llama + Chroma + LangChain)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingerir/atualizar índice")
    p_ing.add_argument("--chunk-size", type=int, default=1000)
    p_ing.add_argument("--chunk-overlap", type=int, default=150)
    p_ing.set_defaults(func=lambda a: ingest_only(a.chunk_size, a.chunk_overlap))

    p_chat = sub.add_parser("chat", help="Iniciar chat conversacional (RAG)")
    p_chat.add_argument("-k", type=int, default=4)
    p_chat.set_defaults(func=lambda a: run_chat(k=a.k))

    args = parser.parse_args()
    args.func(args)
