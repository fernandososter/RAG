from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from models import ModelManager

def get_rag_chain():
    # 1. Configura o ChromaDB como retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        collection_name="meu_rag",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Configura o modelo de linguagem (conecta ao servidor vLLM)
    model_manager = ModelManager(api_base="http://localhost:8000/v1")
    llm = model_manager.llm

    # 3. Define o prompt template
    template = """Responda a pergunta com base apenas no contexto abaixo:

    {context}

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Monta o pipeline
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
