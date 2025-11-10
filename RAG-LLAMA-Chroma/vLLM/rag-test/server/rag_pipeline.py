from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel,RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from models import ModelManager
from callbacks import time_callback

def query_rag(question):
    rag_chain = get_rag_chain()
    result = rag_chain.invoke(question, config={"callbacks": [StdOutCallbackHandler(), time_callback.TimeCallback()]})
    print(result)
    return result

def get_rag_chain():
    # 1. Configura o ChromaDB como retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        collection_name="rag-vllm-test1",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Configura o modelo de linguagem (conecta ao servidor vLLM)
    model_manager = ModelManager(api_base="http://localhost:8000/v1")
    llm = model_manager.llm

    # 3. Define o prompt template
    template = """Responda sempre no mesmo idioma da pergunta do usuário.
    Responda à pergunta com base APENAS no contexto abaixo.
    Se a resposta não estiver no contexto, diga que não encontrou.
    Forneça apenas UMA resposta final, sem repetir a mesma frase.
    Ignore prefixos como "Assistant:" ou "Resposta:" que apareçam no contexto.
    Contexto: {context}

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Monta o pipeline
    def format_docs(docs):
        seen = set()
        parts = []
        for d in docs:
            text = d.page_content.strip()
            if text not in seen:
                seen.add(text)
                print(f"text: { text} ")
                parts.append(text)
        return "\n\n---\n\n".join(parts)

    
    # Função para formatar o contexto e a pergunta em um dicionário
    def format_input(data: dict) -> dict:

        return {
            "context": data["context"],
            "question": data["question"],
    }

    '''
    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    '''

    rag_chain = RunnableSequence(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(format_input)  # Não precisa de RunnableLambda aqui
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
