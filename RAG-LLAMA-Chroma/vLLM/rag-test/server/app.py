import gradio as gr
from rag_pipeline import get_rag_chain

def query_rag(question):
    rag_chain = get_rag_chain()
    return rag_chain.invoke(question)

if __name__ == "__main__":
    demo = gr.Interface(
        fn=query_rag,
        inputs=gr.Textbox(lines=2, placeholder="Digite sua pergunta aqui..."),
        outputs="text",
        title="RAG com vLLM (Servidor Externo)",
        description="Faça perguntas e receba respostas baseadas em seus documentos."
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)
