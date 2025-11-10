import gradio as gr
import rag_pipeline


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Desabilita warnings de multiprocessing


if __name__ == "__main__":
    demo = gr.Interface(
        fn=rag_pipeline.query_rag,
        inputs=gr.Textbox(lines=2, placeholder="Digite sua pergunta aqui..."),
        outputs="text",
        title="RAG com vLLM (Servidor Externo)",
        description="Faça perguntas e receba respostas baseadas em seus documentos.",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)
