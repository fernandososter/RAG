#pip install langchain-chroma


import torch
from langchain_community.llms import VLLM
import chromadb
from vllm.entrypoints.openai.api_server import serve
from vllm.entrypoints.openai.async_api_server import serve 
import asyncio



chroma_client = chromadb.Client()




import multiprocessing



def main():
    asyncio.run(serve(
        model="facebook/opt-125m",   # nome ou caminho do modelo
        host="0.0.0.0",                           # para acesso externo
        port=8000,                                # porta HTTP
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        api_key=None,                             # opcional
        served_model_name="llama3" ,               # nome que aparecerá na API
        chroma_client=chroma_client
    ))




if __name__ == "__main__":
    # para compatibilidade cross-platform com multiprocessing
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass


    # opcional: definir método de start (ver nota abaixo)
    # multiprocessing.set_start_method("fork")   # veja nota
    main()
