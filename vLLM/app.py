#pip install langchain-chroma


import torch
from langchain_community.llms import VLLM
import chromadb
import asyncio
import multiprocessing



chroma_client = chromadb.Client()






# app.py
import sys
import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

def main():
    # monta um parser igual ao CLI do vLLM
    parser = make_arg_parser()

    # você pode passar os argumentos aqui (ou deixar sys.argv[1:] para usar args de linha de comando)
    # Exemplo de args programáticos:
    argv = [
        "--model", "llama-3-8b-instruct",   # troque para o seu modelo/caminho
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-num-batched-tokens", "8192",
        "--max-model-len", "8192",
        "--served-model-name", "llama3_local",
        # adicione outros flags que quiser: --api-key, --dtype, --gpu-memory-utilization, etc.
    ]

    args = parser.parse_args(argv)

    # valida alguns constraints que o vLLM espera
    validate_parsed_serve_args(args)

    # uvloop.run espera uma coroutine; run_server(args) é a coroutine principal do servidor.
    try:
        uvloop.run(run_server(args))
    except Exception as e:
        # trate ou re-raise para ver erro
        raise





if __name__ == "__main__":
    # para compatibilidade cross-platform com multiprocessing
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass


    # opcional: definir método de start (ver nota abaixo)
    # multiprocessing.set_start_method("fork")   # veja nota
    main()
