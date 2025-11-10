# iniciando server
vllm serve --served-model-name Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --max-num-batched-tokens 8192 --port 8000

vllm serve --served-model-name meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --max-num-batched-tokens 8192 --port 8000

