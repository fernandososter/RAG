from vllm import LLM, SamplingParams
from langchain_community.llms import VLLM as LangChainVLLM

class ModelManager:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_k=10,
            top_p=0.95,
        )
        # Configura o vLLM para rodar em single process
        self.vllm_llm = LLM(
            model=model_name,
            max_model_len=16384,  # Reduzi para evitar problemas de memória
            max_num_batched_tokens=16384,
            trust_remote_code=True,
            dtype="auto",
            disable_log_stats=True,
            disable_custom_all_reduce=True,  # Evita problemas de multiprocessing
        )
        # Cria um wrapper LangChain para o vLLM
        self.llm = LangChainVLLM(
            model=model_name,
            vllm_llm=self.vllm_llm,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            temperature=0.7,
        )
