from langchain_community.llms import VLLMOpenAI

class ModelManager:
    def __init__(self, api_base="http://localhost:8000/v1"):
        self.llm = VLLMOpenAI(
            openai_api_key="dummy_key",  # Não é usado, mas obrigatório
            openai_api_base=api_base,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_tokens=512,
            temperature=0.7,
        )
