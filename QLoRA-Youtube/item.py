from transformers import AutoTokenizer
from typing import Optional

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

class Item: 
    prompt: Optional[str] = None
    PREFIX = "Views are "
    QUESTION = "How many views for this video?"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    title: str
    view_count: float

    def __init__(self, title, view_count):
        self.title = title
        self.view_count = view_count 
        self.makePrompt(self.title)

    
    def makePrompt(self, text): 
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n" 
        self.prompt += f"{self.PREFIX}{str(round(self.view_count))}"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))


       
    def test_prompt(self):
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX