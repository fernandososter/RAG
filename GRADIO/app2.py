import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr



from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")




system_message = """
You are a helpful assistant personal assistant.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

def chat(message,history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    print(messages)

    chat = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages) 
    return  chat.choices[0].message.content

gr.ChatInterface(fn=chat,type="messages").launch()

'''
vllm_url = "http://localhost:8000"
ollama = OpenAI(api_key="ollama", base_url=ollama_url)

system_message = """
You are a helpful assistant personal assistant.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

def chat(message,history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = ollama.chat.completions.create(model="hf.co/reedmayhew/claude-3.7-sonnet-reasoning-gemma3-12B:Q8_0", messages=messages)
    return  response.choices[0].message.content

gr.ChatInterface(fn=chat,type="messages").launch()
'''

