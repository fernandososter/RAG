import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


ollama_url = "http://localhost:11434/v1"
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

