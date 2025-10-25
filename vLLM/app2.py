import subprocess
import time

# Define your model and any other desired arguments
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
host = "0.0.0.0"  # Listen on all available network interfaces
port = 8000

# Construct the command
command = [
    "python",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    model_name,
    "--max-model-len 2048", 
    "--max-num-batched-tokens",
      "32768",
    "--host",
    host,
    "--port",
    str(port),
    "--served-model-name", "mistralai/Mistral-7B-Instruct-v0.3",
    #"--chat-template", "./templates/openai_default_template.jinja",
    #"--chat-template-content-format", "openai",
    # Add other arguments as needed, e.g., --tensor-parallel-size, --gpu-memory-utilization, etc.
]



print(f"Starting vLLM server with command: {' '.join(command)}")

# Start the server process
# Use `Popen` for non-blocking execution if you need to do other things in your script
# before or after the server starts.
server_process = subprocess.Popen(command)

print(f"vLLM server started on {host}:{port}. PID: {server_process.pid}")

# You can add a delay to allow the server to fully initialize
time.sleep(10) # Adjust as needed

# Now you can interact with the server, e.g., send requests using an HTTP client
# For example, using the OpenAI Python client:
# from openai import OpenAI
# client = OpenAI(api_key="YOUR_VLLM_API_KEY_IF_SET", base_url=f"http://{host}:{port}/v1")
# response = client.chat.completions.create(
#     model=model_name,
#     messages=[{"role": "user", "content": "Hello, how are you?"}]
# )
# print(response.choices[0].message.content)

# To stop the server (e.g., when your application exits)
# server_process.terminate()
# server_process.wait()
# print("vLLM server terminated.")