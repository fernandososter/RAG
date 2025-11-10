import requests

class RAGClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url

    def query(self, question):
        response = requests.post(f"{self.server_url}/query", json={"query": question})
        return response.json()["answer"]

# Exemplo de uso
if __name__ == "__main__":
    client = RAGClient()
    print(client.query("Qual a capital do Brasil?"))
