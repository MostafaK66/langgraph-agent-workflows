import os
from tavily import TavilyClient
from dotenv import load_dotenv

class TavilyWrapper:
    def __init__(self):
        load_dotenv()
        self.client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def get_client(self):
        return self.client

    def search_with_answer(self, query: str):
        result = self.client.search(query, include_answer=True)
        return result["answer"]



tw = TavilyWrapper()
answer = tw.search_with_answer("What is in Nvidia's new Blackwell GPU?")
print("✅ Tavily Answer:", answer)