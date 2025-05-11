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



city = "Rotterdam"
query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    weather.com
""".strip()

tw = TavilyWrapper()
answer = tw.search_with_answer(query)
print("✅ Tavily Answer:", answer)
