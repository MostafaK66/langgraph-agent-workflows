import os
from tavily import TavilyClient

class TavilyWrapper:
    def __init__(self):
        self.client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

    def get_client(self):
        return self.client
