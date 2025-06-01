from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os

class EssayWriterAgent:
    def __init__(self):
        self.model = self._initialize_model()
        self.tavily = self._initialize_tavily_client()

    def _initialize_model(self):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def _initialize_tavily_client(self):
        return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

