from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os
from prompts import *
from agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage

class EssayWriterAgent:
    def __init__(self):
        self.model = self._initialize_model()
        self.tavily = self._initialize_tavily_client()

    def _initialize_model(self):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def _initialize_tavily_client(self):
        return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        return {"plan": response.content}

