from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utility import Agent
from agent_state import AgentState

def main_agent():
    load_dotenv()
    agent = Agent.from_defaults()
    question = "what about Amsterdam?"
    state = AgentState(messages=[HumanMessage(content=question)])

    result = agent.graph.invoke(state)

    for msg in result["messages"]:
        if msg.type == "ai":
            print("\n✅ Final Answer:")
            print(msg.content)

if __name__ == "__main__":
    main_agent()







