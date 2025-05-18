from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utility import Agent


def main_agent():
    load_dotenv()
    agent = Agent.from_defaults()

    cfg = {"thread_id": "1"}
    messages = [HumanMessage("How is weather in Amsterdam?")]

    print("── First exchange (streaming) ──")
    for event in agent.graph.stream({"messages": messages}, config=cfg):
        for v in event.values():
            messages.extend(v["messages"])
            for m in v["messages"]:
                role = getattr(m, "type", "unknown")
                print(f"[{role.upper()}] {m.content}")

    messages.append(HumanMessage("What about Rotterdam?"))
    print("\n── Follow-up (streaming) ──")
    for event in agent.graph.stream({"messages": messages}, config=cfg):
        for v in event.values():
            messages.extend(v["messages"])
            for m in v["messages"]:
                role = getattr(m, "type", "unknown")
                print(f"[{role.upper()}] {m.content}")


if __name__ == "__main__":
    main_agent()









