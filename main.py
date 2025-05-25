from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utility import Agent

def main_agent():
    load_dotenv()
    agent = Agent.from_defaults()

    messages = [HumanMessage("What's the weather in LA?")]
    thread = {"configurable": {"thread_id": "1"}}
    print("\n── First exchange ──")
    for event in agent.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)

    while agent.graph.get_state(thread).next:
        print("\n── Paused state ──")
        print(agent.graph.get_state(thread), "\n")
        _input = input("Proceed? (y = yes, anything else = abort): ")
        if _input.strip().lower() != "y":
            print("Aborting...")
            break

        for event in agent.graph.stream(None, thread):
            for v in event.values():
                print(v)

if __name__ == "__main__":
    main_agent()











