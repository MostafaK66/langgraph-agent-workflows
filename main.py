import copy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from utility import Agent

def main_agent():
    load_dotenv()
    agent = Agent.from_defaults()
    cfg = {"configurable": {"thread_id": "1"}}

    messages = [HumanMessage("What's the weather in LA?")]

    for event in agent.graph.stream({"messages": messages}, config=cfg):
        node_name, val = next(iter(event.items()))
        if isinstance(val, tuple):
            state, meta = val
        else:
            state, meta = val, None

        ai_msg = state["messages"][-1]
        print("\n[AI “thought” before tool] →", repr(ai_msg.content))
        print("tool_calls:", ai_msg.tool_calls)

        choice = input("edit thought (e) / skip tool (s) / continue (y): ").strip().lower()
        if choice == "e":
            edited = copy.deepcopy(ai_msg)
            edited.content = input("Enter your own thought: ")
            edited.tool_calls = []
            messages.append(edited)
        elif choice == "s":
            skipped = copy.deepcopy(ai_msg)
            skipped.content = "<tool skipped by human>"
            skipped.tool_calls = []
            messages.append(skipped)
        else:
            messages.append(ai_msg)

        break

    for event in agent.graph.stream(None, config=cfg):
        node_name, val = next(iter(event.items()))
        if isinstance(val, tuple):
            state, meta = val
        else:
            state, meta = val, None

        if node_name == "action":
            tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
            print("\n[Tool output]")
            for t in tool_msgs:
                print("→", t.content[:200].replace("\n", " ") + "…")

            choice = input("edit tool output (e) / skip (s) / accept (y): ").strip().lower()
            if choice == "e":
                edited = copy.deepcopy(tool_msgs[-1])
                edited.content = input("Enter your own tool result: ")
                messages.append(edited)
            elif choice == "s":
                skipped = copy.deepcopy(tool_msgs[-1])
                skipped.content = "<tool skipped by human>"
                messages.append(skipped)
            else:
                messages.extend(tool_msgs)

        elif node_name == "llm":
            final_msg = state["messages"][-1]
            print("\n[AI final answer]\n" + final_msg.content)
            messages.append(final_msg)

    print("\n✅ Done.")



if __name__ == "__main__":
    main_agent()














