import copy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from utility import Agent

def main_agent():
    load_dotenv()
    agent = Agent.from_defaults()
    cfg = {"configurable": {"thread_id": "1"}}

    # 1) Start with your original question
    messages = [HumanMessage("What's the weather in LA?")]

    # 2) Give the human a chance to edit that very first question
    print(f"\n📝 Original question:\n  {messages[0].content}")
    edit_q = input("└─ Edit question? (y = yes, anything else = no): ").strip().lower()
    if edit_q == "y":
        new_q = input("Enter new question: ").strip()
        messages[0] = HumanMessage(new_q)
        print(f"→ Question updated to: {messages[0].content}")

    # 3) Run up to the first interrupt (the LLM “thought” before the tool call)
    for event in agent.graph.stream({"messages": messages}, config=cfg):
        node_name, val = next(iter(event.items()))
        if isinstance(val, tuple):
            state, meta = val
        else:
            state, meta = val, None

        ai_msg = state["messages"][-1]
        print("\n[AI “thought” before tool] →", repr(ai_msg.content))
        print("tool_calls:", ai_msg.tool_calls)

        # 4) Let the human edit that thought or skip it
        choice = input("edit thought (e) / skip tool (s) / continue (y): ")\
                 .strip().lower()
        if choice == "e":
            edited = copy.deepcopy(ai_msg)
            edited.content = input("Enter your own thought: ")
            edited.tool_calls = []            # drop the original tool call
            messages.append(edited)
        elif choice == "s":
            skipped = copy.deepcopy(ai_msg)
            skipped.content = "<tool skipped by human>"
            skipped.tool_calls = []
            messages.append(skipped)
        else:
            messages.append(ai_msg)
        break  # now resume from the action step

    # 5) Run the rest of the graph (tool + final answer), with edit hooks
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
                snippet = t.content.replace("\n", " ")[:200] + "…"
                print("→", snippet)

            choice = input("edit tool output (e) / skip (s) / accept (y): ")\
                     .strip().lower()
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

    # 6) Dump final state + history for your inspection
    print("\n✅ Done.")
    print("\n📌 Final State Snapshot:")
    latest_state = agent.graph.get_state(cfg)
    print(latest_state)

    print("\n🕓 State History:")
    for i, snapshot in enumerate(agent.graph.get_state_history(cfg)):
        print(f"\n🔁 State {i}:")
        print(snapshot)

if __name__ == "__main__":
    main_agent()

