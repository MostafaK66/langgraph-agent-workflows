# from utility import EssayWriterAgent
# from dotenv import load_dotenv
# import sys
#
# def main():
#     load_dotenv()
#     agent = EssayWriterAgent()
#     print("✅ Model initialized successfully.")
#
#     print(
#         "📝 Enter your essay topic (multi-line allowed). Press Enter, then Ctrl+D (Linux/macOS) or Ctrl+Z (Windows) when done:")
#     task = sys.stdin.read().strip()
#
#     state = {
#         "task": task,
#         "plan": "",
#         "draft": "",
#         "critique": "",
#         "content": [],
#         "revision_number": 0,
#         "max_revisions": 3
#     }
#
#     try:
#         result = agent.plan_node(state)
#         print("\n🧠 Essay Plan Generated:")
#         print(result["plan"])
#     except Exception as e:
#         print("❌ Failed to generate essay plan:", str(e))
#
#     try:
#         result = agent.research_plan_node(state)
#         state.update(result)
#         print("\n🔍 Research Content Added:")
#         for i, snippet in enumerate(state["content"], 1):
#             print(f"  [{i}] {snippet[:120].strip()}...")
#     except Exception as e:
#         print("❌ Failed to perform research:", str(e))
#         return
#
#     try:
#         result = agent.generation_node(state)
#         state.update(result)
#         print("\n📝 Essay Draft:")
#         print(state["draft"])
#     except Exception as e:
#         print("❌ Failed to generate draft:", str(e))
#         return
#
#     try:
#         result = agent.reflection_node(state)
#         state.update(result)
#         print("\n📋 Essay Critique:")
#         print(state["critique"])
#     except Exception as e:
#         print("❌ Failed to generate critique:", str(e))
#         return
#
#     try:
#         result = agent.research_critique_node(state)
#         state.update(result)
#         print("\n🔬 Additional Research Based on Critique:")
#         for i, snippet in enumerate(state["content"][-6:], 1):  # Show only last 6 additions
#             print(f"  [Critique +{i}] {snippet[:120].strip()}...")
#     except Exception as e:
#         print("❌ Failed to perform critique research:", str(e))
#
#     print("\n✅ Done.")
#
#
# if __name__ == "__main__":
#     main()
#
#
# run_essay_writer.py
# run_essay_writer.py

import sys
from dotenv import load_dotenv
from agent_state import AgentState
from utility import EssayWriterAgent
from graph_builder import EssayWriterGraph
from langgraph.graph import END

def main():
    load_dotenv()
    agent = EssayWriterAgent()
    ew_graph = EssayWriterGraph(agent).graph

    print("✅ Model initialized successfully.")
    print(
        "📝 Enter your essay topic (multi-line allowed). "
        "Press Enter, then Ctrl+D (Linux/macOS) or Ctrl+Z (Windows) when done:"
    )
    task = sys.stdin.read().strip()

    # Initial state
    initial_state: AgentState = {
        "task": task,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 3
    }
    thread = {"configurable": {"thread_id": "1"}}

    # Stream through the graph
    for event in ew_graph.stream(initial_state, thread):
        # Safely unpack node name + its output value
        node_name, val = next(iter(event.items()))
        if isinstance(val, tuple):
            state, meta = val
        else:
            state, meta = val, None

        # Print according to which node just ran:
        if node_name == "planner":
            print("\n🧠 Essay Plan Generated:")
            print(state["plan"])

        elif node_name == "research_plan":
            print("\n🔍 Research Content Added:")
            for i, snippet in enumerate(state["content"], 1):
                print(f"  [{i}] {snippet[:120].strip()}...")

        elif node_name == "generate":
            rev = state["revision_number"]
            print(f"\n📝 Essay Draft (revision #{rev}):")
            print(state["draft"])

        elif node_name == "reflect":
            print("\n📋 Essay Critique:")
            print(state["critique"])

        elif node_name == "research_critique":
            print("\n🔬 Additional Research Based on Critique:")
            for i, snippet in enumerate(state["content"][-6:], 1):
                print(f"  [Critique +{i}] {snippet[:120].strip()}...")

        # Terminate when the graph emits END
        if node_name == END:
            print("\n✅ All done—max revisions reached.")
            break

if __name__ == "__main__":
    main()
