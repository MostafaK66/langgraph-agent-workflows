from utility import EssayWriterAgent
from dotenv import load_dotenv


def main():
    load_dotenv()
    agent = EssayWriterAgent()
    print("✅ Model initialized successfully.")

    task = input("📝 Enter your essay topic: ").strip()

    state = {
        "task": task,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 3
    }

    try:
        result = agent.plan_node(state)
        print("\n🧠 Essay Plan Generated:")
        print(result["plan"])
    except Exception as e:
        print("❌ Failed to generate essay plan:", str(e))

    try:
        result = agent.research_plan_node(state)
        state.update(result)
        print("\n🔍 Research Content Added:")
        for i, snippet in enumerate(state["content"], 1):
            print(f"  [{i}] {snippet[:120].strip()}...")
    except Exception as e:
        print("❌ Failed to perform research:", str(e))
        return

    try:
        result = agent.generation_node(state)
        state.update(result)
        print("\n📝 Essay Draft:")
        print(state["draft"])
    except Exception as e:
        print("❌ Failed to generate draft:", str(e))
        return

    try:
        result = agent.reflection_node(state)
        state.update(result)
        print("\n📋 Essay Critique:")
        print(state["critique"])
    except Exception as e:
        print("❌ Failed to generate critique:", str(e))
        return

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

