from utility import EssayWriterAgent
from dotenv import load_dotenv


def main():
    load_dotenv()
    agent = EssayWriterAgent()
    print("✅ Model initialized successfully.")

    # Ask user for essay topic
    task = input("📝 Enter your essay topic: ").strip()

    # Build initial agent state
    state = {
        "task": task,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 3
    }

    # Run planning step
    try:
        result = agent.plan_node(state)
        print("\n🧠 Essay Plan Generated:")
        print(result["plan"])
    except Exception as e:
        print("❌ Failed to generate essay plan:", str(e))


if __name__ == "__main__":
    main()

