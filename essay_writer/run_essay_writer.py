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


if __name__ == "__main__":
    main()

