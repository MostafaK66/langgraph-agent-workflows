from utility import EssayWriterAgent
from dotenv import load_dotenv


def main():
    load_dotenv()
    agent = EssayWriterAgent()
    print("✅ Model initialized successfully.")

    # Optional: Run a quick prompt to verify connection
    try:
        response = agent.model.invoke([{"role": "user", "content": "Say hello!"}])
        print("🤖 Response:", response.content)
    except Exception as e:
        print("❌ Failed to get response from model:", str(e))


if __name__ == "__main__":
    main()
