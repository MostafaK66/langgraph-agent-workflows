from dotenv import load_dotenv
from utility import ReActAgent

def main():
    load_dotenv()
    agent = ReActAgent()
    response = agent.say_hello()
    print("GPT Response:", response)

if __name__ == "__main__":
    main()






