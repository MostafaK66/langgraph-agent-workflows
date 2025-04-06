from dotenv import load_dotenv
from utility import Agent


def main():
    load_dotenv()
    system_instruction = "You are a helpful assistant."
    agent = Agent(system=system_instruction)
    response = agent("Hello world")
    print("GPT Response:", response)


if __name__ == "__main__":
    main()






