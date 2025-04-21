from dotenv import load_dotenv
from utility import Agent
from prompts import prompt
from agent_tools import AgentTools

def main():
    load_dotenv()

    # Initialize the agent with the ReAct-style prompt
    abot = Agent(system=prompt)

    # Ask a question to the agent
    result = abot("How much does a toy poodle weigh?")
    print("Agent Response:", result)

    # Also directly call the tool function
    tools = AgentTools()
    direct_result = tools.average_dog_weight("Toy Poodle")
    print("Direct Tool Response:", direct_result)

    next_prompt = "Observation: {}".format(direct_result)
    abot(next_prompt)

if __name__ == "__main__":
    main()







