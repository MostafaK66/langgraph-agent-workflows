from dotenv import load_dotenv
from utility import Agent
from prompts import prompt
from agent_tools import AgentTools

def main():
    load_dotenv()
    tools = AgentTools()


    abot = Agent(system=prompt)
    question = """I have 2 dogs, a border collie and a scottish terrier. \
    What is their combined weight"""
    agent_thoughts = abot(question)
    print(agent_thoughts)

    action_border_collie = "Observation: {}".format(tools.average_dog_weight("Border Collie"))
    print(action_border_collie)



    # # Ask a question to the agent
    # result = abot("How much does a toy poodle weigh?")
    # print("Agent Response:", result)
    #
    # # Also directly call the tool function
    # tools = AgentTools()
    # direct_result = tools.average_dog_weight("Toy Poodle")
    # print("Direct Tool Response:", direct_result)
    #
    # next_prompt = "Observation: {}".format(direct_result)
    # abot(next_prompt)
    # print(abot.messages)

if __name__ == "__main__":
    main()







