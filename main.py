from dotenv import load_dotenv
from utility import Agent
from prompts import prompt
from agent_tools import AgentTools
import re  # for extracting numbers

def extract_weight(text):
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0

def main():
    load_dotenv()

    tools = AgentTools()
    abot = Agent(system=prompt)

    question = """I have 2 dogs, a border collie and a scottish terrier. 
    What is their combined weight"""

    agent_thoughts = abot(question)
    print("Agent Response:\n", agent_thoughts)

    border_collie_resp = tools.average_dog_weight("Border Collie")
    scottish_terrier_resp = tools.average_dog_weight("Scottish Terrier")
    print("Observation:", border_collie_resp)
    print("Observation:", scottish_terrier_resp)

    border_weight = extract_weight(border_collie_resp)
    scottish_weight = extract_weight(scottish_terrier_resp)

    combined = tools.calculate(f"{border_weight} + {scottish_weight}")
    print("Final Combined Weight:", combined, "lbs")

if __name__ == "__main__":
    main()







