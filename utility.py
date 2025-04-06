from openai import OpenAI
from settings import *

class ReActAgent:
    def __init__(self):
        self.client = OpenAI()

    def say_hello(self):
        chat_completion = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Hello world"}]
        )
        return chat_completion.choices[0].message.content

