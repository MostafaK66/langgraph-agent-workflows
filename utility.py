from openai import OpenAI

class ReActAgent:
    def __init__(self):
        self.client = OpenAI()

    def say_hello(self):
        chat_completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}]
        )
        return chat_completion.choices[0].message.content

