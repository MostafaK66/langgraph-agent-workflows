from langchain_openai import ChatOpenAI

class EssayWriterAgent:
    def __init__(self):
        self.model = self._initialize_model()

    def _initialize_model(self):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
