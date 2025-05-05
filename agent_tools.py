from langchain_community.tools.tavily_search import TavilySearchResults


class AgentTools:
    def __init__(self):
        self.tool = TavilySearchResults(max_results=4)

    def get_known_actions(self):
        return [self.tool]

