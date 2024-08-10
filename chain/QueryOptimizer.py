from PromptGenerator import QueryOptimizerPromptGenerator
import logging 

logger = logging.getLogger("uvicorn")

class QueryOptimizer:
    def __init__(self, model):
        prompt = QueryOptimizerPromptGenerator().prompt()
        self.optimizer_chain = prompt | model

    def optimize(self, query):
        response = self.optimizer_chain.invoke(query)
        return response.content