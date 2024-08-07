from PromptGenerator import MultiQueryPromptGenerator


class MultiQueryGenerator:
    def __init__(self, query_gen_model):
        self.query_gen_model = query_gen_model

    def _generate_prompt(self, numexpr):
        return MultiQueryPromptGenerator().prompt(numexpr)
    
    def _extract_queries(self, text):
        queries = text.split('\n\n')
        queries[len(queries)-1] = queries[len(queries)-1].replace('.','')
        return queries


    def generate_queries(self, question, numexpr):
        prompt = self._generate_prompt(numexpr)
        generator_chain = prompt | self.query_gen_model
        response = generator_chain.invoke(question)
        return self._extract_queries(response.content)


    