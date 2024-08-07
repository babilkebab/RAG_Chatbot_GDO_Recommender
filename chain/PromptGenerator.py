from   abc import abstractmethod
from prompt_samples import CYPHER_CONTEXT_TEMPLATE, RESPONSER_CONTEXT_TEMPLATE, multiquery_template
from   langchain_core.prompts  import (
       ChatPromptTemplate,
       PromptTemplate,
       SystemMessagePromptTemplate,
       MessagesPlaceholder,
)

import logging


logger = logging.getLogger("uvicorn")

class PromptGenerator:
    def __init__(self, examples):
        self.examples = examples

    #Selection of best few-shot examples
    @abstractmethod
    def few_shot_example_part(self):
        pass

    #Prompt generation using the selected few-shot examples
    @abstractmethod
    def prompt(self):
        pass



class CypherPromptGenerator(PromptGenerator):
    def __init__(self, examples):
        super().__init__(examples)
        

    def prompt(self):

        few_shot_template_str = CYPHER_CONTEXT_TEMPLATE + self.examples + "La domanda è: {question}\n\n"

        few_shot_prompt = SystemMessagePromptTemplate.from_template(few_shot_template_str)

        logger.info(few_shot_template_str)

        return ChatPromptTemplate.from_messages(
            [
                few_shot_prompt,
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )


class ResponserPromptGenerator(PromptGenerator):
    def __init__(self, examples=None):
        super().__init__(examples)


    def prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"], template=RESPONSER_CONTEXT_TEMPLATE
        )
    
class MultiQueryPromptGenerator(PromptGenerator):
    def __init__(self, examples=None):
        super().__init__(examples)

    def prompt(self, numexpr):

        MULTIQUERY_GEN_CONTEXT_TEMPLATE = multiquery_template(numexpr)

        return PromptTemplate(
            input_variables=["question"], template=MULTIQUERY_GEN_CONTEXT_TEMPLATE
        )