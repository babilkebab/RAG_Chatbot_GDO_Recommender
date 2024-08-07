import logging
import re
from langchain_core.messages import HumanMessage, AIMessage
from PromptGenerator import CypherPromptGenerator, ResponserPromptGenerator
from QueryExampleSelector import QueryExampleSelector
from MultiQueryGenerator import MultiQueryGenerator



logger = logging.getLogger("uvicorn")


class CypherChain:
    def __init__(self, examples, chat_model, qa_model, multiquery_model, graph, k, embedding, numexpr):
        self.chat_model = chat_model
        self.qa_model = qa_model
        self.multiquery_model = multiquery_model
        self.graph = graph
        self.schema = graph.structured_schema
        self.k = k
        self.embedding = embedding
        examples = self._generate_multiquery(examples, numexpr)
        examples = self._generate_multilevel(examples)
        logger.info(examples)
        self.selector = QueryExampleSelector(examples, self.k, self.embedding)
        self.chat_history = []


    def _generate_multiquery(self, examples, numexpr):
        gen = MultiQueryGenerator(self.multiquery_model)
        new_examples = []
        for example in examples:
            new_examples.append(example)
            question = example["question"]
            new_questions = gen.generate_queries(question, numexpr)
            for new_question in new_questions:
                new_examples.append({"question": new_question, "query": example["query"]})
        return new_examples
    
    def _generate_multilevel(self, examples):
        new_examples = []
        for example in examples:
            level_tags = ["SECOND 2ND", "THIRD 3RD", "FOURTH 4TH"]
            level_tags_it = ["LIVELLO 2", "LIVELLO 3", "LIVELLO 4"]
            chain_tags = ["MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)-[:`son of`]->(s:lv2Node)<-[:`son of`]-(o2:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)",
                          "MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)",
                          "MATCH (p)-[:`son of`]->(m:lv4Node)<-[:`son of`]-(p2:lv5Node)"]
            nodes_names = ["s.ant", "o.ant", "m.ant"]
            tags = re.findall('\*\*\*(.*?)\*\*\*', example["question"])
            if not tags:
                continue
            tags_it = re.findall('\((.*?)\)', example["question"])
            query_splitted = example["query"].split("\n")
            index = level_tags.index(tags[0])
            level_tags.pop(index)
            level_tags_it.pop(index)
            chain_tags.pop(index)
            nodes_names.pop(index)
            new_examples.append(example)
            for i in range(len(level_tags)):
                new_question = (example["question"].replace(tags[0], level_tags[i])).replace(tags_it[0], level_tags_it[i])
                #sostituisco terza e quarta linea della query (collegamenti tra i nodi) con i livelli giusti
                fourth_line = re.sub("[a-z].ant", nodes_names[i], query_splitted[3])
                new_query = (example["query"].replace(query_splitted[2], chain_tags[i])).replace(query_splitted[3], fourth_line)
                new_examples.append({"question": new_question, "query": new_query})
        return new_examples
                

    def _generate_prompts(self, question):
        self.selected_examples = self.selector.select_examples(question)
        self.cypher_generation_prompt = CypherPromptGenerator(self.selected_examples).prompt()
        self.qa_generation_prompt = ResponserPromptGenerator().prompt()

    def _generate_chain(self, question):

        self._generate_prompts(question)

        self.cypher_chain = self.cypher_generation_prompt | self.chat_model
        self.qa_chain = self.qa_generation_prompt | self.qa_model


    def _extract_cypher(self, text): #if GPT 4 or GPT 4o
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else text

    async def _gen_cypher_query(self, invoke_input):
        cypher_query = await self.cypher_chain.ainvoke(invoke_input)
        logger.info(cypher_query.content)
        return self._extract_cypher(cypher_query.content)
    

    def _query_executor(self, cypher_query, top_k=100):
        if not cypher_query or "match" not in cypher_query.lower():
            return []
        try:
            context = self.graph.query(cypher_query)[:top_k]
            return context
        except Exception as e:
            print(e)
        return []

    async def _gen_qa_answer(self, question, context):
        qa_input = {
            "question" : question,
            "context"  : context,
        }
        qa_answer = await self.qa_chain.ainvoke(qa_input)
        return qa_answer

    async def ainvoke(self, question):

        self._generate_chain(question)

        invoke_input = {
            "question"     : question,
            "schema"       : self.schema,
            "chat_history" : self.chat_history,
        }
        cypher_query = await self._gen_cypher_query(invoke_input)
        context = self._query_executor(cypher_query)
        qa_answer = await self._gen_qa_answer(question, context)

        self.chat_history.extend(
                [
                    HumanMessage(content=question),
                    AIMessage(content=qa_answer.content)
                ]
            )
        self.chat_history = self.chat_history[-4:]
        return qa_answer.content



        





