from chain.MultiQueryGenerator import MultiQueryGenerator
from langchain_openai import ChatOpenAI
import os
import dotenv

question = input("Enter a question: ")

dotenv.load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")

model_name = "gpt-4o-mini"
model_obj = ChatOpenAI(model=model_name, temperature=0.1)

gen = MultiQueryGenerator(model_obj)
queries = gen.generate_queries(question)

print(queries)