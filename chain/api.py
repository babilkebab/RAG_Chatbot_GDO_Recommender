from fastapi import FastAPI
import logging
import os
import dotenv
import json
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from CypherChain import CypherChain


def flag_negs(text):
    if "non" in text.lower():
        return "***NEGATION*** " + text + " ***NEGATION***"
    return text

def flag_level(text):
    if "livello 2" in text.lower():
        return "***SECOND LEVEL*** " + text + " ***SECOND LEVEL***"
    elif "livello 3" in text.lower():
        return "***THIRD LEVEL*** " + text + " ***THIRD LEVEL***"
    elif "livello 4" in text.lower():
        return "***FOURTH LEVEL*** " + text + " ***FOURTH LEVEL***"
    return text

models_map = {
    "GPT 4o Mini": "gpt-4o-mini",
    "GPT 4 Turbo": "gpt-4-turbo",
    "GPT 4o": "gpt-4o",
    "GPT 3.5 Turbo": "gpt-3.5-turbo-0125",
}


chain = None


app = FastAPI()
logger = logging.getLogger("uvicorn")

dotenv.load_dotenv()

MODEL = os.environ.get("LLM_MODEL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CHROMA_PORT = os.environ.get("CHROMA_PORT")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=True
)
graph.refresh_schema()


examples_2 = json.load(open("examples/queries_examples.json"))["examples"]



qa_model = ChatOpenAI(model=MODEL, temperature=0.5)
multiquery_model = ChatOpenAI(model=MODEL, temperature=0.1)

@app.get("/chain_settings")
async def set_chain_settings(k: int, model: str):
    global chain
    logger.info(models_map[model])
    chat_model = ChatOpenAI(model=models_map[model], temperature=0)
    k_param = k
    numexpr = k-1
    chain = CypherChain(examples_2, chat_model, qa_model, multiquery_model, graph, k_param, OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"), numexpr)
    return "Confermato!"




@app.get("/query")
async def get_answer(question: str):
    question = flag_level(flag_negs(question))
    logger.info(question)
    response = await chain.ainvoke(question)
    return response


