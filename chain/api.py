from fastapi import FastAPI
import logging
import os
import dotenv
import json
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from CypherChain import CypherChain


models_map = {
    "GPT 3.5 Turbo": "gpt-3.5-turbo-0125",
    "GPT 4 Turbo": "gpt-4-turbo",
    "GPT 4o": "gpt-4o",
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



qa_model = ChatOpenAI(model=MODEL, temperature=0)

@app.get("/chain_settings")
async def set_chain_settings(k: int, model: str):
    global chain
    logger.info(models_map[model])
    chat_model = ChatOpenAI(model=models_map[model], temperature=0)
    k_param = k
    chain = CypherChain(examples_2, chat_model, qa_model, graph, k_param, OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"))
    return "Confermato!"




@app.get("/query")
async def get_answer(question: str):
    logger.info(question)
    response = await chain.ainvoke(question)
    return response


