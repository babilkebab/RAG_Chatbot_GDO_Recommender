import dotenv
from fastapi import FastAPI
import logging
from model import market_cypher_chain

app = FastAPI()
dotenv.load_dotenv()
logger = logging.getLogger("uvicorn")


@app.get("/query")
async def get_answer(question: str):
    logger.info(question)
    response = await market_cypher_chain.ainvoke(question)
    return response.get("result")
