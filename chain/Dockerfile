FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN  python3 -m pip install langchain langchain_openai fastapi[standard] langchain_community langchain_core chromadb neo4j python-dotenv jq openai

COPY . .

EXPOSE 8090

ENTRYPOINT ["fastapi", "run", "api.py", "--port", "8090"]