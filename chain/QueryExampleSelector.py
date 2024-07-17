import logging
import chromadb


logger = logging.getLogger("uvicorn")


class QueryExampleSelector:
    def __init__(self, examples, k, embedding):
        self.examples = {x["question"]: x["query"] for x in examples}
        self.k = k
        self.embedding = embedding
        self._conf_vectorstore()

    def _conf_vectorstore(self):
        chroma_client = chromadb.Client()
        self.collection = chroma_client.get_or_create_collection(name="examples", embedding_function=self.embedding)
        self.collection.add(
            documents=[doc for doc in self.examples.keys()],
            ids=[f"id{i}" for i in range(len(self.examples.keys()))],
        )

    def select_examples(self, query__):
        docs = self.collection.query(query_texts=[query__], n_results=self.k)["documents"][0]
        str_prompt = ""
        logger.info(docs)
        for doc in docs:
            str_prompt += f"# {doc}\n\n{self.examples[doc]}\n\n"
        return str_prompt



