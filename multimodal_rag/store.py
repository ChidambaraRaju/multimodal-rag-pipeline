"""The Chroma vector store and its embedding model, shared by ingestion and generation."""

import os
import time

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from multimodal_rag import config


class JinaEmbeddings(Embeddings):
    """Jina through its OpenAI-compatible API. It expects task=retrieval.passage for chunks and task=retrieval.query for questions."""

    def _model(self, task):
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=os.environ["JINA_API_KEY"],
            base_url="https://api.jina.ai/v1",
            check_embedding_ctx_length=False,  # send the text itself, not OpenAI token ids
            max_retries=config.MODEL_RETRY_ATTEMPTS,
            model_kwargs={"extra_body": {"task": task}},
        )

    def embed_documents(self, texts):
        return self._model("retrieval.passage").embed_documents(texts)

    def embed_query(self, text):
        return self._model("retrieval.query").embed_query(text)


def open_vector_store():
    """Open the Chroma vector store saved on disk."""
    return Chroma(
        persist_directory=str(config.VECTOR_STORE_DIRECTORY),
        embedding_function=JinaEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def save_documents(documents):
    """Embed the documents and save them, replacing whatever was stored before.

    A free Jina key embeds at most 100,000 tokens a minute, so the documents go in batches, a minute apart.
    """
    vector_store = open_vector_store()
    vector_store.reset_collection()
    for start in range(0, len(documents), config.EMBEDDING_BATCH_SIZE):
        if start:
            time.sleep(60)
        batch = documents[start:start + config.EMBEDDING_BATCH_SIZE]
        print(f"Embedding documents {start + 1} to {start + len(batch)} of {len(documents)}")
        vector_store.add_documents(batch)
