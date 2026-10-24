"""The Chroma vector store and its embedding model, shared by ingestion and generation."""

import os

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from multimodal_rag import config


class NemotronEmbeddings(Embeddings):
    """Nemotron on OpenRouter. It expects input_type=passage for chunks and input_type=query for questions."""

    def _model(self, input_type):
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
            chunk_size=256,  # texts per request; the embedding API accepts at most 256
            model_kwargs={"extra_body": {"input_type": input_type}},
        )

    def embed_documents(self, texts):
        return self._model("passage").embed_documents(texts)

    def embed_query(self, text):
        return self._model("query").embed_query(text)


def open_vector_store():
    """Open the Chroma vector store saved on disk."""
    return Chroma(
        persist_directory=str(config.VECTOR_STORE_DIRECTORY),
        embedding_function=NemotronEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def save_documents(documents):
    """Embed the documents and save them, replacing whatever was stored before."""
    vector_store = open_vector_store()
    vector_store.reset_collection()
    vector_store.add_documents(documents)
