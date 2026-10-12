"""Ingest one PDF into a Chroma vector store.

Steps: parse the PDF with Unstructured, split it into chunks, write an AI description
for every chunk that holds an image or table, embed the chunks, and store them.

Run from the project root:  python -m multimodal_rag.ingest
All settings live in config.py.
"""

import json
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from openrouter.errors import ResponseValidationError, TooManyRequestsResponseError
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from multimodal_rag import config

DESCRIPTION_PROMPT = """You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

{tables}
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""


class NemotronEmbeddings(Embeddings):
    """Nemotron on OpenRouter. It expects input_type=passage for chunks and input_type=query for questions."""

    def _model(self, input_type):
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
            model_kwargs={"extra_body": {"input_type": input_type}},
        )

    def embed_documents(self, texts):
        return self._model("passage").embed_documents(texts)

    def embed_query(self, text):
        return self._model("query").embed_query(text)


def parse_pdf(pdf_path):
    """Extract the text, tables and images of a PDF as a list of Unstructured elements."""
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",  # most accurate layout detection, slower
        infer_table_structure=True,  # keep tables as HTML
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,  # keep images as base64 on the element
    )


def split_into_chunks(elements):
    """Group elements into chunks that follow the document's section titles."""
    return chunk_by_title(
        elements,
        max_characters=config.CHUNK_MAX_CHARACTERS,
        new_after_n_chars=config.CHUNK_NEW_AFTER_CHARACTERS,
        combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER_CHARACTERS,
    )


def tables_and_images_in(chunk):
    """Return the table HTML strings and base64 images found inside a chunk."""
    tables = []
    images = []
    for element in chunk.metadata.orig_elements or []:
        if element.category == "Table":
            tables.append(element.metadata.text_as_html or element.text)
        elif element.category == "Image" and element.metadata.image_base64:
            images.append(element.metadata.image_base64)
    return tables, images


def description_model():
    """The image description model. The free tier is sometimes rate-limited or unavailable, so those errors are retried."""
    llm = ChatOpenRouter(
        model=config.IMAGE_DESCRIPTION_MODEL, temperature=0, max_tokens=config.DESCRIPTION_MAX_TOKENS
    )
    return llm.with_retry(
        retry_if_exception_type=(ResponseValidationError, TooManyRequestsResponseError),
        stop_after_attempt=config.MODEL_RETRY_ATTEMPTS,
    )


def describe_chunk(llm, text, tables, images):
    """Ask the model for a searchable description of a chunk's text, tables and images."""
    tables_text = ""
    if tables:
        tables_text = "TABLES:\n" + "".join(f"Table {i + 1}:\n{table}\n\n" for i, table in enumerate(tables))

    content = [{"type": "text", "text": DESCRIPTION_PROMPT.format(text=text, tables=tables_text)}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})

    return llm.invoke([HumanMessage(content=content)]).content


def build_documents(chunks):
    """Turn chunks into LangChain documents. Chunks with tables or images are embedded by their AI description."""
    llm = description_model()
    documents = []

    for number, chunk in enumerate(chunks, start=1):
        tables, images = tables_and_images_in(chunk)
        print(f"Chunk {number}/{len(chunks)}: {len(tables)} tables, {len(images)} images")

        searchable_text = chunk.text
        if tables or images:
            try:
                searchable_text = describe_chunk(llm, chunk.text, tables, images)
            except Exception as error:
                print(f"   description failed, using the raw text instead: {error}")

        # The original text, tables and images are kept so the answer step can use them
        original_content = {"raw_text": chunk.text, "tables_html": tables, "images_base64": images}
        documents.append(
            Document(page_content=searchable_text, metadata={"original_content": json.dumps(original_content)})
        )

    return documents


def store_documents(documents):
    """Embed the documents and save them in Chroma, replacing whatever was stored before."""
    vector_store = Chroma(
        persist_directory=str(config.VECTOR_STORE_DIRECTORY),
        embedding_function=NemotronEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_store.reset_collection()
    vector_store.add_documents(documents)
    return vector_store


def main():
    load_dotenv(config.PROJECT_ROOT / ".env")

    print(f"Parsing {config.DOCUMENT_PATH}")
    elements = parse_pdf(config.DOCUMENT_PATH)
    chunks = split_into_chunks(elements)
    print(f"{len(elements)} elements, {len(chunks)} chunks")

    documents = build_documents(chunks)
    store_documents(documents)
    print(f"Stored {len(documents)} chunks in {config.VECTOR_STORE_DIRECTORY}")


if __name__ == "__main__":
    main()
