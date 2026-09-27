"""Embed chunks with Nemotron and store them in Pinecone."""

import json
import os
import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from multimodal_rag.config import (
    NEMOTRON_EMBED_MODEL,
    NEMOTRON_EMBEDDING_DIMENSIONS,
    PINECONE_INDEX_NAME,
)


def nemotron_embeddings(input_type: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=NEMOTRON_EMBED_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
        model_kwargs={"extra_body": {"input_type": input_type}},
    )


def flatten_metadata_for_pinecone(chunk_document: Document) -> dict:
    pinecone_metadata = {"source": str(chunk_document.metadata.get("source", ""))}
    if chunk_document.metadata.get("element_type") == "picture":
        pinecone_metadata["element_type"] = "picture"
        if chunk_document.metadata.get("page") is not None:
            pinecone_metadata["page"] = int(chunk_document.metadata["page"])
        if chunk_document.metadata.get("bbox") is not None:
            pinecone_metadata["bbox"] = json.dumps(chunk_document.metadata["bbox"])
        if chunk_document.metadata.get("image_path"):
            pinecone_metadata["image_path"] = chunk_document.metadata["image_path"]
        if chunk_document.metadata.get("caption"):
            pinecone_metadata["caption"] = chunk_document.metadata["caption"]
        return pinecone_metadata

    docling_chunk_metadata = chunk_document.metadata.get("dl_meta") or {}
    docling_items = docling_chunk_metadata.get("doc_items") or []
    item_labels = [item.get("label") for item in docling_items if item.get("label")]
    pinecone_metadata["element_type"] = item_labels[0] if item_labels else "text"
    page_number = None
    bounding_box = None
    for docling_item in docling_items:
        for provenance in docling_item.get("prov") or []:
            if page_number is None and provenance.get("page_no") is not None:
                page_number = int(provenance["page_no"])
            if bounding_box is None and provenance.get("bbox"):
                box = provenance["bbox"]
                bounding_box = [box.get("l"), box.get("t"), box.get("r"), box.get("b")]
    if page_number is not None:
        pinecone_metadata["page"] = page_number
    if bounding_box is not None:
        pinecone_metadata["bbox"] = json.dumps(bounding_box)
    headings = docling_chunk_metadata.get("headings") or []
    if headings:
        pinecone_metadata["headings"] = " > ".join(headings)
    return pinecone_metadata


def open_pinecone_index():
    pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing_index_names = [index_info["name"] for index_info in pinecone_client.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_index_names:
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=NEMOTRON_EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pinecone_client.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    index_description = pinecone_client.describe_index(PINECONE_INDEX_NAME)
    if index_description.dimension != NEMOTRON_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"{PINECONE_INDEX_NAME} dimension is {index_description.dimension}, "
            f"expected {NEMOTRON_EMBEDDING_DIMENSIONS}"
        )
    return pinecone_client.Index(PINECONE_INDEX_NAME)


def upsert_chunk_documents(chunk_documents: list[Document]) -> int:
    documents_to_store = [
        Document(
            page_content=chunk_document.page_content,
            metadata=flatten_metadata_for_pinecone(chunk_document),
        )
        for chunk_document in chunk_documents
        if chunk_document.page_content.strip()
    ]
    chunk_ids = [f"chunk-{index}" for index in range(len(documents_to_store))]
    passage_vector_store = PineconeVectorStore(
        index=open_pinecone_index(),
        embedding=nemotron_embeddings("passage"),
    )
    passage_vector_store.add_documents(documents_to_store, ids=chunk_ids)
    return len(documents_to_store)


def search_stored_chunks(question: str, result_count: int = 3) -> list[tuple[Document, float]]:
    query_vector_store = PineconeVectorStore(
        index=open_pinecone_index(),
        embedding=nemotron_embeddings("query"),
    )
    return query_vector_store.similarity_search_with_score(question, k=result_count)
