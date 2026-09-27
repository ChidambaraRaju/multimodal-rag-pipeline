"""Run the attention-paper ingestion path."""

from multimodal_rag.chunk import build_text_chunk_documents
from multimodal_rag.config import attention_paper_path, load_project_env
from multimodal_rag.parse import convert_pdf_to_docling_document
from multimodal_rag.pictures import build_picture_documents
from multimodal_rag.store import search_stored_chunks, upsert_chunk_documents


def main() -> None:
    load_project_env()
    attention_paper = attention_paper_path()
    docling_document = convert_pdf_to_docling_document(attention_paper)
    text_chunk_documents, dropped_picture_caption_count = build_text_chunk_documents(
        docling_document, attention_paper
    )
    picture_chunk_documents, skipped_picture_count = build_picture_documents(
        docling_document, attention_paper
    )
    chunk_documents = text_chunk_documents + picture_chunk_documents
    upserted_chunk_count = upsert_chunk_documents(chunk_documents)
    print(
        "text chunks",
        len(text_chunk_documents),
        "picture captions left out",
        dropped_picture_caption_count,
        "pictures kept",
        len(picture_chunk_documents),
        "pictures skipped",
        skipped_picture_count,
        "upserted",
        upserted_chunk_count,
    )

    question = "How does multi-head attention work?"
    print("question:", question)
    for rank, (retrieved_chunk, similarity_score) in enumerate(search_stored_chunks(question), start=1):
        print(
            f"#{rank} score={similarity_score:.4f} "
            f"type={retrieved_chunk.metadata.get('element_type')} "
            f"page={retrieved_chunk.metadata.get('page')} "
            f"headings={retrieved_chunk.metadata.get('headings')}"
        )
        print(retrieved_chunk.page_content[:400])


if __name__ == "__main__":
    main()
