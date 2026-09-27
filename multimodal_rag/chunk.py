"""Structure-first text chunks. Picture captions are left out."""

from pathlib import Path

from docling.chunking import HierarchicalChunker
from docling_core.types.doc.document import DoclingDocument
from langchain_core.documents import Document


def picture_caption_reference_ids(docling_document: DoclingDocument) -> set[str]:
    return {
        caption.cref
        for picture in docling_document.pictures
        for caption in picture.captions
    }


def chunk_is_only_picture_caption(chunk, picture_caption_ids: set[str]) -> bool:
    chunk_item_ids = [item.self_ref for item in chunk.meta.doc_items]
    return bool(chunk_item_ids) and all(item_id in picture_caption_ids for item_id in chunk_item_ids)


def build_text_chunk_documents(
    docling_document: DoclingDocument, pdf_path: Path
) -> tuple[list[Document], int]:
    chunker = HierarchicalChunker()
    picture_caption_ids = picture_caption_reference_ids(docling_document)
    layout_chunks = list(chunker.chunk(docling_document))
    dropped_picture_caption_count = sum(
        chunk_is_only_picture_caption(chunk, picture_caption_ids) for chunk in layout_chunks
    )
    text_chunk_documents = [
        Document(
            page_content=chunker.contextualize(chunk=chunk),
            metadata={
                "source": str(pdf_path),
                "dl_meta": chunk.meta.export_json_dict(),
            },
        )
        for chunk in layout_chunks
        if not chunk_is_only_picture_caption(chunk, picture_caption_ids)
    ]
    return text_chunk_documents, dropped_picture_caption_count
