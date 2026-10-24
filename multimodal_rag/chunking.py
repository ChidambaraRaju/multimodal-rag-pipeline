"""Split each page's markdown into chunks at its headings.

Each page is split on its own, so every chunk belongs to exactly one page. A section shorter than
config.MIN_CHUNK_CHARACTERS joins the next one, so a heading never stands alone. Tables stay whole,
as HTML, and there is no maximum size: a page limits it.
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from multimodal_rag import config

HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5"), ("######", "h6")]
splitter = MarkdownHeaderTextSplitter(HEADINGS, strip_headers=False)  # keeps each heading in its section


def split_into_sections(markdown):
    """Cut a page's markdown at its headings. Each section starts with its heading."""
    return [document.page_content for document in splitter.split_text(markdown)]


def merge_small_sections(sections, min_characters):
    """Join each section shorter than min_characters to the next one.

    The last section of a page has no next one, so it joins the one before it.
    """
    chunks = []
    current = ""
    for section in sections:
        current = f"{current}\n\n{section}" if current else section
        if len(current) >= min_characters:
            chunks.append(current)
            current = ""
    if current and chunks:
        chunks[-1] = f"{chunks[-1]}\n\n{current}"
    elif current:
        chunks.append(current)
    return chunks


def build_text_documents(pages):
    """One document per chunk, with its page's metadata."""
    documents = []
    for page_number, page in pages.items():
        chunks = merge_small_sections(split_into_sections(page["markdown"]), config.MIN_CHUNK_CHARACTERS)
        for number, text in enumerate(chunks, start=1):
            documents.append(Document(page_content=text, metadata=page["metadata"], id=f"page{page_number}-{number}"))
    return documents
