"""Ingest one PDF into a Chroma vector store.

1. Parse:    LlamaParse turns each page into markdown and crops its figures (parse.py).
2. Chunk:    each page is split into chunks at its headings (chunking.py).
3. Describe: a vision model describes each figure (figures.py).
4. Store:    the chunks and the figure descriptions are embedded and saved in Chroma (store.py).

Run from the project root:  python -m multimodal_rag.ingest
All settings live in config.py.
"""

from multimodal_rag import config
from multimodal_rag.chunking import build_text_documents
from multimodal_rag.figures import build_figure_documents
from multimodal_rag.parse import load_parsed_pdf
from multimodal_rag.store import save_documents


def main():
    pages, crops = load_parsed_pdf(config.DOCUMENT_PATH)
    text_documents = build_text_documents(pages)
    figure_documents = build_figure_documents(crops, pages)
    save_documents(text_documents + figure_documents)
    print(f"Stored {len(text_documents)} text chunks and {len(figure_documents)} figures in {config.VECTOR_STORE_DIRECTORY}")


if __name__ == "__main__":
    main()
