# Multimodal RAG

A retrieval-augmented generation pipeline for PDF documents that contain text, tables, and images. It answers questions using all three, not only the text.

## How it works

**Ingestion** turns a PDF into a searchable vector store:

1. **Parse.** [Unstructured](https://docs.unstructured.io/) extracts the text, tables (as HTML), and images from the PDF.
2. **Chunk.** Elements are grouped into chunks that follow the document's section titles.
3. **Describe.** Every chunk that holds a table or an image is sent to a vision model, which writes a searchable description of it. Text-only chunks skip this step.
4. **Store.** Chunks are embedded and saved in a local [Chroma](https://www.trychroma.com/) vector store. The original text, tables, and images are kept alongside each chunk.

**Generation** answers a question from that store:

1. **Retrieve.** The question is embedded and the most relevant chunks are fetched.
2. **Generate.** The chunks' original text, tables, and images are sent to a multimodal model, which writes the answer. If the document does not hold the answer, the model says so.

## Setup

Python 3.12 is used in development.

Install the system packages Unstructured needs for PDFs:

```bash
# Linux
sudo apt-get install -y poppler-utils tesseract-ocr libmagic-dev

# macOS
brew install poppler tesseract libmagic
```

Install the Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
```

- `OPENROUTER_API_KEY` is used for image descriptions and embeddings.
- `GROQ_API_KEY` is used for answer generation.

## Usage

Run both commands from the project root.

Ingest the document set in `config.py`:

```bash
python -m multimodal_rag.ingest
```

Ask the test questions and print the retrieved chunks, the model name, and the answer:

```bash
python -m multimodal_rag.generate
```

The test questions are the `TEST_QUESTIONS` list at the top of `multimodal_rag/generate.py`. Each ingestion run replaces the contents of the vector store.

To use the pipeline from your own code:

```python
from multimodal_rag.generate import answer_question

result = answer_question("Your question here")
print(result["answer"])
```

`answer_question` returns a dictionary with `question`, `answer`, `model`, and `chunks`. Each chunk has `score`, `text`, `tables_html`, and `images_base64`.

## Configuration

Every setting lives in `multimodal_rag/config.py`:

| Setting | What it controls |
|---|---|
| `DOCUMENT_PATH` | The PDF to ingest |
| `VECTOR_STORE_DIRECTORY` | Where the Chroma vector store is saved |
| `IMAGE_DESCRIPTION_MODEL` | Vision model that describes images and tables (OpenRouter) |
| `EMBEDDING_MODEL` | Text embedding model (OpenRouter) |
| `GENERATION_MODEL` | Model that writes the answer (Groq) |
| `RETRIEVED_CHUNK_COUNT` | How many chunks are retrieved per question |
| `DESCRIPTION_MAX_TOKENS` | Longest description the vision model may write |
| `MODEL_RETRY_ATTEMPTS` | Retries for a rate-limited or unavailable model call |
| `CHUNK_MAX_CHARACTERS`, `CHUNK_NEW_AFTER_CHARACTERS`, `CHUNK_COMBINE_UNDER_CHARACTERS` | Chunk sizes |

The PDFs in this repository are sample documents used during development. Point `DOCUMENT_PATH` at any PDF to use your own.

## Project layout

```
multimodal_rag/
  config.py     all settings, loads .env
  store.py      Chroma vector store and embedding model
  ingest.py     ingestion pipeline
  generate.py   retrieval and answer generation
notebooks/      experiments the pipeline grew out of
```

## Status

Ingestion and generation work from the command line for a single document. Planned next: a FastAPI endpoint and a web UI.
