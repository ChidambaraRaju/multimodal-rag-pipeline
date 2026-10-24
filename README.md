# Multimodal RAG

A retrieval-augmented generation pipeline for PDF documents that contain text, tables, and images. It answers questions using all three, not only the text, and cites the pages it used.

## How it works

**Ingestion** turns a PDF into a searchable vector store:

1. **Parse.** [LlamaParse](https://developers.llamaindex.ai/llamaparse/parse/getting_started) reads each page with a vision model and returns it as markdown, with tables as HTML. It also crops the figures from each page. The result is saved, so the PDF is parsed only once.
2. **Chunk.** Each page is split at its headings into chunks. A short section joins the next one, so a heading never stands alone, and a table is never cut.
3. **Describe.** Every chart and diagram is sent to a vision model with the text of its page, and the model writes a searchable description of it. Other cropped images, such as photos, logos and signatures, are skipped.
4. **Store.** The text chunks and the chart descriptions are embedded and saved in a local [Chroma](https://www.trychroma.com/) vector store, each with its page number. A chart also keeps the path of its image.

**Generation** is agentic. The answering model is given document search as a tool and decides for itself when to use it:

1. **Decide.** The model reads the question. A greeting or a question that is not about the document is answered directly.
2. **Search.** Otherwise the model calls the search tool, which fetches the most relevant chunks and charts. It can search again with a different query if the first results fall short.
3. **Answer.** The model answers from the text, tables, and chart images it found, citing their pages. If the document does not hold the answer, it says so.

## Setup

Python 3.12 or 3.13. Install the Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```
LLAMA_CLOUD_API_KEY=...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
```

- `LLAMA_CLOUD_API_KEY` is used for parsing (LlamaParse).
- `DEEPSEEK_API_KEY` is used for chart descriptions.
- `OPENROUTER_API_KEY` is used for embeddings.
- `GROQ_API_KEY` is used for answer generation.

## Usage

Run both commands from the project root.

Ingest the document set in `config.py`:

```bash
python -m multimodal_rag.ingest
```

The first run parses the PDF with LlamaParse, which costs 10 credits a page on the `agentic` tier, and describes its figures. Both results are saved in `parsed/<document name>/` and reused by later runs. Delete that folder to parse again. Each ingestion run replaces the contents of the vector store.

Ask the test questions and print the searches the model made, the retrieved chunks, the model name, and the answer:

```bash
python -m multimodal_rag.generate
```

The test questions are the `TEST_QUESTIONS` list at the top of `multimodal_rag/generate.py`.

To use the pipeline from your own code:

```python
from multimodal_rag.generate import answer_question

result = answer_question("Your question here")
print(result["answer"])
```

`answer_question` returns a dictionary with `question`, `answer`, `model`, `search_queries`, and `chunks`. `search_queries` is empty when the model answered without searching. Each chunk has `score`, `page_number`, `printed_page_number` (empty when the page has none), `text`, and `image_path`. `image_path` is set only for a chart, and `text` is then its description.

## Configuration

Every setting lives in `multimodal_rag/config.py`:

| Setting | What it controls |
|---|---|
| `DOCUMENT_PATH` | The PDF to ingest |
| `PARSED_DIRECTORY` | Where the parse result, the cropped images, and the chart descriptions are saved |
| `VECTOR_STORE_DIRECTORY` | Where the Chroma vector store is saved |
| `PARSE_TIER` | LlamaParse tier |
| `IMAGE_DESCRIPTION_MODEL`, `DESCRIPTION_BASE_URL` | Vision model that describes charts (DeepSeek) |
| `EMBEDDING_MODEL` | Text embedding model (OpenRouter) |
| `GENERATION_MODEL` | Model that decides when to search and writes the answer (Groq) |
| `RETRIEVED_CHUNK_COUNT` | How many chunks each search returns |
| `MIN_RELEVANCE_SCORE` | Chunks scoring below this are left out of the search results |
| `DESCRIPTION_MAX_TOKENS` | Longest description the vision model may write |
| `MODEL_RETRY_ATTEMPTS` | Retries for a rate-limited or unavailable model call |
| `MIN_CHUNK_CHARACTERS` | A section shorter than this joins the next one |
The PDFs in this repository are sample documents used during development. Point `DOCUMENT_PATH` at any PDF to use your own.

## Project layout

```
multimodal_rag/
  config.py     all settings, loads .env
  store.py      Chroma vector store and embedding model
  ingest.py     ingestion pipeline: runs the four steps below in order
  parse.py      1. parse the PDF with LlamaParse, once, and save the result and the cropped images
  chunking.py   2. split each page into text chunks at its headings
  figures.py    3. pick the charts among the cropped images and describe them
  generate.py   agent that answers questions, with document search as its tool
notebooks/      experiments the pipeline grew out of
parsed/         saved parse results, cropped images and chart descriptions (not in git)
```

## Status

Ingestion and generation work from the command line for a single document: NVIDIA's 2026 annual report (175 pages), which becomes 377 text chunks and 12 chart descriptions. All 6 test questions in `generate.py` are answered correctly. Planned next: a FastAPI backend and a web UI.
