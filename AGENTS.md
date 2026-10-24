# Multimodal RAG

Single-document RAG over a PDF with text, tables, and images. The target document is `2026-Annual-Report-Web.pdf`. Testing currently runs on `attention-is-all-you-need-first6.pdf`, the first 6 pages of the attention paper. Scale-out (multiple companies, periods, and reports) is out of scope until one document works end to end.

Keep the code simple and easy to read. This is the priority. Prefer one plain function per step over abstractions, and do not add options nobody asked for.

Always use the LangChain MCP servers before calling any LangChain, LangGraph, or LangSmith function, and follow the current docs and API reference those servers return.

- `docs-langchain` (`https://docs.langchain.com/mcp`): guides and how-tos
- `reference-langchain` (`https://reference.langchain.com/mcp`): classes, methods, and signatures

Likewise, use the `docs-llamaindex` MCP server (`https://developers.llamaindex.ai/mcp`) before calling the LlamaParse (`llama_cloud`) SDK. Its `grep_docs` tool needs a `section`; use `llama-cloud` for LlamaParse.

Experiments belong in notebooks. The user reviews each notebook run and decides whether the approach stays. Move an approach into `.py` modules only after that decision.

## Layout

- `multimodal_rag/config.py`: every setting (document path, parse and vector store folders, LlamaParse tier, model names, token limit, retry count, minimum chunk size, number of retrieved chunks). Change settings here and nowhere else. It also loads `.env`.
- `multimodal_rag/store.py`: the Chroma vector store and the embedding model, shared by ingestion and generation, and `save_documents()`, which replaces the stored contents.
- `multimodal_rag/ingest.py`: runs the ingestion steps in order. Run it from the project root with `python -m multimodal_rag.ingest`. Each step lives in its own module:
  - `multimodal_rag/parse.py`: LlamaParse, the saved parse result and the cropped images.
  - `multimodal_rag/chunking.py`: splitting each page into text chunks.
  - `multimodal_rag/figures.py`: picking the figures among the crops and describing them.
- `multimodal_rag/generate.py`: the agent that answers questions, with document search as its tool. Run it from the project root with `python -m multimodal_rag.generate` to try the test questions.
- `notebooks/multi_modal_rag.ipynb`: the notebook the pipeline came from.
- `notebooks/01_docling_parse.ipynb`, `notebooks/02_native_docling_loader.ipynb`: earlier Docling and Pinecone experiments. The pipeline no longer uses either.
- `notebooks/03_annual_report_layout.ipynb` to `notebooks/06_heading_chunks_and_image_documents.ipynb`: the move from Unstructured to LlamaParse. Notebook 06 is the approach `ingest.py` now follows. Notebooks 01, 03 and 05 no longer run against the current code or requirements.

## Ingestion

`ingest.py` runs four steps in order:

1. **Parse** (`parse.py`). LlamaParse (`agentic` tier, `llama_cloud` SDK) turns each page into markdown, with tables as HTML. A table running over several pages is merged onto its first page, and each page's printed page number is read. Figures and tables are cropped as images, which are downloaded right away because their links expire. The result and the images are saved in `config.PARSED_DIRECTORY` (`parsed/<document name>/`), and later runs reuse them instead of parsing again.
2. **Chunk** (`chunking.py`). Each page is split on its own at its headings with LangChain's `MarkdownHeaderTextSplitter`, so every chunk belongs to exactly one page. A section shorter than `MIN_CHUNK_CHARACTERS` joins the next one, so a heading never stands alone. Tables are never cut, and there is no maximum size: a page limits it.
3. **Describe figures** (`figures.py`). Every cropped figure and chart is sent to the image description model with its page's text, and the model returns a searchable description. Table crops are skipped, because the table HTML is in the text. So is a crop lying at least 90% inside a larger crop on the same page. Descriptions are saved in `image_descriptions.json` as they come; delete that file to describe again.
4. **Store** (`store.py`). Text chunks and figure descriptions are embedded and saved in a local Chroma store. Each run replaces the stored contents.

| Document | What gets embedded | Metadata |
|---|---|---|
| Text chunk | The chunk text, its tables as HTML | `page_number`, and `printed_page_number` (`""` when the page has none, since Chroma does not accept `None`) |
| Figure | Its AI description | The same page fields, and `image_path` (relative to the project root) |

Models, all set in `config.py`:

- Image description: `deepseek-flash` (DeepSeek-V4.1-Flash) on DeepSeek's own API, called with `ChatOpenAI` because the API is OpenAI-compatible. It is a reasoning model, and its reasoning tokens count toward `DESCRIPTION_MAX_TOKENS`. Earlier choices, all free on OpenRouter: `nvidia/nemotron-3.5-super-vl:free` (19 to 50 seconds a description), `dots-studio/dots-3-note-preview:free`, and `google/gemma-4-31b-it:free`, whose free pool stayed rate-limited.
- Embeddings: `nvidia/nemotron-3-embed-1b:free` on OpenRouter, text only, 2048 dimensions. It needs `input_type=passage` for chunks and `input_type=query` for questions, which `NemotronEmbeddings` in `store.py` handles.

If a description call still fails after the retries, ingestion stops with the error. The descriptions written so far are saved, so running it again picks up where it stopped.

## Query path

Generation is agentic. `generate.py` builds an agent with LangChain's `create_agent`: `qwen/qwen3.8-27b` on Groq with one tool, `search_document`. The model decides whether to search. Greetings and questions that are not about the document are answered without a search.

- **`search_document(query)`** embeds the query and reads the top `RETRIEVED_CHUNK_COUNT` chunks from Chroma, each with a relevance score from 0 to 1. Chunks scoring below `MIN_RELEVANCE_SCORE` are dropped, so weak matches and their images do not push a request past Groq's free-tier limit of 7,000 input tokens per minute. If none remain, the tool tells the model nothing relevant was found. The model receives each result's text (a passage with its tables, or a figure description) under its page number. The full chunks are kept on the tool message as its artifact.
- **Images.** Groq only accepts a plain string as a tool result, so the `show_retrieved_images` middleware adds a message right after the tool result with the image of each retrieved figure, read from its `image_path`. Do not move the images into the tool's return value.
- **Rate limits.** Groq's free tier allows about 1,000 output tokens per minute for this model, and one question takes two or more model calls. `ModelRetryMiddleware` waits and retries on a rate limit, so several questions in a row are slow.
- The system prompt tells the model to answer document questions only from the search results, to cite the page of each result it uses, and to say so when the results do not hold the answer. Citations use `page_number`, the page in the PDF.

`answer_question(question)` returns a dict with `question`, `answer`, `model`, `search_queries`, and `chunks`. `search_queries` is empty when the model answered without searching. Each chunk has `score`, `page_number`, `printed_page_number` (`""` when the page has none), `text`, and `image_path` (`None` for a text chunk). This is the function a FastAPI endpoint and UI will call later, so keep its return value JSON-friendly.

Running the file alone answers the `TEST_QUESTIONS` at the top of `generate.py` and prints the searches, the retrieved chunks, the model name, and the answer. The first test question needs no search. The last is not covered by the first 6 pages, to check the model refuses instead of guessing.

There is no API or UI yet.

## Environment

Keys are read from `.env`:

- `LLAMA_CLOUD_API_KEY`: LlamaParse parsing
- `DEEPSEEK_API_KEY`: figure descriptions
- `OPENROUTER_API_KEY`: embeddings
- `GROQ_API_KEY`: answer generation
- `SARVAM_API_KEY`: not used. Sarvam's only image model, `gemma4`, needs beta access on `/v2` that this key does not have.
- `PINECONE_API_KEY`: not used by the current pipeline
