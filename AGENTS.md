# Multimodal RAG

Single-document RAG over a PDF with text, tables, and images. The document is `2026-Annual-Report-Web.pdf`, NVIDIA's 2026 annual report (175 pages), and it is fully ingested. `attention-is-all-you-need-first6.pdf`, the first 6 pages of the attention paper, was the earlier test document. Scale-out (multiple companies, periods, and reports) is out of scope until one document works end to end.

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
  - `multimodal_rag/figures.py`: picking the charts among the crops and describing them.
- `multimodal_rag/generate.py`: the agent that answers questions, with document search as its tool. Run it from the project root with `python -m multimodal_rag.generate` to try the test questions.
- `notebooks/multi_modal_rag.ipynb`: the notebook the pipeline came from.
- `notebooks/01_docling_parse.ipynb`, `notebooks/02_native_docling_loader.ipynb`: earlier Docling and Pinecone experiments. The pipeline no longer uses either.
- `notebooks/03_annual_report_layout.ipynb` to `notebooks/06_heading_chunks_and_image_documents.ipynb`: the move from Unstructured to LlamaParse. Notebook 06 is the approach `ingest.py` now follows. Notebooks 01, 03 and 05 no longer run against the current code or requirements.

## Ingestion

`ingest.py` runs four steps in order:

1. **Parse** (`parse.py`). LlamaParse (`agentic` tier, `llama_cloud` SDK) turns each page into markdown, with tables as HTML. A table running over several pages is merged onto its first page, and each page's printed page number is read. Figures and tables are cropped as images. The result is saved in `config.PARSED_DIRECTORY` (`parsed/<document name>/`) as soon as the parse finishes, and later runs reuse it instead of parsing again. Each run then downloads the cropped images not yet on disk, with fresh links from the parse job, because image links expire. If a download fails, running ingestion again fetches the rest.
2. **Chunk** (`chunking.py`). Each page is split on its own at its headings with LangChain's `MarkdownHeaderTextSplitter`, so every chunk belongs to exactly one page. A section shorter than `MIN_CHUNK_CHARACTERS` joins the next one, so a heading never stands alone. Tables are never cut, and there is no maximum size: a page limits it.
3. **Describe charts** (`figures.py`). Every crop LlamaParse files as a `chart` is sent to the image description model with its page's text, and the model returns a searchable description. Diagrams such as page 8's agentic AI architecture are filed as charts too. The other crops are skipped: `table` crops because the table HTML is in the text, and `image` crops because in this report they are photos, logos, signatures and icons that add little beyond the page text. Descriptions are saved in `image_descriptions.json` as they come; delete that file to describe again.
4. **Store** (`store.py`). Text chunks and chart descriptions are embedded and saved in a local Chroma store. Each run replaces the stored contents.

| Document | What gets embedded | Metadata |
|---|---|---|
| Text chunk | The chunk text, its tables as HTML | `page_number`, and `printed_page_number` (`""` when the page has none, since Chroma does not accept `None`) |
| Chart | Its AI description | The same page fields, and `image_path` (relative to the project root) |

Models, all set in `config.py`:

- Image description: `deepseek-flash` (DeepSeek-V4.1-Flash) on DeepSeek's own API, called with `ChatOpenAI` because the API is OpenAI-compatible. It is a reasoning model, and its reasoning tokens count toward `DESCRIPTION_MAX_TOKENS`. Earlier choices, all free on OpenRouter: `nvidia/nemotron-3.5-super-vl:free` (19 to 50 seconds a description), `dots-studio/dots-3-note-preview:free`, and `google/gemma-4-31b-it:free`, whose free pool stayed rate-limited.
- Embeddings: `jina-embeddings-v5-text-small` on Jina's API, text only, 1024 dimensions, up to 32,768 tokens a text. Jina's endpoint takes OpenAI's request format, so `JinaEmbeddings` in `store.py` calls it with LangChain's `OpenAIEmbeddings` (`check_embedding_ctx_length=False`, so the text itself is sent). It passes `task=retrieval.passage` for chunks and `task=retrieval.query` for questions. LangChain's own `JinaEmbeddings` (in `langchain-community`, which is being sunset) cannot pass a task and sends every text in one request, so it is not used. A free Jina key embeds at most 100,000 tokens a minute, so `save_documents()` sends `EMBEDDING_BATCH_SIZE` (64) documents a minute; the whole report takes about 7 minutes. Earlier: `nvidia/nemotron-3-embed-1b:free` on OpenRouter, dropped because free OpenRouter models allow only 50 requests a day.

If a description call still fails after the retries, ingestion stops with the error. The descriptions written so far are saved, so running it again picks up where it stopped.

## Query path

Generation is agentic. `generate.py` builds an agent with LangChain's `create_agent`: `deepseek-flash` (DeepSeek-V4.1-Flash, the model that also describes the charts) on DeepSeek's own API, called with `ChatOpenAI`, with one tool, `search_document`. The model decides whether to search. Greetings and questions that are not about the document are answered without a search. `ChatOpenAI` drops DeepSeek's `reasoning_content`, which keeps the model's thinking out of the answer, and DeepSeek does not need it sent back between tool calls.

- **`search_document(query)`** embeds the query and reads the top `RETRIEVED_CHUNK_COUNT` chunks from Chroma, each with a relevance score from 0 to 1. Chunks scoring below `MIN_RELEVANCE_SCORE` are dropped, so weak matches and their images do not fill the prompt. If none remain, the tool tells the model nothing relevant was found. The model receives each result's text (a passage with its tables, or a figure description) under its page number. The full chunks are kept on the tool message as its artifact.
- **Parallel searches.** The model often makes two or more searches in one turn, and the agent runs them in parallel threads. Opening the Chroma store from several threads at the same time fails inside Chroma, so `generate.py` opens it once, when it is imported, and every search shares it.
- **Images.** A tool result can only hold text, so the `show_retrieved_images` middleware adds a message right after the tool results with the image of each retrieved figure, read from its `image_path`. It checks every tool result since the model's last call, so figures from parallel searches all reach the model. Do not move the images into the tool's return value.
- **Speed and limits.** A question takes two to four model calls and 1 to 6 seconds. `ChatOpenAI` and `OpenAIEmbeddings` retry rate limits and server errors (`MODEL_RETRY_ATTEMPTS`). Each search embeds its query with Jina, about 15 tokens, and one question makes one to five searches. A free Jina key allows 100 requests a minute.
- The system prompt tells the model to answer document questions only from the search results, to cite the page of each result it uses, and to say so when the results do not hold the answer. Citations use `page_number`, the page in the PDF.

`answer_question(question)` returns a dict with `question`, `answer`, `model`, `search_queries`, and `chunks`. `search_queries` is empty when the model answered without searching. Each chunk has `score`, `page_number`, `printed_page_number` (`""` when the page has none), `text`, and `image_path` (`None` for a text chunk). This is the function a FastAPI endpoint and UI will call later, so keep its return value JSON-friendly.

Running the file alone answers the `TEST_QUESTIONS` at the top of `generate.py` and prints the searches, the retrieved chunks, the model name, and the answer. Each test question has a comment with the answer and page it should find. The first needs no search. The last is not answered anywhere in the report, to check the model refuses instead of guessing.

There is no API or UI yet.

## Environment

Keys are read from `.env`:

- `LLAMA_CLOUD_API_KEY`: LlamaParse parsing
- `DEEPSEEK_API_KEY`: chart descriptions and answer generation
- `JINA_API_KEY`: embeddings
- `OPENROUTER_API_KEY`: not used since embeddings moved to Jina
- `GROQ_API_KEY`: not used since answer generation moved to DeepSeek
- `SARVAM_API_KEY`: not used. Sarvam's only image model, `gemma4`, needs beta access on `/v2` that this key does not have.
- `PINECONE_API_KEY`: not used by the current pipeline
