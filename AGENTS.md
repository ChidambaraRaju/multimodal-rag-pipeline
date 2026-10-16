# Multimodal RAG

Single-document RAG over a PDF with text, tables, and images. The target document is `2026-Annual-Report-Web.pdf`. Testing currently runs on `attention-is-all-you-need-first6.pdf`, the first 6 pages of the attention paper. Scale-out (multiple companies, periods, and reports) is out of scope until one document works end to end.

Keep the code simple and easy to read. This is the priority. Prefer one plain function per step over abstractions, and do not add options nobody asked for.

Always use the LangChain MCP servers before calling any LangChain, LangGraph, or LangSmith function, and follow the current docs and API reference those servers return.

- `docs-langchain` (`https://docs.langchain.com/mcp`): guides and how-tos
- `reference-langchain` (`https://reference.langchain.com/mcp`): classes, methods, and signatures

Experiments belong in notebooks. The user reviews each notebook run and decides whether the approach stays. Move an approach into `.py` modules only after that decision.

## Layout

- `multimodal_rag/config.py`: every setting (document path, vector store folder, model names, token limit, retry count, chunk sizes, number of retrieved chunks). Change settings here and nowhere else. It also loads `.env`.
- `multimodal_rag/store.py`: the Chroma vector store and the embedding model, shared by ingestion and generation.
- `multimodal_rag/ingest.py`: the whole ingestion pipeline. Run it from the project root with `python -m multimodal_rag.ingest`.
- `multimodal_rag/generate.py`: the agent that answers questions, with document search as its tool. Run it from the project root with `python -m multimodal_rag.generate` to try the test questions.
- `notebooks/multi_modal_rag.ipynb`: the notebook the pipeline came from.
- `notebooks/01_docling_parse.ipynb`, `notebooks/02_native_docling_loader.ipynb`: earlier Docling and Pinecone experiments. The pipeline no longer uses either.

## Ingestion

`ingest.py` runs four steps in order:

1. **Parse.** Unstructured `partition_pdf` with the `hi_res` strategy. Tables are kept as HTML and images as base64 on the element.
2. **Chunk.** `chunk_by_title` groups elements by section title, using the chunk sizes in `config.py`.
3. **Describe.** A chunk that holds a table or an image is sent to the image description model with its text, table HTML, and images. The model returns a searchable description. Chunks with only text skip this step.
4. **Store.** Chunks are embedded and saved in a local Chroma store. Each run replaces the stored contents.

| Chunk | What gets embedded | What is kept for generation |
|---|---|---|
| Text only | The chunk text | The chunk text |
| Has a table or image | The AI description | The chunk text, table HTML, and base64 images, in the `original_content` metadata field |

Models, all on OpenRouter and all set in `config.py`:

- Image description: `nvidia/nemotron-3.5-super-vl:free`. The free tier is sometimes rate-limited or unavailable, so calls are retried. `google/gemma-4-31b-it:free` was tried first and dropped because its free pool stayed rate-limited. `dots-studio/dots-3-note-preview:free` came next and was replaced by Nemotron.
- Embeddings: `nvidia/nemotron-3-embed-1b:free`, text only, 2048 dimensions. It needs `input_type=passage` for chunks and `input_type=query` for questions, which `NemotronEmbeddings` in `store.py` handles.

If a description call still fails after the retries, the chunk is stored with its raw text and the failure is printed.

## Query path

Generation is agentic. `generate.py` builds an agent with LangChain's `create_agent`: `qwen/qwen3.8-27b` on Groq with one tool, `search_document`. The model decides whether to search. Greetings and questions that are not about the document are answered without a search.

- **`search_document(query)`** embeds the query and reads the top `RETRIEVED_CHUNK_COUNT` chunks from Chroma, each with a relevance score from 0 to 1. Chunks scoring below `MIN_RELEVANCE_SCORE` are dropped, so weak matches and their images do not push a request past Groq's free-tier limit of 7,000 input tokens per minute. If none remain, the tool tells the model nothing relevant was found. The model receives the chunks' original text and table HTML. The full chunks are kept on the tool message as its artifact.
- **Images.** Groq only accepts a plain string as a tool result, so the `show_retrieved_images` middleware adds a message with the retrieved chunks' images right after the tool result. Do not move the images into the tool's return value.
- **Rate limits.** Groq's free tier allows about 1,000 output tokens per minute for this model, and one question takes two or more model calls. `ModelRetryMiddleware` waits and retries on a rate limit, so several questions in a row are slow.
- The system prompt tells the model to answer document questions only from the search results and to say so when they do not hold the answer.

`answer_question(question)` returns a dict with `question`, `answer`, `model`, `search_queries`, and `chunks`. `search_queries` is empty when the model answered without searching. Each chunk has `score`, `text`, `tables_html`, and `images_base64`. This is the function a FastAPI endpoint and UI will call later, so keep its return value JSON-friendly.

Running the file alone answers the `TEST_QUESTIONS` at the top of `generate.py` and prints the searches, the retrieved chunks, the model name, and the answer. The first test question needs no search. The last is not covered by the first 6 pages, to check the model refuses instead of guessing.

There is no API or UI yet.

## Environment

Keys are read from `.env`:

- `OPENROUTER_API_KEY`: image descriptions and embeddings
- `GROQ_API_KEY`: answer generation
- `SARVAM_API_KEY`: not used. Sarvam's only image model, `gemma4`, needs beta access on `/v2` that this key does not have.
- `PINECONE_API_KEY`: not used by the current pipeline
