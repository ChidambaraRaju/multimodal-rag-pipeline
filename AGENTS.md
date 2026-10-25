# Multimodal RAG

Single-document RAG over a PDF with text, tables, and images. The document is `2026-Annual-Report-Web.pdf`, NVIDIA's 2026 annual report (175 pages), and it is fully ingested. `attention-is-all-you-need-first6.pdf`, the first 6 pages of the attention paper, was the earlier test document; the attention paper's PDFs are git-ignored and exist only on the user's machine. Scale-out (multiple companies, periods, and reports) is out of scope until one document works end to end.

Keep the code simple and easy to read. This is the priority. Prefer one plain function per step over abstractions, and do not add options nobody asked for.

Always use the LangChain MCP servers before calling any LangChain, LangGraph, or LangSmith function, and follow the current docs and API reference those servers return.

- `docs-langchain` (`https://docs.langchain.com/mcp`): guides and how-tos
- `reference-langchain` (`https://reference.langchain.com/mcp`): classes, methods, and signatures

Likewise, use the `docs-llamaindex` MCP server (`https://developers.llamaindex.ai/mcp`) before calling the LlamaParse (`llama_cloud`) SDK. Its `grep_docs` tool needs a `section`; use `llama-cloud` for LlamaParse.

Experiments belong in notebooks. The user reviews each notebook run and decides whether the approach stays. Move an approach into `.py` modules only after that decision.

## Layout

- `multimodal_rag/config.py`: every setting (document path and title, parse and vector store folders, LlamaParse tier, model names, token limit, embedding batch size, retry count, minimum chunk size, number of retrieved chunks, chat history turns, conversation limit). Change settings here and nowhere else. It also loads `.env`. The web UI's document title and example questions are the one exception: they live in `frontend/src/lib/document.ts`.
- `multimodal_rag/store.py`: the Chroma vector store and the embedding model, shared by ingestion and generation, and `save_documents()`, which replaces the stored contents.
- `multimodal_rag/ingest.py`: runs the ingestion steps in order. Run it from the project root with `python -m multimodal_rag.ingest`. Each step lives in its own module:
  - `multimodal_rag/parse.py`: LlamaParse, the saved parse result and the cropped images.
  - `multimodal_rag/chunking.py`: splitting each page into text chunks.
  - `multimodal_rag/figures.py`: picking the charts among the crops and describing them.
- `multimodal_rag/generate.py`: the agent that answers questions, with document search as its tool. Run it from the project root with `python -m multimodal_rag.generate` to try the test questions.
- `multimodal_rag/api.py`: the FastAPI chatbot backend, which keeps each conversation's recent turns in memory. Run it from the project root with one worker: `uvicorn multimodal_rag.api:app`.
- `frontend/`: the web UI, a React app built with Vite, TypeScript, Tailwind CSS v4 and shadcn/ui. See "Web UI" below.
- `notebooks/multi_modal_rag.ipynb`: the notebook the pipeline came from.
- `notebooks/01_docling_parse.ipynb`: an earlier Docling and Pinecone experiment. The pipeline no longer uses either.
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
- **Speed and limits.** A question takes two to four model calls and, measured end to end with real search, 3 to 7 seconds (about 3 for one search, 6.5 for three). `ChatOpenAI` and `OpenAIEmbeddings` retry rate limits and server errors (`MODEL_RETRY_ATTEMPTS`). Each search embeds its query with Jina, about 15 tokens, and one question makes one to five searches. A free Jina key allows 100 requests a minute.
- The system prompt names the document by `config.DOCUMENT_TITLE` ("NVIDIA's 2026 Annual Report"). It tells the model to introduce itself as an assistant for that report when asked who it is or what it can do, to answer questions about the report only from the search results, to cite the page of each result it uses, and to say "I don't have enough information to answer that question based on NVIDIA's 2026 Annual Report." when the results do not hold the answer. Citations use `page_number`, the page in the PDF.

`answer_question(agent, question, history=(), thread_id=None)` takes the agent from `build_agent()`, built once, the earlier turns of the conversation, `[{"question": ..., "answer": ...}]`, oldest first, and optionally the conversation's id, which goes to LangSmith as `thread_id` metadata (see "Observability"). The earlier turns go to the model as plain user and assistant messages, without their search results or figures, so later calls stay small. The system prompt tells the model to use them to understand a follow-up but to search again for any facts it needs. It returns a dict with `question`, `answer`, `model`, `search_queries`, and `chunks`. `search_queries` is empty when the model answered without searching. Each chunk has `id` (its id in Chroma: `page139-2` for the 2nd text chunk of page 139, as `chunking.py` names them, or a figure's image name such as `page_123_chart_1_v2`), `score`, `page_number`, `printed_page_number` (`""` when the page has none), `text`, and `image_path` (`None` for a text chunk). A chunk found by two searches is listed once, by its id. The API returns this dict, so keep it JSON-friendly.

Running the file alone answers the `TEST_QUESTIONS` at the top of `generate.py` and prints the searches, the retrieved chunks, the model name, and the answer. Each test question has a comment with the answer and page it should find. The first needs no search. The last is not answered anywhere in the report, to check the model refuses instead of guessing.

## API

`api.py` is a FastAPI app over the query path. The agent is built once, at startup.

- **`POST /ask`** takes `{"question": ..., "conversation_id": ...}` and returns the `answer_question()` dict with the `conversation_id` added. The first question of a conversation leaves the id out, and the server creates one (a UUID). The UI keeps only that id, not the history, so a page reload starts a new conversation. A missing or unknown id, for example after a server restart, starts a new conversation instead of failing.
- **Conversations** live in a plain dict in memory, not a LangGraph checkpointer: conversation id → its last `CHAT_HISTORY_TURNS` (3) turns, question and answer text only. A conversation moves to the end of the dict when used, and past `MAX_CONVERSATIONS` (1,000) the least recently used one is dropped. They are lost when the server restarts.
- **Images.** The cropped images are served at their `image_path`, so the UI shows a chart at `/` + its `image_path`. Only the images folder is served, not the parse result.
- **`GET /health`** returns `{"status": "ok"}`.
- **Errors.** DeepSeek and Jina are both called through the OpenAI client, so a call that still fails after its retries raises an `openai.APIError`. The handler returns `{"error", "service", "status"}`: 429 for a rate limit, 502 for anything else.
- **One worker only.** The conversations are in one process's memory, and Chroma's on-disk store is not safe across processes.
- **No CORS.** The page and the API always share one origin: in development Vite's dev server forwards `/ask`, `/health` and `/parsed` to the API, and the built UI in `frontend/dist` is served by FastAPI itself at `/`. That mount comes last in `api.py`, so the API routes take precedence, and it is skipped when `frontend/dist` does not exist.

## Observability

LangSmith traces every LangChain run, turned on by three settings in `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT=multimodal-rag-pipeline`. The project was created automatically by the first trace. No code sets it up: the `langsmith` package comes with LangChain and reads these settings, so it is not pinned in `requirements.txt`. Set `LANGSMITH_TRACING=false` to stop tracing.

- **One question is one trace**, named `annual_report_agent`, the `name` that `build_agent()` gives `create_agent` (without it, LangGraph calls every trace `LangGraph`). It holds the model calls with their messages, chart images, token counts and cost, each `search_document` call with its query and returned chunks, and the answer. The Jina embedding calls and the FastAPI request are not traced on their own.
- **Threads.** The API passes its `conversation_id` to `answer_question()` as `thread_id`, so the questions of one conversation are grouped in the project's Threads tab.
- **Cost.** LangSmith prices each model call from its token counts, by matching the call's `ls_model_name` and `ls_provider` against its own price table, which lists `deepseek-flash` under the provider `deepseek`, with a lower price for cached input tokens. `ChatOpenAI` reports the provider as `openai`, which matches no entry, so both models (in `generate.py` and `figures.py`) pass `metadata={"ls_provider": "deepseek"}`. A question with one search cost about $0.0003. LangSmith does not recalculate traces already logged, so traces from before October 25, 2026 show no cost.
- **Ingestion** is traced too while tracing is on: each chart description is a DeepSeek call through `ChatOpenAI`.
- **Free plan:** 5,000 traces a month (tracing stops at the limit until a card is added), kept 14 days, one seat. Pydantic Logfire was the alternative considered: it sees the whole server (HTTP requests, every DeepSeek and Jina call) and keeps 30 days, but needs its own package and setup code, and reads the agent's steps through LangSmith's OpenTelemetry export anyway. LangSmith was chosen because it is made for LangChain agents and also has datasets and experiments for evaluation.

## Web UI

`frontend/` was created with `shadcn init -t vite` (preset `nova`, Radix components) and keeps the shadcn layout: its components in `src/components/ui/`, the `cn` class helper from `src/lib/utils.ts`, and the theme in `src/index.css`. Add shadcn components with `npx shadcn@latest add <name>` from `frontend/`. Run `npm run dev` there while working (http://localhost:5173), and `npm run build` for the version FastAPI serves; `npm run lint` checks the code.

- `src/App.tsx`: the conversation. It keeps the turns on screen and only the `conversation_id` in a ref, never in localStorage, so a reload or "New chat" starts a new conversation. "Ask again" resends a failed question.
- `src/lib/api.ts`: `ask()` calls `POST /ask` and turns a 429, a 502 or an unreachable server into a message for the person asking, naming DeepSeek or Jina from the error's `service`.
- `src/lib/document.ts`: the document's title, page count and the example questions on the empty screen. Update them whenever `config.DOCUMENT_PATH` and `config.DOCUMENT_TITLE` change.
- `src/components/exchange.tsx`: one question with its loading state, error or answer.
- `src/components/markdown.tsx`: renders answers and page excerpts with `react-markdown`, `remark-gfm` for tables, and `rehype-raw` then `rehype-sanitize`, because excerpts hold LlamaParse's HTML tables. `linkCitations()` turns "page 139" and "pages 23, 167" in an answer into citation marks; a mark whose page was retrieved opens that page's excerpt.
- `src/components/sources.tsx`: under each answer, the retrieved figures (opening larger in a dialog), one tab per retrieved page however many chunks came from it, and the open page's excerpt with its printed page number and highest relevance score. The excerpt shows every retrieved chunk of that page, in reading order (text chunks by the number in their id, then figure descriptions), with a divider between chunks.
- `src/components/composer.tsx`: the question box. Enter asks, Shift+Enter starts a new line.

Design, chosen for reading a report and checking its evidence: cool paper white (`#f5f6f2`), blue-slate ink (`#1c2534`), ink blue for actions (`#2c47c5`), and a highlighter yellow (`--marker`, `#f3df5a`) used only for page citations, page tabs and the loading sweep. Answers are set in Source Serif 4 (the report's voice), questions and controls in Public Sans (the asker's voice), and tables use tabular figures so columns of numbers line up. The page is a single left-aligned reading column, laid out as a transcript rather than chat bubbles. There is no dark mode.

## Environment

Keys are read from `.env`:

- `LLAMA_CLOUD_API_KEY`: LlamaParse parsing
- `DEEPSEEK_API_KEY`: chart descriptions and answer generation
- `JINA_API_KEY`: embeddings
- `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`: LangSmith tracing (see "Observability")
- `OPENROUTER_API_KEY`: not used since embeddings moved to Jina
- `GROQ_API_KEY`: not used since answer generation moved to DeepSeek
- `SARVAM_API_KEY`: not used. Sarvam's only image model, `gemma4`, needs beta access on `/v2` that this key does not have.
- `PINECONE_API_KEY`: not used by the current pipeline
