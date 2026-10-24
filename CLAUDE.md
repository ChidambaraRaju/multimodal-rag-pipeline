@AGENTS.md

# Parsing: moving from Unstructured to LlamaParse

`ingest.py` used to parse with local Unstructured `hi_res`. On the target annual report (175 pages: 12 wide spreads, pages 1 to 11 and 175, the rest normal pages) it had these problems, shown in `notebooks/03_annual_report_layout.ipynb`:

- Many lines come out twice (as `Title`/`NarrativeText` and again as `UncategorizedText`), though the PDF holds them once.
- A financial table's last column (pages 139 and 142) is split off into loose numbers that form junk chunks. The table HTML is complete, but its row labels are misaligned.
- Diagrams on the spreads (pages 4 and 8) are not detected as images. They become OCR fragments and never reach the vision model.
- The checkmark font in the director tables comes out as `ü`.
- Tesseract garbles math and symbols inside tables, for example `O(n? - d)` in the attention paper.

Reading order was not the problem: spreads are read left half then right half, and multi-column pages column by column. Page 128 has a broken font text layer that only OCR or a vision model can read.

LlamaParse (`agentic` tier, `llama_cloud` SDK) fixed all of these on a 7-page sample in `notebooks/04_llamaparse_test.ipynb`:
- complete HTML tables;
- every number matching the PDF text layer;
- real `✓` marks;
- page 128 read correctly;
- diagrams cropped as whole images (`images_to_save: ["layout"]`), page 8's also rewritten as a Mermaid flowchart.

It is also faster: 12 seconds for 7 pages. So LlamaParse is the parser going forward, and `ingest.py` now uses it.

- **Cost:** 10 credits a page on `agentic`, so the whole report is about 1,750 credits.
- **Account:** the account has 20,000 credits, given for opting into data sharing for model training. Use a separate account without that option for private documents.
- **Unstructured's hosted API was not an option:** sign-up requires a work email.

# Where we left off (October 24, 2026)

**State:** ingestion, generation, the FastAPI chatbot backend and the React web UI work end to end on the full NVIDIA 2026 annual report, and every question is traced in LangSmith.

- `ingest.py` runs the LlamaParse pipeline from notebook 06, described in AGENTS.md. The report's 175 pages became 377 text chunks and 12 chart documents.
- **Generation moved from Groq to DeepSeek** (`deepseek-flash`, the chart description model), to get away from Groq's free-tier rate limits. DeepSeek makes parallel searches, which needed three fixes in `generate.py`: the Chroma store is opened once and shared (opening it from several threads at once failed inside Chroma), `show_retrieved_images` shows the figures of every parallel search, and `answer_question()` lists a chunk found by two searches once.
- **Embeddings moved from Nemotron on OpenRouter to `jina-embeddings-v5-text-small`**, because free OpenRouter models allow only 50 requests a day. The store was re-embedded (7 minutes, no parse or description calls). The Nemotron store was backed up outside the repo for this session only.
- **All 6 test questions in `generate.py` are answered correctly with DeepSeek and Jina**, with page citations: the income statement numbers, the checkmark director table, the page 8 diagram (its image ranked first), the stock performance graph, and a refusal for smartphone revenue, which the report never mentions. The expected page ranked first for 4 of the 5 document questions. For the revenue question, page 23 (the business overview, with revenue but not net income) ranked first and page 139 second, and the answer used both.

## Done: the FastAPI backend

A chatbot over this one document, ingested ahead of time. Ingestion stays a command-line step, not an endpoint. `api.py` is described in AGENTS.md. The user's design decisions:

- **The server keeps the history, the UI does not.** The UI keeps only the conversation id, in a JavaScript variable, not localStorage, so a reload starts a new conversation. The user is fine with conversations vanishing on a reload or a server restart.
- **No LangGraph checkpointer.** A plain dict in memory keeps the last 3 questions and answers of each conversation, text only, and at most 1,000 conversations, dropping the least recently used.

Tested: an offline check with a stand-in for `answer_question()` covered the history cap, unknown ids, dropping the least recently used conversation, the 429/502 error JSON, the image route and the 422 for a missing question. Then a live run of uvicorn: a question about revenue, then "And the year before?" (the model searched for fiscal 2025 and gave $130,497 million revenue and $72,880 million net income, as on page 139), then "Is there a chart comparing that with the stock market?" (the page 123 graph, whose image the API served).

## Done: the web UI

The user has little frontend experience and did not want a plain HTML page, so the choice was React with Vite, TypeScript, Tailwind CSS v4 and shadcn/ui, in `frontend/` (described in AGENTS.md). Chainlit was the Python alternative offered. There is no CORS: Vite's dev server forwards API calls in development, and FastAPI serves the build.

Tested: `npm run build` and `npm run lint` pass. Headless Edge screenshots, driven by Playwright from a throwaway environment, covered the empty screen, the loading sweep, an answer with its citation marks, an opened page 139 excerpt (the income statement table), a follow-up that found the page 123 chart, the chart's dialog, phone width, and a faked 429 with "Ask again". There were no console errors, and the Vite dev server forwarded `/health` and a chart image to the API.

Lessons:
- `shadcn init -t vite` runs `git init` inside the new folder. That nested repository was deleted (it held only the scaffold's first commit), so the main repo tracks `frontend/`.
- On Windows, stopping `npm run dev` leaves Vite's `node` process listening on port 5173. Find it with `netstat -ano` and stop it.
- uvicorn keeps running the Python code it started with (unless started with `--reload`), but serves `frontend/dist` straight from disk. After changing both, restart the server, or a new UI meets an old API. The user may have their own server on port 8000; test on another port (such as 8001) instead of stopping it.
- A page often contributes several chunks to one answer (page 123 gave three: `page123-2`, its chart, `page123-1`). The UI shows them under one tab in reading order, which is why chunks carry their `id`.
- To see the UI without a browser tool: `uv run --no-project --with playwright python <script>` with `p.chromium.launch(channel="msedge")` uses the installed Edge, so no browser download is needed and nothing is added to the project.

## Done: observability with LangSmith

The user compared LangSmith and Pydantic Logfire, both on their free plans, and chose LangSmith (the comparison is in AGENTS.md, "Observability"). It needed no new package: three settings in `.env` and the conversation id passed as `thread_id`. Tested with a question and "And the year before?" in one conversation: both traces reached the `multimodal-rag-pipeline` project and were grouped into one thread; the first had 11 runs (chain, llm, tool) and 4,669 tokens.

Lessons:
- The project need not be created by hand; the first trace creates it. The key works on the default US endpoint, so `LANGSMITH_ENDPOINT` is not set.
- In a script, call `langchain_core.tracers.langchain.wait_for_all_tracers()` before reading traces back, and expect a few seconds before they can be queried; a just-finished trace may still have no end time.
- `Client.list_runs()` and `get_run_url()` are deprecated in `langsmith` 0.14 (removal after January 31, 2027); use `client.runs.query()` and `client.runs.get_url()`.
- A uvicorn server started before the `.env` change does not trace; restart it.
- Traces were first listed as "LangGraph", LangGraph's default name for the agent's graph. `build_agent()` now passes `name="annual_report_agent"` to `create_agent`, so new traces carry that name; earlier ones keep "LangGraph".

## Next

Nothing is decided. Candidates:
- **Streaming:** show the answer as it is written, and the searches as they happen ("Searching: fiscal 2025 revenue"). It needs a streaming endpoint in FastAPI (for example server-sent events with the agent's `stream`) and a UI that reads it. Measured end to end, answers take 3 to 7 seconds (3 with one search, 6.5 with three), so this matters less than first thought, when the wait was guessed at 5 to 20 seconds.
- **Check the Jina balance** on its dashboard: the re-embed and the tests since spent an estimated 300,000 tokens of the one-time grant.
- **The model's own arithmetic** (see Open issues): one system prompt line.
- **The evaluation set** of 25 to 30 questions, as a LangSmith dataset: run the agent over it as an experiment, score the answer and the retrieved pages, and compare experiments after each change.

## Open issues

- **Jina's free tokens are a one-time grant.** Embeddings moved from Nemotron on OpenRouter, whose free models allow only 50 requests a day, to `jina-embeddings-v5-text-small`. The key's balance is on Jina's dashboard (a third-party source says new keys get 10 million tokens; older ones say 1 million). Each full re-embed spends about 280,000 tokens (an estimate at 2.6 characters a token), and each search about 15. When the grant is used up, the key needs paid tokens (a third-party source says Jina then answers 402); a full re-embed would then cost about a cent.
- **Groq's limits no longer apply.** Its 7,000 input tokens a minute made a question that retrieves the three largest chunks (about 8,100 tokens, all financial tables) fail with a 413. With DeepSeek, `RETRIEVED_CHUNK_COUNT` could be raised if answers miss context.
- **Removed dependencies:** `langchain-groq`, `langchain-openrouter` and `openrouter` were removed, since only the old notebooks 01 and `multi_modal_rag` import them; install them separately to run those. `GROQ_API_KEY` and `OPENROUTER_API_KEY` in `.env` are unused.
- **Cost:** DeepSeek is paid per token. A question took 2 to 4 model calls, 400 to 5,000 input tokens and 50 to 350 output tokens each.
- **The relevance cutoff only drops noise.** With Jina, the expected chunks scored 0.50 to 0.82, an off-topic question ("What is the capital of France?") at most 0.22, and the smartphone question, which has no answer, 0.66 at the top. So `MIN_RELEVANCE_SCORE` (0.3) still sits between the noise and the answers, but cannot detect unanswerable questions; the system prompt does that, and it worked. The director question's right chunks score lowest (0.35 to 0.56), so a cutoff above about 0.35 would start dropping them.
- **The refusal wording varies.** For the smartphone question DeepSeek once said the report "does not report a separate revenue figure for smartphone sales" instead of the system prompt's exact sentence. Since the prompt names the report (`config.DOCUMENT_TITLE`), it has used the exact sentence, after a short explanation. The meaning was right both times, but a UI that looks for the exact sentence could miss it.
- **Identity.** The system prompt used to say only "a document", so "Who are you?" got "an assistant that answers questions about a document you've provided". It now names the report and says how to introduce itself; "Who are you?" gets "I'm an assistant that answers questions about NVIDIA's 2026 Annual Report, using its text, financial tables and charts, and citing the pages I use."
- **Unchecked:** whether tables continued over pages (merged onto their first page) are found by questions about them.
- **Later:** an evaluation set of 25 to 30 report questions with their expected page and answer, to measure how often the right page is retrieved and the answer is correct.
- **The model adds its own arithmetic.** For the stock question, Qwen called NVIDIA's return "7.2 times" the S&P 500's, which is the ratio of ending values, not of returns. DeepSeek gave the returns correctly (about 1,349% against 100%), but still adds figures the report does not state, such as "14.5x the initial investment" and "outperformed the S&P 500 by more than 7x". In one API test it called the ending values of $100 ($1,448.75 and $200.33) a "~1,449% versus ~200%" cumulative return, which is wrong (the returns are about 1,349% and 100%). A system prompt line such as "quote figures as the document states them, and do not compute new ones" may help.
- **Small things:**
  - `main.py` is the unused `uv init` stub.
  - `pyproject.toml` says `requires-python >= 3.13`, while the VM setup below uses 3.12. That setup installs from `requirements.txt`, which works, but `uv sync` would refuse.
  - The PDFs are committed to the public repo; the user may prefer linking to their source.

## The ingested data exists only on this machine

`parsed/2026-Annual-Report-Web/` (the parse result, 202 cropped images, 12 chart descriptions) and `vector_store/` are git-ignored. They exist only in `D:\Claude Code Projects\multimodal-rag-pipeline` on the user's Windows machine. There, `python -m multimodal_rag.ingest` spends no LlamaParse credits and makes no DeepSeek calls: it reuses the saved parse and descriptions and only re-embeds, which takes about 7 minutes and about 280,000 Jina tokens.

On another machine (the VM, a deployment server), do not run ingestion straight away. With no `parsed/` folder it parses the whole report again, for about 1,750 credits. Copy `parsed/` (or `vector_store/`) over instead. Or recover the parse from its LlamaParse job, which costs nothing, if LlamaParse still keeps the job (its retention period is not published):

```python
import json
from llama_cloud import LlamaCloud
from multimodal_rag.parse import RESULT_PATH  # importing it also loads .env

job = LlamaCloud().parsing.get("pjb-uwmj2thw29hu9hp0noial2x4p9k1", expand=["markdown", "metadata", "images_content_metadata"])
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(job.model_dump(mode="json"), indent=1))
```

Then `python -m multimodal_rag.ingest` downloads the images and describes the 12 charts (12 DeepSeek calls).

`frontend/node_modules` and `frontend/dist` are git-ignored too: run `npm install` and `npm run build` in `frontend/` on a new machine.

## Local development (Windows)

- Windows 11, `.venv` with Python 3.13. Packages are managed with uv (`pyproject.toml`, `uv.lock`), and `requirements.txt` is kept in step: add a package with `uv add "<package>==<version>"` and the same pin in `requirements.txt`. `unstructured[all-docs]` is no longer a dependency; only notebooks 01 and 03 need it.
- Run modules from the project root with the venv's Python: `.venv/Scripts/python.exe -u -m multimodal_rag.ingest`. `-u` shows progress when output goes to a file. Start the API the same way: `.venv/Scripts/python.exe -m uvicorn multimodal_rag.api:app --port 8000`.
- The UI needs Node.js (22 is installed). Run `npm install` once in `frontend/`, then `npm run dev` for development or `npm run build` for the version the API serves at http://127.0.0.1:8000/.
- Set `PYTHONUTF8=1` when output is redirected or piped. Otherwise Python writes cp1252, and printing a `✓` from the director table raises `UnicodeEncodeError`.
- `nbclient` and `nbformat` are installed in `.venv` only, to run notebooks headless. `uv remove` syncs `.venv` exactly to the lock file and uninstalls them (`uv add` kept them), so after a `uv remove` reinstall them: `uv pip install --python .venv/Scripts/python.exe "nbclient==0.11.0" "nbformat==5.11.1"`.

## Lessons from the full run

- **Parse:** job `pjb-uwmj2thw29hu9hp0noial2x4p9k1`, 175 pages, 202 crops (100 tables, 90 images, 12 charts). 152 pages have a printed page number. The first run's parse succeeded, but one image download dropped its connection, and the result was lost because it was saved only after the images. The job was recovered with `LlamaCloud().parsing.get(job_id, expand=[...])` at no extra cost, and `parse.py` now saves the result first and downloads any missing images on every run.
- **Charts only:** of the 90 `image` crops, 66 were icons about 20 to 25 points across (the skill icons on pages 35 to 41), and most of the rest were photos, logos, signatures and product shots. So only the 12 `chart` crops are described and stored. Two `image` crops were real diagrams and are dropped with the rest: page 4's DSX diagram, whose labels are all in the page text, and page 3's AI five-layer stack, whose lists of application areas and model types appear nowhere else. To keep one, add an exception for its page in `pick_charts()`.
- **Embeddings:** OpenRouter's Nemotron API took at most 256 texts per request. Jina's free key takes at most 100,000 tokens a minute, so `store.py` now sends 64 documents a minute.

Notebooks so far:

- `03_annual_report_layout.ipynb`: the Unstructured problems listed above.
- `04_llamaparse_test.ipynb`: LlamaParse on 7 sample pages (4, 8, 13, 26, 128, 139, 142). It writes `notebooks/llamaparse_output/`, which git ignores: the parse result, the cropped images, the description caches and the test Chroma stores. On a new machine, run 04's parse first (about 70 credits), then 06.
- `05_llamaparse_ingestion.ipynb`: the first prototype. Pages packed into chunks of up to 3,000 characters, page images attached to every chunk, and table chunks embedded by a description. Superseded by 06, and it no longer runs against the current `ingest.py`.
- `06_heading_chunks_and_image_documents.ipynb`: the approach now in `ingest.py`. Each page split at its headings with a minimum size, tables embedded as raw HTML, and each figure its own document, described by `deepseek-flash` and skipped when it lies inside a larger crop. The right page came first for 6 of 6 test questions, as in 05. `ingest.py` has since narrowed the figures to `chart` crops only, which made the nested-crop check unnecessary.

Other context:

- **Models:** DeepSeek's own API (`https://api.deepseek.com`, key `DEEPSEEK_API_KEY`) is OpenAI-compatible, so `ChatOpenAI` calls it. There, `deepseek-flash` (DeepSeek-V4.1-Flash) reads text and images, and `deepseek-v4-pro` reads text only. `ingest.py` describes charts with `deepseek-flash`, and `generate.py` answers with it; it is a reasoning model, and its reasoning tokens count toward `max_tokens`. On OpenRouter, the DeepSeek models that take images are `deepseek-v4.1-flash` and `deepseek-v4-flash-vision-exp`. DeepSeek has no embedding model, so embeddings are on Jina.
- **Description speed:** `deepseek-flash` took 8 to 44 seconds a chart, against 19 to 50 seconds on the free Nemotron model. Only charts are described, not tables or other images.
- **Citations:** answers cite `page_number`, the page in the PDF, which every page has. The number printed on the page is kept in the chunk data (`printed_page_number`) but not given to the model; switch to it if the report's readers should look pages up by their printed numbers.
- **Working style:** the user reviews each step and makes the design decisions; propose and explain before larger changes. Keep the markdown files current after every change, and commit directly on `main`, pushing only when asked.

# Setting up a new VM

Follow this on a fresh Linux VM with an NVIDIA GPU and no sudo. It was first done on an A100 (a MIG slice, about 5 GB) with driver 550.90, which supports CUDA 12.4 at most. `requirements.txt` installs CPU-only or CUDA 13 packages, so the GPU steps below replace some of them inside `.venv` only. Do not change `requirements.txt` for this.

Check first: `nvidia-smi` (the "CUDA Version" is the highest the driver supports) and `sudo -n true` (fails when there is no sudo).

## 1. Python environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Sections 2 and 3 and the OpenCV fix below are only for notebooks 01 and 03, which use Unstructured. `ingest.py` no longer does, and `unstructured[all-docs]` is no longer in `requirements.txt`, so install it separately for those notebooks.

If importing `unstructured` fails with `libGL.so.1: cannot open shared object file`, swap OpenCV for the headless build of the same version:

```bash
uv pip uninstall --python .venv/bin/python opencv-python
uv pip install --python .venv/bin/python opencv-python-headless==<same version>
```

## 2. PDF tools without sudo

Unstructured needs poppler, tesseract and libmagic. Install them from conda-forge into the home directory:

```bash
/opt/conda/bin/conda create -y -p ~/.local/pdf-tools -c conda-forge --override-channels poppler tesseract libmagic
for b in pdftoppm pdfinfo pdftotext tesseract; do ln -sf ~/.local/pdf-tools/bin/$b ~/.local/bin/$b; done
mkdir -p ~/.local/lib/libmagic
ln -sf ~/.local/pdf-tools/lib/libmagic.so.1 ~/.local/lib/libmagic/libmagic.so.1
```

`~/.local/bin` must be on PATH. Then add to `~/.bashrc`:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/libmagic${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Only libmagic goes on `LD_LIBRARY_PATH`. Do not add the whole `~/.local/pdf-tools/lib`, because its libraries would shadow system ones.

## 3. GPU for ingestion

Two parts of `partition_pdf` (strategy `hi_res`) can use the GPU:

- Layout detection (YOLOX) runs on onnxruntime. The PyPI `onnxruntime` is CPU only.
- Table structure (Table Transformer) runs on PyTorch and uses CUDA when `torch.cuda.is_available()`.

Tesseract OCR is always CPU only.

**PyTorch.** Use a CUDA 12 build, at least cu128. A CUDA 12.x build runs on a 12.4 driver through CUDA minor version compatibility. Use cu128 or newer, not cu126, because onnxruntime's CUDA 12 build needs `cudaLibraryGetKernel`, which the 12.6 runtime lacks. Check https://download.pytorch.org/whl/cu129/torch/ for the newest version. In October 2026 that was 2.13.0:

```bash
uv pip install --python .venv/bin/python "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
  --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

**onnxruntime.** `onnxruntime-gpu` on PyPI is built for CUDA 13. The CUDA 12 builds are on a separate feed: https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/onnxruntime-gpu/ (1.29.0 was the newest; unstructured-inference needs >=1.25.0). Installing by version with that feed as an index still pulled PyPI's CUDA 13 wheel of the same version. So download the cp312 manylinux x86_64 wheel from the feed page and install the file directly:

```bash
uv pip uninstall --python .venv/bin/python onnxruntime
uv pip install --python .venv/bin/python --no-deps --reinstall ./onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_<...>_x86_64.whl
```

The wheel file must keep its full original name, or uv rejects it. Confirm it is the CUDA 12 build:

```bash
readelf -d .venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so | grep cublasLt
# must show libcublasLt.so.12, not .so.13
```

**Verify.** The parse must show both models on the GPU:

```bash
.venv/bin/python -c "
from unstructured.partition.pdf import partition_pdf
partition_pdf(filename='attention-is-all-you-need-first6.pdf', strategy='hi_res', infer_table_structure=True)
from unstructured_inference.models.base import get_model
from unstructured_inference.models import tables
print('layout:', get_model('yolox').model.get_providers())  # CUDAExecutionProvider first
print('tables:', tables.tables_agent.device)                 # cuda:0
"
```

onnxruntime falls back to CPU silently when a CUDA library fails to load. To see the real reason, create an `InferenceSession` with only `providers=['CUDAExecutionProvider']` and read the error. An "EP Error" about TensorRT on every run is harmless: TensorRT is not installed, so onnxruntime falls back to CUDA.

## 4. Keys

Create `.env` with `JINA_API_KEY`, `LLAMA_CLOUD_API_KEY` and `DEEPSEEK_API_KEY`, and for tracing `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT=multimodal-rag-pipeline` (ask the user). To check them:

- LangSmith: `GET https://api.smith.langchain.com/api/v1/sessions?limit=1` with the key in an `x-api-key` header must return 200.

- Jina: `POST https://api.jina.ai/v1/embeddings` with the key as a Bearer token and `{"model": "jina-embeddings-v5-text-small", "task": "retrieval.query", "input": ["test"]}` must return 200. It spends a few tokens.
- DeepSeek: `GET https://api.deepseek.com/models` must return 200 and list `deepseek-flash`.
- LlamaParse: `GET https://api.cloud.llamaindex.ai/api/v1/projects` with the key as a Bearer token must return 200. It spends no credits. (`LlamaCloud().files.list()` is lazy and does not check the key.)

Notebooks are run headless with `nbclient` and `nbformat`, which are not in `requirements.txt`. Install them into `.venv` only.

## 5. Web UI

Not tried on a VM yet. The UI needs Node.js: check `node --version` first. Without sudo, conda-forge's `nodejs` package installed into the home directory, like the PDF tools in section 2, should work. Then run `npm install` and `npm run build` in `frontend/`, and the API serves the UI at `/`. To reach it from another machine, start uvicorn with `--host 0.0.0.0` or forward the port over SSH.
