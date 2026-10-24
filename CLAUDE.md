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

`ingest.py` and `generate.py` run the LlamaParse pipeline from notebook 06, described in AGENTS.md. It was tested end to end on the attention paper (6 pages, about 60 credits): 17 text chunks and 3 figure descriptions. All 5 test questions were answered correctly with page citations, and the figure question got the figure's image. `DOCUMENT_PATH` still points at the attention paper.

`requirements.txt` now has `llama-cloud` and `langchain-text-splitters`, and no longer has `unstructured[all-docs]`, which only notebooks 01 and 03 need.

Notebooks so far:

- `03_annual_report_layout.ipynb`: the Unstructured problems listed above.
- `04_llamaparse_test.ipynb`: LlamaParse on 7 sample pages (4, 8, 13, 26, 128, 139, 142). It writes `notebooks/llamaparse_output/`, which git ignores: the parse result, the cropped images, the description caches and the test Chroma stores. On a new machine, run 04's parse first (about 70 credits), then 06.
- `05_llamaparse_ingestion.ipynb`: the first prototype. Pages packed into chunks of up to 3,000 characters, page images attached to every chunk, and table chunks embedded by a description. Superseded by 06, and it no longer runs against the current `ingest.py`.
- `06_heading_chunks_and_image_documents.ipynb`: the approach now in `ingest.py`. Each page split at its headings with a minimum size, tables embedded as raw HTML, and each figure its own document, described by `deepseek-flash` and skipped when it lies inside a larger crop. The right page came first for 6 of 6 test questions, as in 05.

Next steps:

1. Point `DOCUMENT_PATH` at the annual report and run `ingest.py` (a full parse is about 1,750 credits). Replace `TEST_QUESTIONS` with report questions that have known answers: an income statement number, the checkmark table, the page 8 diagram, and one the report cannot answer. The employee count is in the 10-K, so it does not qualify.
2. After the full parse, check:
   - the longest chunks (sort `open_vector_store().get()["documents"]` by length). With `merge_continued_tables` on, a multi-page table becomes one chunk on its first page, which could pass the embedding model's 4,096 tokens or Groq's 7,000 input tokens a minute;
   - how many photos are among the described figures;
   - whether tables continued over pages land where questions about them can find them.
3. Re-check `MIN_RELEVANCE_SCORE` on the full report. On the 7 sample pages, an unanswerable question still scored 0.375.
4. After that: the FastAPI backend, then the web UI.

Other context:

- **Models:** DeepSeek's own API (`https://api.deepseek.com`, key `DEEPSEEK_API_KEY`) is OpenAI-compatible, so `ChatOpenAI` calls it. There, `deepseek-flash` (DeepSeek-V4.1-Flash) reads text and images, and `deepseek-v4-pro` reads text only. `ingest.py` describes figures with `deepseek-flash`; it is a reasoning model, and its reasoning tokens count toward `max_tokens`. On OpenRouter, the DeepSeek models that take images are `deepseek-v4.1-flash` and `deepseek-v4-flash-vision-exp`. DeepSeek has no embedding model, so embeddings stay on Nemotron (OpenRouter).
- **Description speed:** `deepseek-flash` took 8 to 44 seconds a figure, against 19 to 50 seconds on the free Nemotron model. Only figures are described now, not tables.
- **Citations:** answers cite `page_number`, the page in the PDF, which every page has. The number printed on the page is kept in the chunk data (`printed_page_number`) but not given to the model; switch to it if the report's readers should look pages up by their printed numbers.
- **API backend, to plan for:**
  - run one uvicorn worker, because Chroma's on-disk store is not safe across processes;
  - build the agent once at startup;
  - return clear errors for Groq 413 and 429;
  - keep ingestion as a command-line step, not an endpoint.

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

Create `.env` with `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `LLAMA_CLOUD_API_KEY` and `DEEPSEEK_API_KEY` (ask the user). To check them:

- OpenRouter: `GET https://openrouter.ai/api/v1/key` must return 200.
- Groq: `GET https://api.groq.com/openai/v1/models` must return 200.
- DeepSeek: `GET https://api.deepseek.com/models` must return 200 and list `deepseek-flash`.
- LlamaParse: `GET https://api.cloud.llamaindex.ai/api/v1/projects` with the key as a Bearer token must return 200. It spends no credits. (`LlamaCloud().files.list()` is lazy and does not check the key.)

Notebooks are run headless with `nbclient` and `nbformat`, which are not in `requirements.txt`. Install them into `.venv` only.
