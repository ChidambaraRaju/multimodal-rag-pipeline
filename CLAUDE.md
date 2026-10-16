@AGENTS.md

# Setting up a new VM

Follow this on a fresh Linux VM with an NVIDIA GPU and no sudo. It was first done on an A100 (a MIG slice, about 5 GB) with driver 550.90, which supports CUDA 12.4 at most. `requirements.txt` installs CPU-only or CUDA 13 packages, so the GPU steps below replace some of them inside `.venv` only. Do not change `requirements.txt` for this.

Check first: `nvidia-smi` (the "CUDA Version" is the highest the driver supports) and `sudo -n true` (fails when there is no sudo).

## 1. Python environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

If importing `multimodal_rag.ingest` fails with `libGL.so.1: cannot open shared object file`, swap OpenCV for the headless build of the same version:

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
from multimodal_rag import config
from multimodal_rag.ingest import parse_pdf
parse_pdf(config.DOCUMENT_PATH)
from unstructured_inference.models.base import get_model
from unstructured_inference.models import tables
print('layout:', get_model('yolox').model.get_providers())  # CUDAExecutionProvider first
print('tables:', tables.tables_agent.device)                 # cuda:0
"
```

onnxruntime falls back to CPU silently when a CUDA library fails to load. To see the real reason, create an `InferenceSession` with only `providers=['CUDAExecutionProvider']` and read the error. An "EP Error" about TensorRT on every run is harmless: TensorRT is not installed, so onnxruntime falls back to CUDA.

## 4. Keys

Create `.env` with `OPENROUTER_API_KEY` and `GROQ_API_KEY` (ask the user). To check them:

- OpenRouter: `GET https://openrouter.ai/api/v1/key` must return 200.
- Groq: `GET https://api.groq.com/openai/v1/models` must return 200.
