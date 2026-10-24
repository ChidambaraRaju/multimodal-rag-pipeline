"""All settings in one place. Change values here, nothing else needs editing.

API keys are read from .env: LLAMA_CLOUD_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY and GROQ_API_KEY.
"""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Document to ingest
DOCUMENT_PATH = PROJECT_ROOT / "2026-Annual-Report-Web.pdf"

# Where the LlamaParse result, its cropped images and the figure descriptions are saved, one folder per document.
# Ingestion reuses them, so the document is parsed only once. Delete the folder to parse again.
PARSED_DIRECTORY = PROJECT_ROOT / "parsed" / DOCUMENT_PATH.stem

# Where the Chroma vector store is saved. Each ingestion run replaces its contents.
VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "vector_store" / "chroma_db"

# LlamaParse tier. "agentic" has a vision model read each page, for 10 credits a page.
PARSE_TIER = "agentic"

# Models
IMAGE_DESCRIPTION_MODEL = "deepseek-flash"  # DeepSeek-V4.1-Flash on DeepSeek's own API, describes figures
DESCRIPTION_BASE_URL = "https://api.deepseek.com"  # OpenAI-compatible, so ChatOpenAI can call it
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"  # OpenRouter, text embeddings, 2048 dimensions
GENERATION_MODEL = "qwen/qwen3.8-27b"  # Groq, answers the question from the retrieved chunks

# How many chunks are retrieved for each question
RETRIEVED_CHUNK_COUNT = 3

# Chunks scoring below this (0 to 1) are not shown to the model, so weak matches do not fill the prompt
MIN_RELEVANCE_SCORE = 0.3

# Longest description the model may write, in tokens, its reasoning included.
# The embedding model accepts at most 4096 tokens per chunk.
DESCRIPTION_MAX_TOKENS = 3000

# How many times to retry a model call that is rate-limited or temporarily unavailable
MODEL_RETRY_ATTEMPTS = 8

# Chunking: each page is split at its headings, and a section shorter than this many characters joins the next one
MIN_CHUNK_CHARACTERS = 500
