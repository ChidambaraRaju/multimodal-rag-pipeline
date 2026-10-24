"""All settings in one place. Change values here, nothing else needs editing.

API keys are read from .env: LLAMA_CLOUD_API_KEY, DEEPSEEK_API_KEY and JINA_API_KEY.
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
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # DeepSeek's own API, OpenAI-compatible, so ChatOpenAI can call it
IMAGE_DESCRIPTION_MODEL = "deepseek-flash"  # DeepSeek-V4.1-Flash, describes figures
EMBEDDING_MODEL = "jina-embeddings-v5-text-small"  # Jina, text embeddings, 1024 dimensions
GENERATION_MODEL = "deepseek-flash"  # DeepSeek-V4.1-Flash, reads text and images, answers from the retrieved chunks

# How many chunks are retrieved for each question
RETRIEVED_CHUNK_COUNT = 3

# Chunks scoring below this (0 to 1) are not shown to the model, so weak matches do not fill the prompt
MIN_RELEVANCE_SCORE = 0.3

# Longest description the model may write, in tokens, its reasoning included.
DESCRIPTION_MAX_TOKENS = 3000

# Documents embedded per request during ingestion. A free Jina key embeds at most 100,000 tokens a minute,
# so one batch is sent a minute. 64 documents of the annual report come to at most about 70,000 tokens.
EMBEDDING_BATCH_SIZE = 64

# How many times to retry a model call that is rate-limited or temporarily unavailable
MODEL_RETRY_ATTEMPTS = 8

# Chunking: each page is split at its headings, and a section shorter than this many characters joins the next one
MIN_CHUNK_CHARACTERS = 500
