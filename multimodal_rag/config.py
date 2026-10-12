"""All settings in one place. Change values here, nothing else needs editing.

API keys are read from .env: OPENROUTER_API_KEY and GROQ_API_KEY.
"""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Document to ingest
DOCUMENT_PATH = PROJECT_ROOT / "attention-is-all-you-need-first6.pdf"

# Where the Chroma vector store is saved. Each ingestion run replaces its contents.
VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "vector_store" / "chroma_db"

# Models
IMAGE_DESCRIPTION_MODEL = "dots-studio/dots-3-note-preview:free"  # OpenRouter, describes images and tables
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"  # OpenRouter, text embeddings, 2048 dimensions
GENERATION_MODEL = "qwen/qwen3.8-27b"  # Groq, answers the question from the retrieved chunks

# How many chunks are retrieved for each question
RETRIEVED_CHUNK_COUNT = 3

# Longest description the model may write, in tokens. The embedding model accepts at most 4096 tokens per chunk.
DESCRIPTION_MAX_TOKENS = 3000

# How many times to retry a model call that is rate-limited or temporarily unavailable
MODEL_RETRY_ATTEMPTS = 8

# Chunking, in characters
CHUNK_MAX_CHARACTERS = 3000  # hard limit per chunk
CHUNK_NEW_AFTER_CHARACTERS = 2400  # start a new chunk after this many characters when possible
CHUNK_COMBINE_UNDER_CHARACTERS = 500  # merge chunks smaller than this into a neighbour
