"""Paths, environment, and model names for the attention-paper trial."""

import os
from pathlib import Path

from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ATTENTION_PAPER_FILENAME = "attention-is-all-you-need.pdf"
PINECONE_INDEX_NAME = "multimodal-rag"
NEMOTRON_EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
GROQ_PICTURE_MODEL = "qwen/qwen3.8-27b"
NEMOTRON_EMBEDDING_DIMENSIONS = 2048


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    load_dotenv(project_root() / ".env")


def attention_paper_path() -> Path:
    pdf_path = project_root() / ATTENTION_PAPER_FILENAME
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    return pdf_path


def picture_crop_directory(pdf_path: Path) -> Path:
    crop_directory = project_root() / "crops" / pdf_path.stem
    crop_directory.mkdir(parents=True, exist_ok=True)
    return crop_directory
