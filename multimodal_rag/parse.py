"""Parse the PDF with LlamaParse, once.

LlamaParse reads each page with a vision model and returns it as markdown, with tables as HTML.
It also crops the figures and tables from each page. The result and the cropped images are saved
in config.PARSED_DIRECTORY and reused by later runs, so only the first run spends LlamaParse credits.
Delete that folder to parse again.
"""

import json

import httpx
from llama_cloud import LlamaCloud

from multimodal_rag import config

RESULT_PATH = config.PARSED_DIRECTORY / "parse_result.json"
IMAGE_DIRECTORY = config.PARSED_DIRECTORY / "images"


def parse_pdf(pdf_path):
    """Parse the PDF and save the result right away, since the parse is what costs credits.

    A table running over several pages is merged onto its first page, and the number printed
    on each page is read.
    """
    client = LlamaCloud()  # reads LLAMA_CLOUD_API_KEY from the environment
    file = client.files.create(file=str(pdf_path), purpose="parse")
    result = client.parsing.parse(
        file_id=file.id,
        tier=config.PARSE_TIER,
        version="latest",
        output_options={
            "markdown": {"tables": {"output_tables_as_markdown": False, "merge_continued_tables": True}},
            "images_to_save": ["layout"],  # the figures and tables cropped from each page
            "extract_printed_page_number": True,
        },
        expand=["markdown", "metadata", "images_content_metadata"],
    ).model_dump(mode="json")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=1))
    return result


def download_images(result):
    """Download the cropped images that are not on disk yet.

    Image links expire, so fresh ones are fetched from the parse job. If a download fails,
    running ingestion again downloads the rest.
    """
    missing = [image["filename"] for image in result["images_content_metadata"]["images"]
               if not (IMAGE_DIRECTORY / image["filename"]).exists()]
    if not missing:
        return

    job = LlamaCloud().parsing.get(result["job"]["id"], expand=["images_content_metadata"]).model_dump(mode="json")
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(missing)} images")
    for image in job["images_content_metadata"]["images"]:
        if image["filename"] in missing:
            response = httpx.get(image["presigned_url"], timeout=60)
            response.raise_for_status()
            (IMAGE_DIRECTORY / image["filename"]).write_bytes(response.content)


def read_pages(result):
    """Each page by its number, with its markdown and the metadata its documents carry.

    The metadata is the page number and the number printed on the page, "" when it has none.
    """
    printed_numbers = {page["page_number"]: page["printed_page_number"] for page in result["metadata"]["pages"]}
    pages = {}
    for page in result["markdown"]["pages"]:
        number = page["page_number"]
        metadata = {"page_number": number, "printed_page_number": printed_numbers[number] or ""}
        pages[number] = {"markdown": page["markdown"], "metadata": metadata}
    return pages


def load_parsed_pdf(pdf_path):
    """The PDF's pages and its cropped images, from the saved parse. The first run parses the PDF."""
    if RESULT_PATH.exists():
        result = json.loads(RESULT_PATH.read_text())
    else:
        print(f"Parsing {pdf_path.name} with LlamaParse")
        result = parse_pdf(pdf_path)
    download_images(result)
    return read_pages(result), result["images_content_metadata"]["images"]
