"""Pick the charts among the cropped images, and describe each one with a vision model.

A chart is stored by its description, which is what gets searched, and keeps the path of its image,
which the answering model is shown. Descriptions are saved as they come, so a later run, or one restarted
after a failure, only describes the figures it has not seen. Delete image_descriptions.json to describe them all again.
"""

import base64
import json
import os

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from multimodal_rag import config
from multimodal_rag.parse import IMAGE_DIRECTORY

DESCRIPTIONS_PATH = config.PARSED_DIRECTORY / "image_descriptions.json"

IMAGE_PROMPT = """You are writing a searchable description of an image from page {page_number} of a document.
The text of that page is below, for context. Describe the image itself, not the whole page.

PAGE TEXT:
{page_text}

Cover:
1. What kind of image it is (diagram, chart, photo, ...) and what it shows
2. Every label, name, number and connection you can read in it
3. What the image explains, in a few sentences
4. Questions it could answer, and other words people might search for

DESCRIPTION:"""


def crop_page(crop):
    """The page number in a crop's file name, such as page_8_chart_1_v2.jpg."""
    return int(crop["filename"].split("_")[1])


def crop_kind(crop):
    """'image', 'chart' or 'table', from a crop's file name such as page_8_chart_1_v2.jpg."""
    return crop["filename"].split("_")[2]


def pick_charts(crops):
    """The charts to describe, diagrams that LlamaParse files as charts included.

    Image crops (photos, logos, signatures, icons) add little beyond the page text, and table crops
    are covered by the table HTML in the text, so both are left out.
    """
    return [crop for crop in crops if crop_kind(crop) == "chart"]


def description_model():
    """DeepSeek Flash through its OpenAI-compatible API. Rate limits and server errors are retried."""
    return ChatOpenAI(
        model=config.IMAGE_DESCRIPTION_MODEL,
        base_url=config.DESCRIPTION_BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
        max_tokens=config.DESCRIPTION_MAX_TOKENS,
        max_retries=config.MODEL_RETRY_ATTEMPTS,
    )


def describe_figure(llm, image_path, page):
    """A searchable description of one figure, with its page's text as context."""
    image = base64.b64encode(image_path.read_bytes()).decode()
    prompt = IMAGE_PROMPT.format(page_number=page["metadata"]["page_number"], page_text=page["markdown"])
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    ]
    return llm.invoke([HumanMessage(content=content)]).content


def build_figure_documents(crops, pages):
    """One document per figure: its description, its page's metadata and the path of its image."""
    descriptions = json.loads(DESCRIPTIONS_PATH.read_text()) if DESCRIPTIONS_PATH.exists() else {}
    llm = description_model()
    documents = []

    for crop in pick_charts(crops):
        filename = crop["filename"]
        image_path = IMAGE_DIRECTORY / filename
        page = pages[crop_page(crop)]

        if filename not in descriptions:
            print(f"Describing {filename}")
            descriptions[filename] = describe_figure(llm, image_path, page)
            DESCRIPTIONS_PATH.write_text(json.dumps(descriptions, indent=1))

        metadata = {**page["metadata"], "image_path": image_path.relative_to(config.PROJECT_ROOT).as_posix()}
        documents.append(Document(page_content=descriptions[filename], metadata=metadata, id=image_path.stem))

    return documents
