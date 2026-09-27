"""Describe picture crops with Groq and keep the files on disk."""

import base64
import json
from io import BytesIO
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from multimodal_rag.config import GROQ_PICTURE_MODEL, picture_crop_directory

PICTURE_DESCRIPTION_PROMPT = """You describe a figure so it can be retrieved by search.

Return JSON only, with no markdown fence:
{"skip": false, "retrieval_text": "..."}

Set skip to true for a header, footer, page number, logo, icon, or any other image that is not informative. When skip is true, retrieval_text must be an empty string.

When skip is false, retrieval_text is a search document. Include the metric, period, comparison basis, units, and series, and only values that are readable in the image. Do not invent values that are not visible. If the figure is informative but has no numeric series, describe only what is readable.

A native caption is context. It does not add values that are not in the image. If a Docling chart table is included, use it only where it matches the image, and still describe the figure from the crop.
"""

GROQ_MIN_IMAGE_SIDE_PIXELS = 32


def parse_picture_description_json(model_response_text: str) -> dict:
    response_text = model_response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        if json_start == -1 or json_end <= json_start:
            raise
        return json.loads(response_text[json_start : json_end + 1])


def docling_chart_table_note(picture) -> str:
    picture_meta = picture.meta
    tabular_chart = picture_meta.tabular_chart if picture_meta is not None else None
    if tabular_chart is None or tabular_chart.chart_data is None:
        return ""
    return f"\nDocling chart table:\n{tabular_chart.chart_data}"


def build_picture_documents(
    docling_document: DoclingDocument, pdf_path: Path
) -> tuple[list[Document], int]:
    picture_description_model = ChatGroq(
        model=GROQ_PICTURE_MODEL,
        temperature=0,
        reasoning_format="parsed",
    )
    picture_chunk_documents = []
    skipped_picture_count = 0
    crop_directory = picture_crop_directory(pdf_path)

    for picture_index, picture in enumerate(docling_document.pictures):
        picture_crop = picture.get_image(docling_document)
        provenance = picture.prov[0] if picture.prov else None
        page_number = provenance.page_no if provenance is not None else None
        bounding_box = list(provenance.bbox.as_tuple()) if provenance is not None else None
        native_caption = picture.caption_text(docling_document)
        crop_is_too_small = (
            picture_crop is None
            or picture_crop.width < GROQ_MIN_IMAGE_SIDE_PIXELS
            or picture_crop.height < GROQ_MIN_IMAGE_SIDE_PIXELS
        )
        if crop_is_too_small:
            skipped_picture_count += 1
            continue

        png_buffer = BytesIO()
        picture_crop.save(png_buffer, format="PNG")
        image_base64 = base64.b64encode(png_buffer.getvalue()).decode("utf-8")
        picture_prompt = PICTURE_DESCRIPTION_PROMPT
        if native_caption:
            picture_prompt += f"\nNative caption: {native_caption}"
        picture_prompt += docling_chart_table_note(picture)
        model_response = picture_description_model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": picture_prompt},
                        {"type": "image", "base64": image_base64, "mime_type": "image/png"},
                    ]
                )
            ]
        )
        picture_description = parse_picture_description_json(model_response.content)
        retrieval_text = (picture_description.get("retrieval_text") or "").strip()
        if picture_description.get("skip") or not retrieval_text:
            skipped_picture_count += 1
            continue

        crop_file_path = crop_directory / f"page-{page_number}-picture-{picture_index}.png"
        crop_file_path.write_bytes(png_buffer.getvalue())
        picture_chunk_documents.append(
            Document(
                page_content=retrieval_text,
                metadata={
                    "source": str(pdf_path),
                    "element_type": "picture",
                    "page": page_number,
                    "bbox": bounding_box,
                    "caption": native_caption,
                    "image_path": str(crop_file_path),
                },
            )
        )
    return picture_chunk_documents, skipped_picture_count
