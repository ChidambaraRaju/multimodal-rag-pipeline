"""Parse a PDF with Docling, keeping picture crops."""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument


def convert_pdf_to_docling_document(pdf_path: Path) -> DoclingDocument:
    pdf_pipeline_options = PdfPipelineOptions(
        generate_picture_images=True,
        generate_page_images=True,
    )
    document_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
        }
    )
    conversion_result = document_converter.convert(source=str(pdf_path))
    return conversion_result.document
