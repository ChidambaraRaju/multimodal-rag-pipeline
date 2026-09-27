# Multimodal RAG

Single-document financial RAG over `2026-Annual-Report-Web.pdf`. Scale-out (multiple companies, periods, and reports) is out of scope until this file works end to end.

Always use the LangChain MCP servers before calling any LangChain, LangGraph, or LangSmith function, and follow the current docs and API reference those servers return.

Experiments belong in notebooks. The user reviews each notebook run and decides whether the approach stays. Move an approach into `.py` modules only after that decision.

- `docs-langchain` (`https://docs.langchain.com/mcp`): guides and how-tos
- `reference-langchain` (`https://reference.langchain.com/mcp`): classes, methods, and signatures

## Ingestion

Docling parses the PDF into layout elements (prose, tables, pictures) with page, bounding box, and reading order. Before writing a custom parser, check the LangChain docs via MCP for the current Docling integration and use it when it exposes the elements this pipeline needs: `TableItem` markdown, `PictureItem` crops, captions, and bounding boxes. A markdown-only export is not enough.

Walk those elements into LangChain `Document` chunks. Ingestion is a fixed pipeline. LangGraph is for the query path.

| Element | What gets embedded | What generation receives |
|---|---|---|
| Prose | Structure-first chunks, about 500–800 tokens, split on headings, heading kept on the chunk | Parent section text |
| `TableItem` | Table markdown plus a short label (statement name and period when visible) | The same markdown |
| `PictureItem` (chart or informative image) | VLM `retrieval_text` | Crop, native caption, and retrieval text |
| Headers, footers, page numbers, logos, icons | Nothing | Dropped |

Call `qwen/qwen3.8-27b` on Groq for picture descriptions. Ask for JSON with `retrieval_text` written as a search document: metric, period, comparison basis, units, series, and only values that are readable. Return a skip flag for non-informative images and do not embed those. Keep Docling's own generic picture-description prompt unused. Use Docling chart-to-table output when it returns a real table, and still keep the chart crop.

Embed with `nvidia/nemotron-3-embed-1b:free` on OpenRouter (text only, 2048 dimensions). Chunk for retrieval quality, not for the 32,768-token context limit. Confirm whether this route exposes passage versus query input types before trusting recall.

Store one Pinecone index. Every chunk points at a parent section and carries `page`, `bbox`, `element_type`, `parent_id`, and `image_path` when a crop exists. The index dimension must match Nemotron's 2048-dimensional embeddings. Read `PINECONE_API_KEY`, `GROQ_API_KEY`, and `OPENROUTER_API_KEY` from `.env`.

## Query path

LangGraph embeds the question, searches that collection, and merges hits that share a parent. When a surviving hit is a chart or image, generation receives the crop plus the retrieved text. Prose and table hits go to the model as text. The answer model is `qwen/qwen3.8-27b` on Groq.
