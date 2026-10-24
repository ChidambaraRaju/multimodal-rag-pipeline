"""Answer a question with an agent that can search the ingested document.

The model decides for itself whether to search. When it does, it calls the search_document
tool, reads the chunks that come back (passages with their tables, and figures), and then answers
with the page numbers it used. Questions that need nothing from the document are answered without a search.

Use answer_question() from other code, for example an API endpoint.
Run this file alone to try the test questions:  python -m multimodal_rag.generate
All settings live in config.py. Run ingest.py first so the vector store exists.
"""

import base64
import os

from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from multimodal_rag import config
from multimodal_rag.store import open_vector_store

# Questions about the 2026 annual report, each with the answer it should find. The first needs no search.
# The last is not answered anywhere in the report, to check the model says so instead of guessing.
TEST_QUESTIONS = [
    "Hi, what can you help me with?",
    # $215,938 million and $120,067 million (page 139, income statement)
    "What were NVIDIA's revenue and net income in fiscal year 2026?",
    # Not independent: Jen-Hsun Huang. Financial experts: Hudson, Jones, Lora, Seawell (page 26, checkmark table)
    "Which director nominees are not independent, and which are financial experts?",
    # The prompt, files, computer use, tools, OpenShell, memory, the LLM and sub-agents (page 8, diagram)
    "In the agentic AI architecture diagram, what does NemoClaw connect to?",
    # $1,448.75, against $200.33 for the S&P 500 (page 123, stock performance graph)
    "What was $100 invested in NVIDIA stock in January 2021 worth by January 2026, compared with the S&P 500?",
    "How much revenue did NVIDIA make from smartphone sales in fiscal 2026?",
]

SYSTEM_PROMPT = """You answer questions about a document.

Use the search_document tool whenever a question needs information from the document. \
You may search again with a different query if the first results are not enough.
Answer document questions only from what the tool returns, using its text, tables and figures. \
Cite the page of each result you use, for example (page 12).
If the results do not contain the answer, say "I don't have enough information to answer that question based on the document."
Reply directly, without searching, to greetings and to questions that are not about the document."""

# Opened once and shared by every search. The model can run several searches at once, and opening
# the store from several threads at the same time fails inside Chroma.
vector_store = open_vector_store()


@tool(response_format="content_and_artifact")
def search_document(query: str):
    """Search the document for the passages, tables and figures most relevant to the query."""
    results = vector_store.similarity_search_with_relevance_scores(query, k=config.RETRIEVED_CHUNK_COUNT)
    results = [(document, score) for document, score in results if score >= config.MIN_RELEVANCE_SCORE]
    if not results:
        return "No relevant passages were found in the document for this query.", []

    chunks = []
    text_for_model = ""
    for number, (document, score) in enumerate(results, start=1):
        chunk = {
            "score": score,  # 0 to 1, higher is more relevant
            "page_number": document.metadata["page_number"],  # page in the PDF, counting from 1
            "printed_page_number": document.metadata["printed_page_number"],  # as printed on the page, "" if none
            "text": document.page_content,  # a passage with its tables as HTML, or the description of a figure
            "image_path": document.metadata.get("image_path"),  # a figure's image, relative to the project root
        }
        chunks.append(chunk)

        kind = "Figure description (the figure itself follows)" if chunk["image_path"] else "Passage"
        text_for_model += f"--- Result {number}, page {chunk['page_number']}: {kind} ---\n{chunk['text']}\n\n"

    # The model reads text_for_model. The chunks are kept on the tool message as its artifact.
    return text_for_model, chunks


@before_model
def show_retrieved_images(state, runtime):
    """A tool result can only hold text, so the retrieved figures follow in their own message.

    The model can run several searches at once, so every search result since its last call is checked.
    """
    figures = {}
    for message in reversed(state["messages"]):
        if message.type != "tool":
            break
        for chunk in message.artifact or []:
            if chunk["image_path"]:
                figures[chunk["image_path"]] = chunk  # two searches can return the same figure
    if not figures:
        return None

    content = []
    for chunk in figures.values():
        image = base64.b64encode((config.PROJECT_ROOT / chunk["image_path"]).read_bytes()).decode()
        content.append({"type": "text", "text": f"Figure from page {chunk['page_number']}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    return {"messages": [HumanMessage(content=content)]}


def build_agent():
    """The generation model with the search tool attached. Rate limits and server errors are retried."""
    llm = ChatOpenAI(
        model=config.GENERATION_MODEL,
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
        max_retries=config.MODEL_RETRY_ATTEMPTS,
    )
    return create_agent(llm, [search_document], system_prompt=SYSTEM_PROMPT, middleware=[show_retrieved_images])


def answer_question(question):
    """Let the agent answer one question. Returns the answer, the model, and any searches and chunks it used."""
    result = build_agent().invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    final_message = messages[-1]

    search_queries = []
    chunks = {}  # by text, since two searches can return the same chunk with different scores
    for message in messages:
        if message.type == "ai":
            search_queries += [tool_call["args"]["query"] for tool_call in message.tool_calls]
        elif message.type == "tool" and message.artifact:
            for chunk in message.artifact:
                chunks.setdefault(chunk["text"], chunk)

    return {
        "question": question,
        "answer": final_message.text,
        "model": final_message.response_metadata["model_name"],
        "search_queries": search_queries,  # empty when the model answered without searching
        "chunks": list(chunks.values()),
    }


def print_result(result):
    """Show the searches, the retrieved chunks, the model and the answer in the terminal."""
    print("=" * 100)
    print(f"QUESTION: {result['question']}\n")

    if result["search_queries"]:
        print(f"SEARCHES: {result['search_queries']}\n")
    else:
        print("SEARCHES: none, the model answered without searching\n")

    for number, chunk in enumerate(result["chunks"], start=1):
        figure = f"  figure={chunk['image_path']}" if chunk["image_path"] else ""
        print(f"CHUNK {number}  score={chunk['score']:.3f}  page={chunk['page_number']}{figure}")
        print(f"{chunk['text']}\n")

    print(f"GENERATED BY: {result['model']}\n")
    print(f"ANSWER:\n{result['answer']}\n")


def main():
    for question in TEST_QUESTIONS:
        print_result(answer_question(question))


if __name__ == "__main__":
    main()
