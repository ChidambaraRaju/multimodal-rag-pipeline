"""Answer a question with an agent that can search the ingested document.

The model decides for itself whether to search. When it does, it calls the search_document
tool, reads the chunks that come back (text, tables and images), and then answers.
Questions that need nothing from the document are answered without a search.

Use answer_question() from other code, for example an API endpoint.
Run this file alone to try the test questions:  python -m multimodal_rag.generate
All settings live in config.py. Run ingest.py first so the vector store exists.
"""

import json

from groq import RateLimitError
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, before_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from multimodal_rag import config
from multimodal_rag.store import open_vector_store

# The first question needs no search. The last is not covered by the first 6 pages,
# to check the model says so instead of guessing.
TEST_QUESTIONS = [
    "Hi, what can you help me with?",
    "How many attention heads does the Transformer use, and what is the dimension of each head?",
    "In the Transformer architecture figure, what sits directly above the decoder stack?",
    "How does the complexity per layer of self-attention compare with recurrent and convolutional layers?",
    "Which optimizer and learning rate schedule were used to train the Transformer?",
]

SYSTEM_PROMPT = """You answer questions about a document.

Use the search_document tool whenever a question needs information from the document. \
You may search again with a different query if the first results are not enough.
Answer document questions only from what the tool returns, using its text, tables and images. \
If the results do not contain the answer, say "I don't have enough information to answer that question based on the document."
Reply directly, without searching, to greetings and to questions that are not about the document."""


@tool(response_format="content_and_artifact")
def search_document(query: str):
    """Search the document for the passages, tables and figures most relevant to the query."""
    results = open_vector_store().similarity_search_with_relevance_scores(query, k=config.RETRIEVED_CHUNK_COUNT)

    chunks = []
    text_for_model = ""
    for number, (document, score) in enumerate(results, start=1):
        original_content = json.loads(document.metadata["original_content"])
        chunk = {
            "score": score,  # 0 to 1, higher is more relevant
            "text": original_content["raw_text"],
            "tables_html": original_content["tables_html"],
            "images_base64": original_content["images_base64"],
        }
        chunks.append(chunk)

        text_for_model += f"--- Document {number} ---\nTEXT:\n{chunk['text']}\n\n"
        for table_number, table in enumerate(chunk["tables_html"], start=1):
            text_for_model += f"Table {table_number}:\n{table}\n\n"

    # The model reads text_for_model. The chunks are kept on the tool message as its artifact.
    return text_for_model, chunks


@before_model
def show_retrieved_images(state, runtime):
    """Groq only accepts text in a tool result, so the images of the retrieved chunks follow in their own message."""
    last_message = state["messages"][-1]
    if last_message.type != "tool" or not last_message.artifact:
        return None

    images = [image for chunk in last_message.artifact for image in chunk["images_base64"]]
    if not images:
        return None

    content = [{"type": "text", "text": "Images from the retrieved documents:"}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    return {"messages": [HumanMessage(content=content)]}


def build_agent():
    """The generation model with the search tool attached."""
    # reasoning_format="parsed" keeps the model's thinking out of the answer text
    llm = ChatGroq(model=config.GENERATION_MODEL, temperature=0, reasoning_format="parsed")

    # Groq's free tier allows few output tokens per minute, so a rate-limited call waits and tries again
    wait_out_rate_limits = ModelRetryMiddleware(
        max_retries=config.MODEL_RETRY_ATTEMPTS, retry_on=(RateLimitError,), on_failure="error"
    )
    return create_agent(
        llm,
        [search_document],
        system_prompt=SYSTEM_PROMPT,
        middleware=[show_retrieved_images, wait_out_rate_limits],
    )


def answer_question(question):
    """Let the agent answer one question. Returns the answer, the model, and any searches and chunks it used."""
    result = build_agent().invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    final_message = messages[-1]

    search_queries = []
    chunks = []
    for message in messages:
        if message.type == "ai":
            search_queries += [tool_call["args"]["query"] for tool_call in message.tool_calls]
        elif message.type == "tool" and message.artifact:
            # A second search can return a chunk the first one already found
            chunks += [chunk for chunk in message.artifact if chunk not in chunks]

    return {
        "question": question,
        "answer": final_message.text,
        "model": final_message.response_metadata.get("model_name", config.GENERATION_MODEL),
        "search_queries": search_queries,  # empty when the model answered without searching
        "chunks": chunks,
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
        print(
            f"CHUNK {number}  score={chunk['score']:.3f}  "
            f"tables={len(chunk['tables_html'])}  images={len(chunk['images_base64'])}"
        )
        print(f"{chunk['text']}\n")

    print(f"GENERATED BY: {result['model']}\n")
    print(f"ANSWER:\n{result['answer']}\n")


def main():
    for question in TEST_QUESTIONS:
        print_result(answer_question(question))


if __name__ == "__main__":
    main()
