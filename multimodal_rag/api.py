"""The chatbot API over the ingested document.

Each answer streams to the page as server-sent events while the agent works: its searches, then the
answer piece by piece, then the whole result with its sources.

The server keeps the recent questions and answers of each conversation in memory, so a follow-up
question can refer to earlier ones. The UI only keeps the conversation id. Conversations are lost
when the server restarts, and a new page starts a new one.

Run from the project root, with one worker:  uvicorn multimodal_rag.api:app
One worker, because the conversations live in this process's memory and Chroma's on-disk store
is not safe across processes. Run ingest.py first so the vector store exists.
"""

import uuid

import openai
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from multimodal_rag import config
from multimodal_rag.generate import build_agent, stream_answer
from multimodal_rag.parse import IMAGE_DIRECTORY

app = FastAPI(title="ReportLens")
agent = build_agent()

# The cropped images, at the image_path each figure chunk carries,
# for example /parsed/2026-Annual-Report-Web/images/page_8_chart_1_v2.jpg
image_route = "/" + IMAGE_DIRECTORY.relative_to(config.PROJECT_ROOT).as_posix()
app.mount(image_route, StaticFiles(directory=IMAGE_DIRECTORY), name="images")

# Conversation id -> its last CHAT_HISTORY_TURNS turns, [{"question": ..., "answer": ...}], oldest first.
# A conversation moves to the end of the dict when it is used, so the first one is the least recently used.
conversations = {}


class Question(BaseModel):
    question: str
    conversation_id: str | None = None  # left out for the first question of a conversation


@app.post("/ask", response_class=EventSourceResponse)
def ask(request: Question):
    """Answer a question as a stream of events, with the earlier turns of its conversation as context.

    The events: "search" when the model starts a search, "text" for each new piece of the answer, and last
    "done", with the conversation id, the answer, the model, the searches and the retrieved chunks.
    Text before a search was the model's lead-in, not the answer, so the UI drops it when the search arrives.
    A missing or unknown conversation id starts a new conversation, for example after the server restarted.
    """
    history = conversations.pop(request.conversation_id, None)
    if history is None:
        conversation_id, history = str(uuid.uuid4()), []
    else:
        conversation_id = request.conversation_id
    conversations[conversation_id] = history
    if len(conversations) > config.MAX_CONVERSATIONS:
        del conversations[next(iter(conversations))]

    try:
        for kind, data in stream_answer(agent, request.question, history, thread_id=conversation_id):
            if kind == "done":
                history.append({"question": request.question, "answer": data["answer"]})
                del history[:-config.CHAT_HISTORY_TURNS]
                data = {"conversation_id": conversation_id, **data}
            yield ServerSentEvent(event=kind, data=data)
    except openai.APIError as error:
        # A DeepSeek or Jina call that still failed after its retries, for example a rate limit or a used-up
        # Jina balance. Both are called through the OpenAI client, so both raise its errors. The stream has
        # already started with status 200, so the failure ends it as an "error" event, with the service's
        # own status (None when it could not be reached), so the UI can tell a rate limit from other failures.
        status = getattr(error, "status_code", None)
        yield ServerSentEvent(
            event="error",
            data={"error": error.message, "service": error.request.url.host, "status": status},
        )


@app.get("/health")
def health():
    return {"status": "ok"}


# The built web UI (npm run build in frontend/), served at / so the page and the API share one origin
# and need no CORS. It is mounted last, so the routes above take precedence.
FRONTEND_DIRECTORY = config.PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIRECTORY.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIRECTORY, html=True), name="frontend")
