"""The chatbot API over the ingested document.

The server keeps the recent questions and answers of each conversation in memory, so a follow-up
question can refer to earlier ones. The UI only keeps the conversation id. Conversations are lost
when the server restarts, and a new page starts a new one.

Run from the project root, with one worker:  uvicorn multimodal_rag.api:app
One worker, because the conversations live in this process's memory and Chroma's on-disk store
is not safe across processes. Run ingest.py first so the vector store exists.
"""

import uuid

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from multimodal_rag import config
from multimodal_rag.generate import answer_question, build_agent
from multimodal_rag.parse import IMAGE_DIRECTORY

app = FastAPI(title="Multimodal RAG")
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


@app.post("/ask")
def ask(request: Question):
    """Answer a question, with the earlier turns of its conversation as context.

    A missing or unknown conversation id starts a new conversation, for example after the server restarted.
    Returns the conversation id with the answer, the model, the searches and the retrieved chunks.
    """
    history = conversations.pop(request.conversation_id, None)
    if history is None:
        conversation_id, history = str(uuid.uuid4()), []
    else:
        conversation_id = request.conversation_id
    conversations[conversation_id] = history
    if len(conversations) > config.MAX_CONVERSATIONS:
        del conversations[next(iter(conversations))]

    result = answer_question(agent, request.question, history)
    history.append({"question": request.question, "answer": result["answer"]})
    del history[:-config.CHAT_HISTORY_TURNS]
    return {"conversation_id": conversation_id, **result}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(openai.APIError)
async def service_error(request: Request, error: openai.APIError):
    """A DeepSeek or Jina call that still failed after its retries, for example a rate limit or a used-up Jina balance.

    Both are called through the OpenAI client, so both raise its errors. A rate limit stays a 429, so the UI
    can ask the user to wait; anything else is a 502, since the failure is in a service behind this API.
    """
    status = getattr(error, "status_code", None)  # None when the service could not be reached
    return JSONResponse(
        status_code=429 if status == 429 else 502,
        content={"error": error.message, "service": error.request.url.host, "status": status},
    )


# The built web UI (npm run build in frontend/), served at / so the page and the API share one origin
# and need no CORS. It is mounted last, so the routes above take precedence.
FRONTEND_DIRECTORY = config.PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIRECTORY.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIRECTORY, html=True), name="frontend")
