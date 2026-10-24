// Calls to the FastAPI backend in multimodal_rag/api.py. During development, Vite forwards them to it.

export type Chunk = {
  id: string // page139-2 for the 2nd text chunk of page 139, or a figure's image name
  score: number // 0 to 1, higher is more relevant
  page_number: number // page in the PDF, counting from 1
  printed_page_number: string // as printed on the page, "" if none
  text: string // a passage with its tables as HTML, or a figure's description
  image_path: string | null // a figure's image, relative to the project root
}

export type Answer = {
  conversation_id: string
  question: string
  answer: string
  model: string
  search_queries: string[] // empty when the model answered without searching
  chunks: Chunk[]
}

// The services behind the API, by the host its error names
const SERVICES: Record<string, string> = {
  "api.deepseek.com": "The answer model (DeepSeek)",
  "api.jina.ai": "The search service (Jina)",
}

// A failed question, with a message written for the person asking
export class AskError extends Error {}

// Ask a question. Leave conversationId null for the first question of a conversation.
export async function ask(
  question: string,
  conversationId: string | null
): Promise<Answer> {
  let response: Response
  try {
    response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: conversationId }),
    })
  } catch {
    throw new AskError(
      "The server isn't responding. Start it with uvicorn multimodal_rag.api:app, then ask again."
    )
  }
  if (response.ok) {
    return response.json()
  }

  const body = await response.json().catch(() => null)
  const service = SERVICES[body?.service] ?? "A service behind the server"
  if (response.status === 429) {
    throw new AskError(
      `${service} is rate-limited right now. Wait a few seconds, then ask again.`
    )
  }
  if (response.status === 502 && body?.error) {
    throw new AskError(
      `${service} returned an error, so there is no answer: ${body.error}`
    )
  }
  throw new AskError(
    `The server couldn't answer (status ${response.status}). Check that it is running and look at its log.`
  )
}
