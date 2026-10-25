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

// What the page hears while the answer is being written
type Progress = {
  onSearch: (query: string) => void // the model started a search; any text before it was not the answer
  onText: (text: string) => void // the next piece of the answer
}

// Ask a question, following the answer as the server streams it. Leave conversationId null for the
// first question of a conversation. Resolves with the whole answer and its sources once it is complete.
export async function ask(
  question: string,
  conversationId: string | null,
  { onSearch, onText }: Progress
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
  if (!response.ok || !response.body) {
    throw new AskError(
      `The server couldn't answer (status ${response.status}). Check that it is running and look at its log.`
    )
  }

  // Server-sent events: blocks separated by a blank line, each with an "event:" and a "data:" line
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      buffer += value
      const blocks = buffer.split("\n\n")
      buffer = blocks.pop() ?? "" // the last block may still be arriving
      for (const block of blocks) {
        const event = parseEvent(block)
        if (event?.name === "search") onSearch(event.data.query)
        else if (event?.name === "text") onText(event.data.text)
        else if (event?.name === "done") return event.data
        else if (event?.name === "error") throw serviceError(event.data)
      }
    }
  } catch (error) {
    if (error instanceof AskError) throw error
    throw new AskError("The connection to the server broke off before the answer was complete. Ask again.")
  }
  throw new AskError("The answer stopped before it was complete. Look at the server's log, then ask again.")
}

// One server-sent event, or null for the server's keep-alive comments
function parseEvent(block: string) {
  let name = ""
  let data = ""
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) name = line.slice("event:".length).trim()
    else if (line.startsWith("data:")) data += line.slice("data:".length).trim()
  }
  return data ? { name, data: JSON.parse(data) } : null
}

// A DeepSeek or Jina call that failed on the server, in words for the person asking
function serviceError(body: { error: string; service: string; status: number | null }) {
  const service = SERVICES[body.service] ?? "A service behind the server"
  if (body.status === 429) {
    return new AskError(`${service} is rate-limited right now. Wait a few seconds, then ask again.`)
  }
  return new AskError(`${service} returned an error, so there is no answer: ${body.error}`)
}
