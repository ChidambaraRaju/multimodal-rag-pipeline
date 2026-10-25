import { useEffect, useRef, useState } from "react"
import { SquarePen } from "lucide-react"

import { Composer } from "@/components/composer"
import { EmptyState } from "@/components/empty-state"
import { Exchange, type Turn } from "@/components/exchange"
import { Button } from "@/components/ui/button"
import { ask, AskError } from "@/lib/api"
import { DOCUMENT_TITLE } from "@/lib/document"

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  // The server keeps the conversation's history; the page keeps only its id, so a reload starts afresh
  const conversationId = useRef<string | null>(null)
  const nextTurnId = useRef(1)
  const end = useRef<HTMLDivElement>(null)
  const busy = turns.some((turn) => turn.status === "loading")

  // Keep the newest question and its answer in view
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" })
  }, [turns.length])

  // Changes to one turn, given as values or worked out from the turn as it is now
  function update(id: number, changes: Partial<Turn> | ((turn: Turn) => Partial<Turn>)) {
    setTurns((current) =>
      current.map((turn) =>
        turn.id === id ? { ...turn, ...(typeof changes === "function" ? changes(turn) : changes) } : turn
      )
    )
  }

  async function answer(id: number, question: string) {
    try {
      const result = await ask(question, conversationId.current, {
        // Text before a search was the model's lead-in, not the answer, so it is dropped
        onSearch: (query) => update(id, (turn) => ({ searches: [...turn.searches, query], draft: "" })),
        onText: (text) => update(id, (turn) => ({ draft: turn.draft + text })),
      })
      conversationId.current = result.conversation_id
      update(id, { status: "done", answer: result })
    } catch (error) {
      const message =
        error instanceof AskError ? error.message : "Something went wrong in the page. Reload it and ask again."
      update(id, { status: "error", error: message })
    }
  }

  function askQuestion(question: string) {
    const id = nextTurnId.current++
    setTurns((current) => [...current, { id, question, status: "loading", searches: [], draft: "" }])
    answer(id, question)
  }

  function askAgain(turn: Turn) {
    update(turn.id, { status: "loading", error: undefined, searches: [], draft: "" })
    answer(turn.id, turn.question)
  }

  function startNewChat() {
    conversationId.current = null
    setTurns([])
  }

  return (
    <div className="flex min-h-svh flex-col">
      {turns.length > 0 && (
        <header className="sticky top-0 z-10 border-b bg-background/90 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-3xl items-center justify-between gap-4 px-4">
            <span className="truncate text-sm font-semibold">{DOCUMENT_TITLE}</span>
            <Button variant="ghost" size="sm" onClick={startNewChat} disabled={busy}>
              <SquarePen /> New chat
            </Button>
          </div>
        </header>
      )}

      <main className="mx-auto w-full max-w-3xl flex-1 px-4">
        {turns.length === 0 ? (
          <EmptyState onAsk={askQuestion} />
        ) : (
          turns.map((turn) => <Exchange key={turn.id} turn={turn} onRetry={() => askAgain(turn)} />)
        )}
        <div ref={end} />
      </main>

      <footer className="sticky bottom-0 bg-linear-to-t from-background from-70% to-transparent pt-6">
        <div className="mx-auto max-w-3xl px-4 pb-4">
          <Composer onAsk={askQuestion} busy={busy} />
        </div>
      </footer>
    </div>
  )
}
