import { useState } from "react"
import { RotateCcw, Search } from "lucide-react"

import { linkCitations, Markdown } from "@/components/markdown"
import { Sources } from "@/components/sources"
import { Button } from "@/components/ui/button"
import type { Answer } from "@/lib/api"

// One question of the conversation, and its answer as it streams in
export type Turn = {
  id: number
  question: string
  status: "loading" | "done" | "error"
  searches: string[] // the searches the model has started so far
  draft: string // the answer as it streams in, until the whole answer arrives
  answer?: Answer
  error?: string // a message for the person asking
}

export function Exchange({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  const [openPage, setOpenPage] = useState<number | null>(null)
  const citedPages = new Set(turn.answer?.chunks.map((chunk) => chunk.page_number))
  const searches = turn.answer?.search_queries ?? turn.searches
  // While the answer streams in, its citations are plain marks; they open excerpts once its sources arrive
  const text = turn.answer?.answer ?? (turn.status === "loading" ? turn.draft : "")

  return (
    <article className="border-t py-10 first:border-t-0">
      <h2 className="text-base leading-snug font-semibold text-pretty">{turn.question}</h2>

      {turn.status !== "error" && searches.length > 0 && (
        <p className="mt-2 flex gap-1.5 text-sm text-muted-foreground">
          <Search className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <span>
            Searched for{" "}
            {searches.map((query, index) => (
              <span key={index}>
                {index > 0 && ", "}“{query}”
              </span>
            ))}
          </span>
        </p>
      )}

      {turn.status === "loading" && !turn.draft && (
        <div role="status" className="mt-5">
          <p className="answer">
            <span className="reading">{searches.length > 0 ? "Reading the report" : "Reading your question"}</span>
          </p>
          <p className="mt-1 text-sm text-muted-foreground">Answers usually take 3 to 10 seconds.</p>
        </div>
      )}

      {turn.status === "error" && (
        <div role="alert" className="mt-5 border-l-2 border-destructive pl-4">
          <p className="text-sm leading-relaxed">{turn.error}</p>
          <Button variant="outline" size="sm" className="mt-3" onClick={onRetry}>
            <RotateCcw /> Ask again
          </Button>
        </div>
      )}

      {text && (
        <Markdown className="answer mt-5" text={linkCitations(text)} citedPages={citedPages} onCite={setOpenPage} />
      )}

      {turn.answer && (
        <Sources turnId={turn.id} chunks={turn.answer.chunks} openPage={openPage} onOpenPage={setOpenPage} />
      )}
    </article>
  )
}
