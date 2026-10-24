import { useState } from "react"
import { RotateCcw } from "lucide-react"

import { linkCitations, Markdown } from "@/components/markdown"
import { Sources } from "@/components/sources"
import { Button } from "@/components/ui/button"
import type { Answer } from "@/lib/api"

// One question of the conversation, and its answer once it arrives
export type Turn = {
  id: number
  question: string
  status: "loading" | "done" | "error"
  answer?: Answer
  error?: string // a message for the person asking
}

export function Exchange({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  const [openPage, setOpenPage] = useState<number | null>(null)
  const citedPages = new Set(turn.answer?.chunks.map((chunk) => chunk.page_number))

  return (
    <article className="border-t py-10 first:border-t-0">
      <h2 className="text-base leading-snug font-semibold text-pretty">{turn.question}</h2>

      {turn.status === "loading" && (
        <div role="status" className="mt-5">
          <p className="answer">
            <span className="reading">Searching the report</span>
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

      {turn.answer && (
        <>
          <Markdown
            className="answer mt-5"
            text={linkCitations(turn.answer.answer)}
            citedPages={citedPages}
            onCite={setOpenPage}
          />
          <Sources
            turnId={turn.id}
            chunks={turn.answer.chunks}
            openPage={openPage}
            onOpenPage={setOpenPage}
          />
        </>
      )}
    </article>
  )
}
