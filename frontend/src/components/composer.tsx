import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

// The question box. Enter asks, Shift+Enter starts a new line.
export function Composer({ onAsk, busy }: { onAsk: (question: string) => void; busy: boolean }) {
  const [text, setText] = useState("")
  const question = text.trim()

  function send() {
    if (question && !busy) {
      onAsk(question)
      setText("")
    }
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        send()
      }}
      className="flex items-end gap-2 rounded-xl border bg-card p-2 shadow-sm focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/20"
    >
      <Textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault()
            send()
          }
        }}
        rows={1}
        placeholder="Ask about the annual report"
        aria-label="Your question"
        className="max-h-40 min-h-10 resize-none border-0 bg-transparent px-2 py-2 text-base shadow-none focus-visible:ring-0 md:text-base"
      />
      <Button type="submit" size="lg" disabled={busy || !question} className="h-10 px-4">
        Ask
      </Button>
    </form>
  )
}
