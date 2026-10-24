import { DOCUMENT_PAGES, DOCUMENT_TITLE, EXAMPLE_QUESTIONS } from "@/lib/document"

// The start of a conversation: what the chatbot answers about, and questions to begin with
export function EmptyState({ onAsk }: { onAsk: (question: string) => void }) {
  return (
    <section className="pt-[12vh] pb-10">
      <h1 className="font-serif text-4xl leading-[1.1] font-semibold text-balance sm:text-5xl">
        {DOCUMENT_TITLE}
      </h1>
      <p className="mt-5 max-w-[58ch] leading-relaxed text-pretty text-muted-foreground">
        Ask about anything in its {DOCUMENT_PAGES} pages, from the financial statements to the
        charts. Every answer cites the pages it comes from, and you can open each one.
      </p>

      <p className="mt-12 text-sm text-muted-foreground">Some questions to start with</p>
      <ul className="mt-3 border-t">
        {EXAMPLE_QUESTIONS.map((question) => (
          <li key={question} className="border-b">
            <button
              type="button"
              onClick={() => onAsk(question)}
              className="w-full py-3.5 text-left text-[0.9375rem] leading-snug text-pretty transition-colors hover:text-primary focus-visible:outline-2 focus-visible:outline-ring"
            >
              {question}
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
