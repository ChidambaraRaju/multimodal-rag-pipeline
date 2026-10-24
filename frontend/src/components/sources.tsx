import { useEffect } from "react"

import { Markdown } from "@/components/markdown"
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import type { Chunk } from "@/lib/api"
import { cn } from "@/lib/utils"

type SourcesProps = {
  turnId: number
  chunks: Chunk[]
  openPage: number | null // the page whose excerpt is shown, or null
  onOpenPage: (page: number | null) => void
}

// What an answer was based on: the retrieved figures, and a tab for each retrieved page that opens its excerpt
export function Sources({ turnId, chunks, openPage, onOpenPage }: SourcesProps) {
  const excerptId = `excerpt-${turnId}`
  const pages = [...new Set(chunks.map((chunk) => chunk.page_number))]
  const figures = chunks.filter((chunk) => chunk.image_path)

  // A citation in the answer can open an excerpt further down, so bring it into view
  useEffect(() => {
    if (openPage !== null) {
      document.getElementById(excerptId)?.scrollIntoView({ block: "nearest" })
    }
  }, [openPage, excerptId])

  if (chunks.length === 0) {
    return null
  }
  return (
    <div className="mt-7 space-y-4">
      {figures.length > 0 && (
        <div className="flex flex-wrap gap-4">
          {figures.map((chunk) => (
            <FigureThumbnail key={chunk.image_path} chunk={chunk} />
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-2 text-sm">
        <span className="mr-1 text-muted-foreground">Sources</span>
        {pages.map((page) => (
          <button
            key={page}
            type="button"
            aria-expanded={openPage === page}
            aria-controls={excerptId}
            onClick={() => onOpenPage(openPage === page ? null : page)}
            className={cn(
              "rounded-sm px-1.5 py-0.5 tabular-nums transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
              openPage === page ? "bg-marker" : "bg-marker/35 hover:bg-marker/70"
            )}
          >
            p. {page}
          </button>
        ))}
      </div>

      {openPage !== null && (
        <PageExcerpt
          id={excerptId}
          page={openPage}
          chunks={chunks.filter((chunk) => chunk.page_number === openPage)}
        />
      )}
    </div>
  )
}

// The retrieved text of one page, and the descriptions of its retrieved figures
function PageExcerpt({ id, page, chunks }: { id: string; page: number; chunks: Chunk[] }) {
  const printed = chunks[0].printed_page_number
  const relevance = Math.max(...chunks.map((chunk) => chunk.score))
  return (
    <section id={id} aria-label={`Excerpt from page ${page}`} className="rounded-md border bg-card">
      <header className="flex items-baseline justify-between gap-4 border-b px-4 py-2.5 text-sm">
        <span className="font-semibold">
          Page {page}
          {printed && <span className="font-normal text-muted-foreground"> (printed as page {printed})</span>}
        </span>
        <span className="text-muted-foreground tabular-nums">Relevance {relevance.toFixed(2)}</span>
      </header>
      <div className="max-h-[28rem] space-y-6 overflow-y-auto px-4 py-4">
        {chunks.map((chunk, index) =>
          chunk.image_path ? (
            <div key={index} className="space-y-2">
              <p className="text-sm text-muted-foreground">
                The search matched this description of the figure on this page:
              </p>
              <Markdown className="excerpt" text={chunk.text} />
            </div>
          ) : (
            <Markdown key={index} className="excerpt" text={chunk.text} />
          )
        )}
      </div>
    </section>
  )
}

// A retrieved figure, which opens larger when selected
function FigureThumbnail({ chunk }: { chunk: Chunk }) {
  const src = `/${chunk.image_path}`
  const label = `Figure from page ${chunk.page_number}`
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          title="Enlarge"
          className="group block rounded-sm text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          <img
            src={src}
            alt={label}
            className="h-36 w-auto max-w-full rounded-sm border bg-card object-contain p-1 transition-shadow group-hover:shadow-md"
          />
          <span className="mt-1.5 block text-xs text-muted-foreground">Figure, page {chunk.page_number}</span>
        </button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-5xl" aria-describedby={undefined}>
        <DialogTitle>{label}</DialogTitle>
        <img src={src} alt={label} className="max-h-[78vh] w-full object-contain" />
      </DialogContent>
    </Dialog>
  )
}
