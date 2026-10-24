import ReactMarkdown from "react-markdown"
import rehypeRaw from "rehype-raw"
import rehypeSanitize from "rehype-sanitize"
import remarkGfm from "remark-gfm"

// Answers are markdown with tables. Page excerpts hold LlamaParse's HTML tables, so raw HTML
// is rendered too, then sanitized so only safe elements remain.
const remarkPlugins = [remarkGfm]
const rehypePlugins = [rehypeRaw, rehypeSanitize]

// "page 139" and "pages 23, 167" in an answer
const CITATION = /\b(pages?)\s+(\d+(?:\s*(?:,|and)\s*\d+)*)/gi

// Turn the answer's page citations into #page-N links, which Markdown renders as citation marks
export function linkCitations(markdown: string) {
  return markdown.replace(CITATION, (match, word: string, numbers: string) =>
    word.toLowerCase() === "page"
      ? `[${match}](#page-${numbers})`
      : `${word} ${numbers.replace(/\d+/g, (page) => `[${page}](#page-${page})`)}`
  )
}

type MarkdownProps = {
  text: string
  className: string
  citedPages?: Set<number> // pages with a retrieved source, whose citations can open it
  onCite?: (page: number) => void
}

export function Markdown({ text, className, citedPages, onCite }: MarkdownProps) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={{
          a({ href, children }) {
            if (!href?.startsWith("#page-")) {
              return (
                <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
                  {children}
                </a>
              )
            }
            const page = Number(href.slice("#page-".length))
            if (onCite && citedPages?.has(page)) {
              return (
                <button
                  type="button"
                  className="citation"
                  title={`Show the excerpt from page ${page}`}
                  onClick={() => onCite(page)}
                >
                  {children}
                </button>
              )
            }
            return <span className="citation">{children}</span>
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
