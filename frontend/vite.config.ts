import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The FastAPI backend: uvicorn multimodal_rag.api:app
const API = "http://127.0.0.1:8000"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    // During development the page and the API come from one origin, so the API needs no CORS
    proxy: {
      "/ask": API,
      "/health": API,
      "/parsed": API,
    },
  },
})
