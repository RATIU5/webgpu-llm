# WebGPU and TypeGPU Documentation

A comprehensive documentation site for WebGPU and TypeGPU, built with [Astro Starlight](https://starlight.astro.build/). This site provides guides, tutorials, and reference material for developers working with modern GPU programming on the web.

## Features

- **WebGPU Fundamentals** - Core concepts, device initialization, command encoders, and error handling
- **WGSL Shading Language** - Syntax, data types, address spaces, and built-in functions
- **TypeGPU Integration** - Type-safe GPU programming with TypeScript
- **Render & Compute Pipelines** - Graphics rendering and general-purpose GPU computing
- **Practical Tutorials** - Step-by-step guides including particle systems and real-time graphics
- **LLM-Optimized Content** - Includes `llms.txt` for AI/LLM consumption via [starlight-llms-txt](https://github.com/humanspeak/starlight-llms-txt)

## Getting Started

```bash
# Install dependencies
bun install

# Start development server
bun run dev

# Build for production
bun run build

# Preview production build
bun run preview
```

## Project Structure

```
src/
  content/docs/         # Documentation pages (MDX)
  lib/
    pagefind-server.ts  # Server-side search implementation
  pages/
    search.ts           # Search API endpoint
```

## Server-Side Pagefind Search

This project implements server-side search using [Pagefind](https://pagefind.app/) v1.4.0+, which added Node.js runtime support. This allows search queries to be processed on the server rather than requiring client-side JavaScript.

### How It Works

Pagefind was originally designed for client-side browser usage. To run it on the server, we use a workaround that:

1. **Loads the Pagefind module via data URL** - The `pagefind.js` file is read from disk and imported as a base64-encoded data URL to bypass ES module resolution issues.

2. **Mocks browser globals** - Pagefind expects browser APIs, so we temporarily inject mock `window`, `document`, and `location` objects into `globalThis`.

3. **Overrides `fetch` for filesystem access** - The critical piece: we replace `globalThis.fetch` with a custom implementation that reads files from the local filesystem instead of making HTTP requests. This allows Pagefind to load its WASM binary, metadata, and index chunks from `dist/client/pagefind/`.

4. **Normalizes result URLs** - Search results are post-processed to convert absolute file paths back to relative web URLs.

### Implementation Files

- **`src/lib/pagefind-server.ts`** - Core search logic with browser mocking and filesystem fetch
- **`src/pages/api/search.ts`** - Astro API route exposing the search endpoint

### Usage

The search endpoint accepts POST requests:

```bash
curl -X POST http://localhost:4321/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "webgpu"}'
```

Response:

```json
{
  "results": [
    {
      "url": "/fundamentals/webgpu-core-concepts/",
      "title": "WebGPU Core Concepts",
      "excerpt": "...highlighted <mark>WebGPU</mark> content...",
      "score": 5.2
    }
  ],
  "totalResults": 28
}
```

### Adapting for Your Project

To use this approach in your own Astro/Starlight project:

1. Copy `src/lib/pagefind-server.ts` to your project
2. Create an API route that calls `searchPagefind(query, pagefindDir)`
3. Ensure `pagefindDir` points to your built Pagefind index (typically `dist/client/pagefind/`)
4. The Pagefind module is initialized once and cached for subsequent requests

### Caveats

- Requires Pagefind v1.4.0+ for Node.js runtime detection
- The browser globals mock is minimal; complex Pagefind features may need additional mocking
- First search request incurs WASM initialization overhead (~100-200ms)

## Tech Stack

- [Astro](https://astro.build/) - Static site generator
- [Starlight](https://starlight.astro.build/) - Documentation theme
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [Pagefind](https://pagefind.app/) - Static search
- [Bun](https://bun.sh/) - JavaScript runtime

## License

MIT
