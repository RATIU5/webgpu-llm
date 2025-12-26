# Starlight Search CLI

CLI tool for searching Starlight documentation and fetching llms.txt content.

## Installation

```sh
npm install -g @ratiu5/starlight-search-cli
```

## Commands

### search

Search documentation for a keyword.

```sh
sls search -d https://docs.example.com "buffers"
```

**Options:**

- `-d, --domain` - API domain (e.g. `http://localhost:4321`)

**Output:**

```json
{
  "results": [
    {
      "url": "https://docs.example.com/guide/buffers",
      "title": "GPU Buffers",
      "excerpt": "Learn about WebGPU buffers...",
      "score": 0.95,
      "llmsTxt": "https://docs.example.com/guide/buffers/_llms-txt"
    }
  ],
  "totalResults": 1
}
```

### show

Fetch and display llms.txt content from a URL.

```sh
sls show "https://docs.example.com/guide/buffers/_llms-txt"
```

## Development

```sh
pnpm dev      # watch mode
pnpm build    # build for production
pnpm typecheck
```
