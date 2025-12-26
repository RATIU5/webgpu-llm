---
title: Installing the CLI
description: Use AI to learn easier with a CLI that searches the docs
---

The Starlight Search CLI (`sls`) lets you search this documentation and retrieve LLM-formatted content directly from your terminal.

## Prerequisites

- Node.js 20 or later

## Installation

```sh
npm install -g @ratiu5/starlight-search-cli
```

## Commands

### search

Search the documentation for a keyword or phrase. Shorter queries work best.

```sh
sls search -d https://webgpu-llm-docs.vercel.app "buffers"
```

**Options:**

| Flag           | Description                               |
| -------------- | ----------------------------------------- |
| `-d, --domain` | The root domain for the documentation API |

**Example output:**

```json
{
  "results": [
    {
      "url": "https://webgpu-llm-docs.vercel.app/data-and-buffers/buffers-memory-management",
      "title": "Buffers & Memory Management",
      "excerpt": "Learn about WebGPU buffers...",
      "score": 0.95,
      "llmsTxt": "https://webgpu-llm-docs.vercel.app/data-and-buffers/buffers-memory-management/_llms-txt"
    }
  ],
  "totalResults": 1
}
```

### show

Fetch and display the LLM-formatted content for a specific documentation page.

```sh
sls show "https://webgpu-llm-docs.vercel.app/data-and-buffers/buffers-memory-management/_llms-txt"
```

The URL must contain `_llms-txt` or `llms.txt` to be valid.

## Workflow Example

1. Search for a topic:

   ```sh
   sls search -d https://webgpu-llm-docs.vercel.app "compute shaders"
   ```

2. Copy the `llmsTxt` URL from a result

3. Fetch the full content:

   ```sh
   sls show "https://webgpu-llm-docs.vercel.app/shaders-and-pipelines/compute-pipelines/_llms-txt"
   ```

4. Use the output with your preferred LLM

## Configuring AI Agents

Add the following to your `AGENTS.md` or `CLAUDE.md` file to enable AI assistants to use the CLI for WebGPU/TypeGPU documentation lookup.

````markdown
## WebGPU/TypeGPU Documentation

This project uses **sls** (Starlight Search) for WebGPU and TypeGPU documentation lookup.

### Quick Reference

```bash
sls search -d https://webgpu-llm-docs.vercel.app "<query>"  # Search docs
sls show "<llmsTxt-url>"                                     # Fetch full content
```

### Usage Rules

- **Always search before implementing** WebGPU/TypeGPU code
- **Use 1-3 word queries**: "buffers", "compute pipelines", "bind groups"
- **Search concepts, not sentences**: "texture sampling" not "how do I sample a texture"
- **Fetch full docs** with `sls show` using the `llmsTxt` URL from results

### Example Session

```bash
# Find buffer documentation
sls search -d https://webgpu-llm-docs.vercel.app "buffers"

# Fetch the full content
sls show "https://webgpu-llm-docs.vercel.app/data-and-buffers/buffers-memory-management/_llms-txt"
```

### Workflow

1. When working with WebGPU or TypeGPU code, search for the relevant concept
2. Extract the `llmsTxt` URL from search results
3. Fetch and read the full documentation with `sls show`
4. Apply the patterns and examples to the current task

### Topic Coverage

| Area         | Topics                                                |
| ------------ | ----------------------------------------------------- |
| Fundamentals | Device initialization, adapters, canvas configuration |
| Data         | Buffers, memory management, data schemas              |
| Shaders      | WGSL, TGSL functions, compute/render pipelines        |
| Resources    | Bind groups, layouts, textures, samplers              |
| TypeGPU      | Type-safe abstractions, slots, derived values         |

### When to Search

- Before implementing any WebGPU API calls
- When encountering GPU-related errors
- When optimizing render or compute pipelines
- When working with TypeGPU's type system
````

To support both `AGENTS.md` and `CLAUDE.md`, create one file and symlink the other:

```sh
# If you created AGENTS.md
ln -s AGENTS.md CLAUDE.md

# Or if you created CLAUDE.md
ln -s CLAUDE.md AGENTS.md
```
