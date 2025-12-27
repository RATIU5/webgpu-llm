import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import * as fs from 'node:fs/promises';
export { r as renderers } from '../../chunks/_@astro-renderers_BCHh4Q3R.mjs';

let pagefindModule = null;
let initPromise = null;
const llmsTxtMap = /* @__PURE__ */ new Map();
let llmsTxtInitialized = false;
function createFileFetch(baseDir) {
  return async (url) => {
    let urlString;
    if (url instanceof Request) {
      urlString = url.url;
    } else if (url instanceof URL) {
      urlString = url.toString();
    } else {
      urlString = url;
    }
    const urlWithoutQuery = urlString.split("?")[0];
    let filePath;
    if (urlWithoutQuery.startsWith("file://")) {
      filePath = urlWithoutQuery.slice(7);
    } else if (path.isAbsolute(urlWithoutQuery)) {
      filePath = urlWithoutQuery;
    } else if (urlWithoutQuery.startsWith("/") || urlWithoutQuery.startsWith("./")) {
      const cleaned = urlWithoutQuery.replace(/^\.?\//, "");
      filePath = path.join(baseDir, cleaned);
    } else {
      filePath = path.join(baseDir, urlWithoutQuery);
    }
    try {
      const fileContents = await fs.readFile(filePath);
      return new Response(new Uint8Array(fileContents), {
        status: 200,
        headers: { "Content-Type": "application/octet-stream" }
      });
    } catch (err) {
      console.error(`Failed to read file: ${filePath}`, err);
      throw err;
    }
  };
}
function slugify(text) {
  return text.toLowerCase().replace(/['']/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}
async function initLlmsTxtMap(clientDir) {
  if (llmsTxtInitialized) return;
  const llmsTxtPath = path.join(clientDir, "llms.txt");
  try {
    const content = await fs.readFile(llmsTxtPath, "utf-8");
    const linkRegex = /\[([^\]]+)\]\([^)]*\/_llms-txt\/([^)]+\.txt)\)/g;
    for (const match of content.matchAll(linkRegex)) {
      const title = match[1];
      const llmsTxtUrl = `/_llms-txt/${match[2]}`;
      const titleSlug = slugify(title);
      llmsTxtMap.set(titleSlug, llmsTxtUrl);
    }
    llmsTxtInitialized = true;
  } catch {
    llmsTxtInitialized = true;
  }
}
function findLlmsTxtUrl(title) {
  if (llmsTxtMap.size === 0) return null;
  const titleSlug = slugify(title);
  const exactMatch = llmsTxtMap.get(titleSlug);
  if (exactMatch) return exactMatch;
  for (const [key, url] of llmsTxtMap) {
    if (key.includes(titleSlug) || titleSlug.includes(key)) {
      return url;
    }
  }
  return null;
}
async function initPagefind(dir) {
  if (pagefindModule) {
    return pagefindModule;
  }
  if (initPromise) {
    return initPromise;
  }
  initPromise = (async () => {
    const pagefindPath = path.join(dir, "pagefind.js");
    const contents = await fs.readFile(pagefindPath);
    const moduleUrl = `data:application/javascript;base64,${contents.toString(
      "base64"
    )}`;
    const originalFetch = globalThis.fetch;
    const fileFetch = createFileFetch(dir);
    Object.assign(globalThis, {
      window: { location: { origin: "" } },
      document: {
        querySelector: () => ({ getAttribute: () => "en" }),
        currentScript: null
      },
      location: { href: `file://${pagefindPath}` },
      fetch: fileFetch
    });
    try {
      const module = await import(moduleUrl);
      await module.options({ basePath: `${dir}/` });
      await module.init();
      pagefindModule = module;
      return module;
    } finally {
      globalThis.fetch = originalFetch;
    }
  })();
  return initPromise;
}
function normalizeUrl(url, clientDir) {
  if (url.includes(clientDir)) {
    const clientIndex = url.indexOf(clientDir);
    return url.slice(clientIndex + clientDir.length);
  }
  if (url.startsWith("/") && !url.startsWith("//")) {
    return url;
  }
  return `/${url}`;
}
async function searchPagefind(query, dir) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = createFileFetch(dir);
  const clientDir = dir.replace(/\/pagefind\/?$/, "");
  await initLlmsTxtMap(clientDir);
  try {
    const pagefind = await initPagefind(dir);
    const searchResponse = await pagefind.search(query);
    const results = await Promise.all(
      searchResponse.results.slice(0, 10).map(async (result) => {
        const data = await result.data();
        const url = normalizeUrl(data.raw_url || data.url, clientDir);
        const title = data.meta?.title || "Untitled";
        return {
          url,
          title,
          excerpt: data.excerpt,
          score: result.score,
          llmsTxt: findLlmsTxtUrl(title)
        };
      })
    );
    return {
      results,
      totalResults: searchResponse.unfilteredResultCount
    };
  } finally {
    globalThis.fetch = originalFetch;
  }
}

const prerender = false;
const __dirname$1 = path.dirname(fileURLToPath(import.meta.url));
const pagefindDir = path.resolve(__dirname$1, "../../../client/pagefind");
const rateLimiter = new RateLimiterMemory({
  points: 20,
  duration: 60
});
function getClientIP(request) {
  return request.headers.get("x-forwarded-for")?.split(",")[0].trim() || request.headers.get("x-real-ip") || "unknown";
}
const POST = async ({ request }) => {
  const clientIP = getClientIP(request);
  try {
    await rateLimiter.consume(clientIP);
  } catch {
    return new Response(
      JSON.stringify({
        error: "Rate limit exceeded",
        retryAfter: 60
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": "60"
        }
      }
    );
  }
  const body = await request.json();
  if (!body?.query) {
    return new Response(
      JSON.stringify({
        error: "No query provided"
      }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  try {
    const searchResults = await searchPagefind(body.query, pagefindDir);
    return new Response(JSON.stringify(searchResults), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    console.error("Search error:", error);
    return new Response(
      JSON.stringify({
        error: "Search failed",
        details: error instanceof Error ? error.message : "Unknown error"
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
};

const _page = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
  __proto__: null,
  POST,
  prerender
}, Symbol.toStringTag, { value: 'Module' }));

const page = () => _page;

export { page };
