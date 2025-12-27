import * as fs from "node:fs/promises";
import * as path from "node:path";

type PagefindSearchResult = {
  id: string;
  score: number;
  words: number[];
  data: () => Promise<PagefindResultData>;
};

type PagefindResultData = {
  url: string;
  excerpt: string;
  meta: Record<string, string>;
  content: string;
  raw_content: string;
  raw_url: string;
  sub_results: Array<{
    title: string;
    url: string;
    excerpt: string;
  }>;
};

type PagefindResponse = {
  results: PagefindSearchResult[];
  unfilteredResultCount: number;
  filters: Record<string, Record<string, number>>;
  totalFilters: Record<string, Record<string, number>>;
  timings: {
    preload: number;
    search: number;
    total: number;
  };
};

type PagefindModule = {
  options: (opts: { basePath: string }) => Promise<void>;
  init: () => Promise<void>;
  search: (query: string) => Promise<PagefindResponse>;
  debouncedSearch: (
    query: string,
    options?: Record<string, unknown>,
    debounceMs?: number,
  ) => Promise<PagefindResponse | null>;
  filters: () => Promise<Record<string, Record<string, number>>>;
  destroy: () => Promise<void>;
};

let pagefindModule: PagefindModule | null = null;
let initPromise: Promise<PagefindModule> | null = null;

// O(1) lookup map: slugified title -> llms.txt URL
const llmsTxtMap = new Map<string, string>();
// Set of available llms-txt file slugs for URL-based lookup
const llmsTxtFiles = new Set<string>();
let llmsTxtInitialized = false;

function createFileFetch(baseDir: string) {
  return async (url: string | URL | Request): Promise<Response> => {
    let urlString: string;
    if (url instanceof Request) {
      urlString = url.url;
    } else if (url instanceof URL) {
      urlString = url.toString();
    } else {
      urlString = url;
    }

    const urlWithoutQuery = urlString.split("?")[0];

    let filePath: string;
    if (urlWithoutQuery.startsWith("file://")) {
      filePath = urlWithoutQuery.slice(7);
    } else if (path.isAbsolute(urlWithoutQuery)) {
      filePath = urlWithoutQuery;
    } else if (
      urlWithoutQuery.startsWith("/") ||
      urlWithoutQuery.startsWith("./")
    ) {
      const cleaned = urlWithoutQuery.replace(/^\.?\//, "");
      filePath = path.join(baseDir, cleaned);
    } else {
      filePath = path.join(baseDir, urlWithoutQuery);
    }

    try {
      const fileContents = await fs.readFile(filePath);
      return new Response(new Uint8Array(fileContents), {
        status: 200,
        headers: { "Content-Type": "application/octet-stream" },
      });
    } catch (err) {
      console.error(`Failed to read file: ${filePath}`, err);
      throw err;
    }
  };
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/['']/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

async function initLlmsTxtMap(clientDir: string): Promise<void> {
  if (llmsTxtInitialized) return;

  const llmsTxtPath = path.join(clientDir, "llms.txt");
  const llmsTxtDir = path.join(clientDir, "_llms-txt");

  try {
    const content = await fs.readFile(llmsTxtPath, "utf-8");

    // Parse markdown links: [Title](url)
    // Match lines like: - [Title](http://.../_llms-txt/slug.txt): description
    const linkRegex = /\[([^\]]+)\]\([^)]*\/_llms-txt\/([^)]+\.txt)\)/g;
    for (const match of content.matchAll(linkRegex)) {
      const title = match[1];
      const filename = match[2];
      const llmsTxtUrl = `/_llms-txt/${filename}`;
      const titleSlug = slugify(title);

      llmsTxtMap.set(titleSlug, llmsTxtUrl);
      // Also store the filename slug for URL-based lookup
      llmsTxtFiles.add(filename.replace(".txt", ""));
    }

    llmsTxtInitialized = true;
  } catch {
    // Try to read the _llms-txt directory directly as fallback
    try {
      const files = await fs.readdir(llmsTxtDir);
      for (const file of files) {
        if (file.endsWith(".txt")) {
          llmsTxtFiles.add(file.replace(".txt", ""));
        }
      }
    } catch {
      // Directory doesn't exist, continue without it
    }
    llmsTxtInitialized = true;
  }
}

function findLlmsTxtUrl(title: string, pageUrl: string): string | null {
  const titleSlug = slugify(title);

  // O(1) exact lookup by title
  const exactMatch = llmsTxtMap.get(titleSlug);
  if (exactMatch) return exactMatch;

  // Try URL-based lookup: extract the last path segment
  // e.g., /fundamentals/error-handling-validation/ -> error-handling-validation
  const urlSlug = pageUrl
    .replace(/\/$/, "")
    .split("/")
    .pop();
  if (urlSlug && llmsTxtFiles.has(urlSlug)) {
    return `/_llms-txt/${urlSlug}.txt`;
  }

  // Fallback: check if any key contains the title slug or vice versa
  for (const [key, url] of llmsTxtMap) {
    if (key.includes(titleSlug) || titleSlug.includes(key)) {
      return url;
    }
  }

  return null;
}

async function initPagefind(dir: string): Promise<PagefindModule> {
  if (pagefindModule) {
    return pagefindModule;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = (async (): Promise<PagefindModule> => {
    const pagefindPath = path.join(dir, "pagefind.js");
    const contents = await fs.readFile(pagefindPath);
    const moduleUrl = `data:application/javascript;base64,${contents.toString(
      "base64",
    )}`;

    const originalFetch = globalThis.fetch;
    const fileFetch = createFileFetch(dir);

    Object.assign(globalThis, {
      window: { location: { origin: "" } },
      document: {
        querySelector: () => ({ getAttribute: () => "en" }),
        currentScript: null,
      },
      location: { href: `file://${pagefindPath}` },
      fetch: fileFetch,
    });

    try {
      const module = (await import(moduleUrl)) as PagefindModule;

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

function normalizeUrl(url: string, clientDir: string): string {
  if (url.includes(clientDir)) {
    const clientIndex = url.indexOf(clientDir);
    return url.slice(clientIndex + clientDir.length);
  }
  if (url.startsWith("/") && !url.startsWith("//")) {
    return url;
  }
  return `/${url}`;
}

export async function searchPagefind(
  query: string,
  dir: string,
): Promise<{
  results: Array<{
    url: string;
    title: string;
    excerpt: string;
    score: number;
    llmsTxt: string;
  }>;
  totalResults: number;
}> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = createFileFetch(dir);

  const clientDir = dir.replace(/\/pagefind\/?$/, "");

  // Initialize llms.txt lookup map (O(1) lookups after init)
  await initLlmsTxtMap(clientDir);

  try {
    const pagefind = await initPagefind(dir);
    const searchResponse = await pagefind.search(query);

    const allResults = await Promise.all(
      searchResponse.results.slice(0, 20).map(async (result) => {
        const data = await result.data();
        const url = normalizeUrl(data.raw_url || data.url, clientDir);
        const title = data.meta?.title || "Untitled";
        const llmsTxt = findLlmsTxtUrl(title, url);
        return {
          url,
          title,
          excerpt: data.excerpt,
          score: result.score,
          llmsTxt,
        };
      }),
    );

    // Filter out results without llms-txt files and limit to 10
    const results = allResults
      .filter((r): r is typeof r & { llmsTxt: string } => r.llmsTxt !== null)
      .slice(0, 10);

    return {
      results,
      totalResults: results.length,
    };
  } finally {
    globalThis.fetch = originalFetch;
  }
}

export type { PagefindResponse, PagefindResultData };
