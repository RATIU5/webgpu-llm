import { cpSync, existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const src = new URL("../dist/client/pagefind", import.meta.url);
const dest = new URL(
  "../.vercel/output/functions/_render.func/docs/dist/client/pagefind",
  import.meta.url,
);

const srcPath = fileURLToPath(src);
const destPath = fileURLToPath(dest);

if (!existsSync(srcPath)) {
  console.error("Source pagefind directory not found:", srcPath);
  process.exit(1);
}

mkdirSync(destPath, { recursive: true });
cpSync(srcPath, destPath, { recursive: true });
console.log("Copied pagefind to serverless function bundle");
