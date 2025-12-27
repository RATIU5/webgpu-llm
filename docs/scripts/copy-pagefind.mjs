import { cpSync, existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const baseDir = new URL("../dist/client/", import.meta.url);
const destBaseDir = new URL(
  "../.vercel/output/functions/_render.func/docs/dist/client/",
  import.meta.url,
);

// Files/directories to copy for server-side search
const toCopy = ["pagefind", "llms.txt", "_llms-txt"];

for (const item of toCopy) {
  const src = new URL(item, baseDir);
  const dest = new URL(item, destBaseDir);
  const srcPath = fileURLToPath(src);
  const destPath = fileURLToPath(dest);

  if (!existsSync(srcPath)) {
    console.warn(`Warning: ${item} not found at ${srcPath}`);
    continue;
  }

  mkdirSync(fileURLToPath(destBaseDir), { recursive: true });
  cpSync(srcPath, destPath, { recursive: true });
  console.log(`Copied ${item} to serverless function bundle`);
}
