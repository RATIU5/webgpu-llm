import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/bin.ts"],
  clean: true,
  format: "esm",
  publicDir: true,
  treeshake: "smallest",
  external: ["@parcel/watcher"],
});
