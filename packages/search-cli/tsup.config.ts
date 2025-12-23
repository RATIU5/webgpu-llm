import { defineConfig } from "tsup";

export default defineConfig({
	entry: ["src/bin.ts"],
	clean: true,
	format: "esm",
	dts: {
		compilerOptions: {
			composite: false,
			incremental: false,
		},
	},
	sourcemap: true,
	treeshake: "smallest",
	external: ["@parcel/watcher"],
	banner: {
		js: "#!/usr/bin/env node",
	},
});
