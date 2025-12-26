import { FileSystem, Path } from "@effect/platform";
import { NodeContext } from "@effect/platform-node";
import { Effect } from "effect";

interface PackageJson {
  name: string;
  version: string;
  type: string;
  description: string;
  engines?: Record<string, string>;
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  repository?: { type: string; url: string; directory?: string };
  author?: string;
  license?: string;
  bugs?: { url: string };
  homepage?: string;
  keywords?: string[];
}

const program = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  yield* Effect.log("[Build] Copying package.json ...");
  const json: PackageJson = yield* fs
    .readFileString("package.json")
    .pipe(Effect.map((s) => JSON.parse(s) as PackageJson));
  const pkg = {
    name: json.name,
    version: json.version,
    type: json.type,
    description: json.description,
    main: "bin.js",
    types: "bin.d.ts",
    bin: {
      sls: "bin.js",
    },
    files: ["bin.js", "bin.d.ts", "bin.js.map"],
    engines: json.engines,
    dependencies: json.dependencies,
    peerDependencies: json.peerDependencies,
    repository: json.repository,
    author: json.author,
    license: json.license,
    bugs: json.bugs,
    homepage: json.homepage,
    keywords: json.keywords,
  };
  yield* fs.writeFileString(
    path.join("dist", "package.json"),
    JSON.stringify(pkg, null, 2),
  );
  yield* Effect.log("[Build] Copying README.md ...");
  yield* fs.copyFile("README.md", path.join("dist", "README.md"));
  yield* Effect.log("[Build] Build completed.");
}).pipe(Effect.provide(NodeContext.layer));

Effect.runPromise(program).catch(console.error);
