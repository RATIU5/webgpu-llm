import { Args, Command, Options } from "@effect/cli";
import { FileSystem, Path } from "@effect/platform";
import { Effect, Layer } from "effect";
import { makeApiClientLive } from "./HttpService.js";
import { SearchClient, SearchClientLive } from "./Search.js";
import { ShowClient, ShowClientLive, ShowInvalidUrl } from "./Show.js";

const getPackageVersion = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const modulePath = yield* path.fromFileUrl(new URL(import.meta.url));
  const moduleDir = path.dirname(modulePath);

  const tryReadVersion = (filePath: string) =>
    fs.readFileString(filePath).pipe(
      Effect.map((content) => JSON.parse(content) as { version: string }),
      Effect.map((pkg) => pkg.version),
    );

  const sameDir = path.join(moduleDir, "package.json");
  const parentDir = path.join(moduleDir, "..", "package.json");

  return yield* tryReadVersion(sameDir).pipe(
    Effect.orElse(() => tryReadVersion(parentDir)),
    Effect.orElse(() => Effect.succeed("0.0.0")),
  );
});

const command = Command.make("sls");

const searchDomain = Options.text("domain").pipe(
  Options.withAlias("d"),
  Options.withDescription("The root domain for the documentation API."),
);

const queryArg = Args.text({ name: "query" }).pipe(
  Args.withDescription(
    "The text used as the search term. Shorter queries are better.",
  ),
);

const searchCommand = Command.make(
  "search",
  { query: queryArg, domain: searchDomain },
  ({ domain, query }) =>
    Effect.gen(function* () {
      const searchClient = yield* SearchClient;
      const response = yield* searchClient.search(query);
      const results = response.results.map((d) => ({
        ...d,
        url: `${domain}${d.url}`,
        llmsTxt: `${domain}${d.llmsTxt}`,
      }));
      console.log({ results, totalResults: response.totalResults });
    }).pipe(
      Effect.provide(
        SearchClientLive.pipe(Layer.provide(makeApiClientLive(domain))),
      ),
      Effect.catchTags({
        SearchApiError: (e) => {
          const msg = e.details ? `${e.error}: ${e.details}` : e.error;
          const retryMsg = e.retryAfter
            ? ` (retry after ${e.retryAfter}s)`
            : "";
          return Effect.logError(`API Error: ${msg}${retryMsg}`).pipe(
            Effect.andThen(Effect.void),
          );
        },
        SearchParseError: (e) =>
          Effect.logError(e.cause).pipe(Effect.andThen(Effect.void)),
        SearchRequestError: (e) =>
          Effect.logError(e.cause).pipe(Effect.andThen(Effect.void)),
      }),
    ),
).pipe(
  Command.withDescription(
    "Search the documentation for a specific keyword or string and get a list of search results.",
  ),
);

const showUrl = Args.text({ name: "url" }).pipe(
  Args.withDescription("A full URL path to an llms.txt resource."),
);

const showCommand = Command.make("show", { showUrl }, ({ showUrl }) =>
  Effect.gen(function* () {
    const showClient = yield* ShowClient;
    if (!showUrl.includes("_llms-txt") && !showUrl.includes("llms.txt")) {
      return yield* new ShowInvalidUrl({ cause: "Not a valid llms.txt file" });
    }
    const response = yield* showClient.fetch(showUrl);
    console.log(response);
  }).pipe(
    Effect.provide(ShowClientLive),
    Effect.catchTags({
      ShowInvalidUrl: (e) =>
        Effect.logError(e.cause).pipe(Effect.andThen(Effect.void)),
      ShowRequestError: (e) =>
        Effect.logError(e.cause).pipe(Effect.andThen(Effect.void)),
    }),
  ),
).pipe(
  Command.withDescription(
    "Show the LLM documentation for a specific url provided.",
  ),
);

const commandWithSearch = command.pipe(
  Command.withSubcommands([searchCommand, showCommand]),
);

export const run = (args: ReadonlyArray<string>) =>
  Effect.gen(function* () {
    const version = yield* getPackageVersion;
    const cli = Command.run(commandWithSearch, {
      name: "Starlight Search",
      version,
    });
    yield* cli(args);
  });
