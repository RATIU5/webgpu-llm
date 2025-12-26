import { Args, Command, Options } from "@effect/cli";
import { Effect, Layer } from "effect";
import { makeApiClientLive } from "./HttpService.js";
import { SearchClient, SearchClientLive } from "./Search.js";
import { ShowClient, ShowClientLive, ShowInvalidUrl } from "./Show.js";

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

export const run = Command.run(commandWithSearch, {
	name: "Starlight Search",
	version: "0.0.1",
});
