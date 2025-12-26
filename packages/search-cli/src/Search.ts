import {
	type HttpBody,
	type HttpClientError,
	HttpClientRequest,
	HttpClientResponse,
} from "@effect/platform";
import { Context, Data, Effect, Layer, type ParseResult, Schema } from "effect";
import { ApiClient } from "./HttpService.js";

export const SearchResult = Schema.Struct({
	url: Schema.String,
	title: Schema.String,
	excerpt: Schema.String,
	score: Schema.Number,
	llmsTxt: Schema.String,
});

export const SearchResponse = Schema.Struct({
	results: Schema.Array(SearchResult),
	totalResults: Schema.Number,
});

export type SearchResult = typeof SearchResult.Type;
export type SearchResponse = typeof SearchResponse.Type;

export class SearchRequestError extends Data.TaggedError("SearchRequestError")<{
	readonly cause: HttpClientError.HttpClientError | HttpBody.HttpBodyError;
}> {}

export class SearchParseError extends Data.TaggedError("SearchParseError")<{
	readonly cause: ParseResult.ParseError | HttpClientError.ResponseError;
}> {}

export type SearchError = SearchRequestError | SearchParseError;

export class SearchClient extends Context.Tag("SearchClient")<
	SearchClient,
	{
		readonly search: (
			query: string,
		) => Effect.Effect<SearchResponse, SearchError>;
	}
>() {}

export const SearchClientLive = Layer.effect(
	SearchClient,
	Effect.gen(function* () {
		const { client } = yield* ApiClient;
		return {
			search: (query: string) =>
				HttpClientRequest.post("/api/search").pipe(
					HttpClientRequest.bodyJson({ query }),
					Effect.flatMap(client.execute),
					Effect.mapError((e) => new SearchRequestError({ cause: e })),
					Effect.flatMap(HttpClientResponse.schemaBodyJson(SearchResponse)),
					Effect.mapError((e) =>
						e instanceof SearchRequestError
							? e
							: new SearchParseError({ cause: e }),
					),
				),
		};
	}),
);
