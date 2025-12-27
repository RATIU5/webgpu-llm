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
  llmsTxt: Schema.NullOr(Schema.String),
});

export const SearchResponse = Schema.Struct({
  results: Schema.Array(SearchResult),
  totalResults: Schema.Number,
});

export const ApiErrorResponse = Schema.Struct({
  error: Schema.String,
  details: Schema.optional(Schema.String),
  retryAfter: Schema.optional(Schema.Number),
});

export type SearchResult = typeof SearchResult.Type;
export type SearchResponse = typeof SearchResponse.Type;
export type ApiErrorResponse = typeof ApiErrorResponse.Type;

export class SearchRequestError extends Data.TaggedError("SearchRequestError")<{
  readonly cause: HttpClientError.HttpClientError | HttpBody.HttpBodyError;
}> {}

export class SearchParseError extends Data.TaggedError("SearchParseError")<{
  readonly cause: ParseResult.ParseError | HttpClientError.ResponseError;
}> {}

export class SearchApiError extends Data.TaggedError("SearchApiError")<{
  readonly error: string;
  readonly details?: string | undefined;
  readonly retryAfter?: number | undefined;
}> {}

export type SearchError = SearchRequestError | SearchParseError | SearchApiError;

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
          Effect.flatMap((response) =>
            Effect.gen(function* () {
              if (response.status >= 400) {
                const errorBody = yield* HttpClientResponse.schemaBodyJson(
                  ApiErrorResponse,
                )(response);
                return yield* new SearchApiError(errorBody);
              }
              return yield* HttpClientResponse.schemaBodyJson(SearchResponse)(
                response,
              );
            }),
          ),
          Effect.mapError((e) => {
            switch (e._tag) {
              case "ParseError":
              case "ResponseError":
                return new SearchParseError({ cause: e });
              case "SearchRequestError":
              case "SearchApiError":
                return e;
            }
          }),
        ),
    };
  }),
);
