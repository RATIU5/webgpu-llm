import { HttpClient, type HttpClientError } from "@effect/platform";
import { Context, Data, Effect, Layer } from "effect";

export class ShowRequestError extends Data.TaggedError("ShowRequestError")<{
  readonly cause:
    | HttpClientError.HttpClientError
    | HttpClientError.ResponseError;
}> {}

export class ShowInvalidUrl extends Data.TaggedError("ShowInvalidUrl")<{
  readonly cause: string;
}> {}

export class ShowClient extends Context.Tag("ShowClient")<
  ShowClient,
  {
    readonly fetch: (url: string) => Effect.Effect<string, ShowRequestError>;
  }
>() {}

export const ShowClientLive = Layer.effect(
  ShowClient,
  Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient;
    return {
      fetch: (url: string) =>
        client.get(url).pipe(
          Effect.flatMap((response) => response.text),
          Effect.mapError((e) => new ShowRequestError({ cause: e })),
        ),
    };
  }),
);
