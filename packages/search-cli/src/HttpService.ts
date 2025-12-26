import { HttpClient, HttpClientRequest } from "@effect/platform";
import { Context, Effect, Layer } from "effect";

export class ApiClient extends Context.Tag("ApiClient")<
	ApiClient,
	{
		readonly client: HttpClient.HttpClient;
		readonly baseUrl: string;
	}
>() {}

export const makeApiClientLive = (baseUrl: string) =>
	Layer.effect(
		ApiClient,
		Effect.gen(function* () {
			const client = yield* HttpClient.HttpClient;
			return {
				client: client.pipe(
					HttpClient.mapRequest(HttpClientRequest.prependUrl(baseUrl)),
				),
				baseUrl,
			};
		}),
	);
