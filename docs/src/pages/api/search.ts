import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { APIRoute } from "astro";
import { RateLimiterMemory } from "rate-limiter-flexible";
import { searchPagefind } from "../../lib/pagefind-server";

export const prerender = false;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pagefindDir = path.resolve(__dirname, "../../../client/pagefind");

const rateLimiter = new RateLimiterMemory({
	points: 20,
	duration: 60,
});

function getClientIP(request: Request): string {
	return (
		request.headers.get("x-forwarded-for")?.split(",")[0].trim() ||
		request.headers.get("x-real-ip") ||
		"unknown"
	);
}

export const POST: APIRoute = async ({ request }) => {
	const clientIP = getClientIP(request);

	try {
		await rateLimiter.consume(clientIP);
	} catch {
		return new Response(
			JSON.stringify({
				error: "Rate limit exceeded",
				retryAfter: 60,
			}),
			{
				status: 429,
				headers: {
					"Content-Type": "application/json",
					"Retry-After": "60",
				},
			},
		);
	}

	const body = (await request.json()) as {
		query?: string;
	};

	if (!body?.query) {
		return new Response(
			JSON.stringify({
				error: "No query provided",
			}),
			{ status: 400, headers: { "Content-Type": "application/json" } },
		);
	}

	try {
		const searchResults = await searchPagefind(body.query, pagefindDir);
		return new Response(JSON.stringify(searchResults), {
			headers: { "Content-Type": "application/json" },
		});
	} catch (error) {
		console.error("Search error:", error);
		return new Response(
			JSON.stringify({
				error: "Search failed",
				details: error instanceof Error ? error.message : "Unknown error",
			}),
			{ status: 500, headers: { "Content-Type": "application/json" } },
		);
	}
};
