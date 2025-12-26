import * as NodeContext from "@effect/platform-node/NodeContext";
import * as NodeHttpClient from "@effect/platform-node/NodeHttpClient";
import * as NodeRuntime from "@effect/platform-node/NodeRuntime";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import { run } from "./Cli.js";

const MainLayer = Layer.mergeAll(NodeContext.layer, NodeHttpClient.layer);

run(process.argv).pipe(
	Effect.provide(MainLayer),
	NodeRuntime.runMain({ disableErrorReporting: true }),
);
