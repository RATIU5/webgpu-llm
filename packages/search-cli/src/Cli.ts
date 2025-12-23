import * as Command from "@effect/cli/Command";

const command = Command.make("sls");

export const run = Command.run(command, {
  name: "Starlight Search",
  version: "0.0.1",
});
