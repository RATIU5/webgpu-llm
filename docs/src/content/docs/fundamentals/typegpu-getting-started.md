---
title: TypeGPU Getting Started
sidebar:
  order: 60
---

## Overview

TypeGPU is a modular toolkit for WebGPU that brings type safety and developer-friendly abstractions to GPU programming. Developed by Software Mansion, it provides advanced type inference and enables writing GPU shaders directly in TypeScript through TGSL (TypeGPU Shading Language).

:::tip[Why TypeGPU?]
TypeGPU mirrors WGSL syntax in TypeScript while providing compile-time type checking, autocomplete support, and static analysis. The toolkit is non-opinionated, allowing incremental adoption and the ability to eject to vanilla WebGPU at any point.
:::

**Primary use cases:**
- **Foundation for new projects** — Handles data serialization, buffer management, and shader composition
- **Integration with existing code** — Type-safe APIs can be adopted independently
- **Library interoperability** — Enables typed data sharing between WebGPU libraries

## Core Features

### Type-Safe Data Schemas

TypeGPU uses composable data schemas to manage data transfer between CPU and GPU. Every WGSL data type is represented as JavaScript schemas imported from `typegpu/data`:

```typescript title="Defining a typed schema"
import { struct, f32, vec3f, arrayOf } from "typegpu/data";

const Particle = struct({
  position: vec3f,
  velocity: vec3f,
  mass: f32,
});

// TypeScript type is automatically inferred
type ParticleData = typeof Particle.infer;
```

Schemas provide automatic serialization/deserialization, compile-time validation, and self-documenting code.

### TGSL: TypeScript Shaders

TGSL allows writing GPU shader code in TypeScript. Functions marked with `'use gpu'` are transpiled to WGSL at build time:

```typescript title="Writing a TGSL shader" {3}
const squareNumbers = tgpu
  .fn([inputBuffer, outputBuffer])
  .does(() => {
    "use gpu";
    const idx = builtin.globalInvocationId.x;
    outputBuffer[idx] = inputBuffer[idx] * inputBuffer[idx];
  })
  .$name("squareNumbers");
```

:::note[Vector Operations]
JavaScript lacks operator overloading, so vector and matrix operations require functions from `typegpu/std` (like `add`, `mul`). As of version 0.7+, vectors and matrices also support method chaining: `v1.mul(2).add(v2)`.
:::

### Bindless Resources

TypeGPU uses descriptive string keys instead of numeric binding indices:

```typescript title="Named resource bindings"
const resources = {
  particles: root.createBuffer(particleSchema).$usage("storage"),
  forces: root.createBuffer(forceSchema).$usage("storage"),
};
```

This improves code readability and reduces binding errors.

## Installation

```bash title="Install TypeGPU"
npm install typegpu
```

```bash title="Install bundler plugin for TGSL"
npm install --save-dev unplugin-typegpu
```

```bash title="Add WebGPU type definitions"
npm install --save-dev @webgpu/types
```

## Bundler Configuration

TypeGPU's shader transpilation requires `unplugin-typegpu`. The plugin supports Vite, Webpack, Rollup, esbuild, and other bundlers via unplugin.

<details>
<summary>**Vite (Recommended)**</summary>

```typescript title="vite.config.ts"
import { defineConfig } from "vite";
import typegpuPlugin from "unplugin-typegpu/vite";

export default defineConfig({
  plugins: [
    typegpuPlugin({
      autoNamingEnabled: true,
      earlyPruning: true,
    }),
  ],
});
```

| Option | Default | Description |
|--------|---------|-------------|
| `autoNamingEnabled` | `true` | Names resources based on variable names |
| `earlyPruning` | `true` | Skips files without TypeGPU imports |
| `include` | `/\.m?[jt]sx?$/` | File patterns to process |
| `exclude` | — | Patterns to skip |

</details>

<details>
<summary>**Webpack**</summary>

```javascript title="webpack.config.js"
const TypeGPUPlugin = require("unplugin-typegpu/webpack");

module.exports = {
  plugins: [
    TypeGPUPlugin({
      autoNamingEnabled: true,
      earlyPruning: true,
    }),
  ],
};
```

</details>

<details>
<summary>**Babel (React Native)**</summary>

```javascript title="babel.config.js"
module.exports = (api) => {
  api.cache(true);
  return {
    presets: ["@babel/preset-typescript"],
    plugins: ["unplugin-typegpu/babel"],
  };
};
```

</details>

## First Program

A complete example that squares an array of numbers on the GPU:

```typescript title="complete-example.ts" {8-11,17-22,25-32}
import tgpu from "typegpu";
import { arrayOf, f32 } from "typegpu/data";

async function main() {
  const root = await tgpu.init();

  const inputData = [1, 2, 3, 4, 5];

  // Create buffers with type schemas
  const inputBuffer = root
    .createBuffer(arrayOf(f32, 5), inputData)
    .$usage("storage");

  const outputBuffer = root
    .createBuffer(arrayOf(f32, 5))
    .$usage("storage");

  // Define compute shader
  const squareNumbers = tgpu
    .fn([inputBuffer, outputBuffer])
    .does(() => {
      "use gpu";
      const idx = builtin.globalInvocationId.x;
      outputBuffer[idx] = inputBuffer[idx] * inputBuffer[idx];
    })
    .$name("squareNumbers");

  // Create and execute pipeline
  const pipeline = root.makeComputePipeline(squareNumbers).$workgroupSize(1);

  root
    .createCommandEncoder()
    .beginComputePass()
    .setPipeline(pipeline)
    .dispatchWorkgroups(5)
    .end()
    .submit();

  const results = await outputBuffer.read();
  console.log("Output:", Array.from(results)); // [1, 4, 9, 16, 25]

  root.destroy();
}

main();
```

### Key Components

| Component | Description |
|-----------|-------------|
| `tgpu.init()` | Requests a GPU device and returns a root object managing all TypeGPU operations |
| `arrayOf(f32, 5)` | Defines an array of 5 floats; used for buffer sizes, validation, and serialization |
| `$usage("storage")` | Sets buffer usage flags (`storage`, `uniform`, `vertex`, `copy-src`, `copy-dst`) |
| `"use gpu"` directive | Marks functions for TGSL transpilation |
| `makeComputePipeline()` | Creates a compute pipeline from a TGSL function |
| `buffer.read()` | Asynchronously retrieves data from GPU to CPU |

## Initialization Patterns

<details>
<summary>**Basic initialization**</summary>

```typescript
const root = await tgpu.init();
```

</details>

<details>
<summary>**Custom device options**</summary>

```typescript title="High-performance with features"
const root = await tgpu.init({
  adapter: { powerPreference: "high-performance" },
  device: {
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      maxStorageBufferBindingSize: 1024 * 1024 * 1024,
    },
  },
});
```

</details>

<details>
<summary>**From existing device**</summary>

```typescript title="Integrating with existing WebGPU code"
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const root = tgpu.initFromDevice(device);
```

</details>

## TypeScript Configuration

```json title="tsconfig.json"
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "types": ["@webgpu/types"],
    "strict": true
  }
}
```

For Vite projects, add to `src/vite-env.d.ts`:

```typescript title="src/vite-env.d.ts"
/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />
```

## Working with Schemas

### Primitive Types

```typescript title="Available primitives"
import { f32, i32, u32, bool, vec2f, vec3f, vec4f, mat4x4f } from "typegpu/data";
```

### Structs

```typescript title="Defining a struct schema"
import { struct, f32, vec3f } from "typegpu/data";

const Material = struct({
  albedo: vec3f,
  roughness: f32,
  metallic: f32,
});
```

### Arrays

```typescript title="Fixed-size arrays"
import { arrayOf, f32 } from "typegpu/data";

const FloatArray = arrayOf(f32, 100);
```

### Type Extraction

```typescript title="Extracting TypeScript types from schemas" {8}
const Particle = struct({
  position: vec3f,
  velocity: vec3f,
  mass: f32,
});

type ParticleData = typeof Particle.infer;

const particle: ParticleData = {
  position: [0, 0, 0],
  velocity: [1, 0, 0],
  mass: 1.0,
};
```

## Buffer Usage

Buffers require appropriate usage flags for different operations:

```typescript title="Buffer usage patterns"
// Storage buffer for shader read/write
const storage = root.createBuffer(schema).$usage("storage");

// Uniform buffer for constants
const uniform = root.createBuffer(schema).$usage("uniform");

// Buffer readable from CPU
const readable = root
  .createBuffer(schema)
  .$usage("storage")
  .$usage("copy-src");

// Buffer writable from CPU after creation
const writable = root
  .createBuffer(schema)
  .$usage("storage")
  .$usage("copy-dst");
```

## Naming Resources

Use `$name()` for debugging:

```typescript title="Named resources for debugging"
const particleBuffer = root
  .createBuffer(particleSchema)
  .$usage("storage")
  .$name("particlePositions");

const updateShader = tgpu
  .fn([particleBuffer])
  .does(() => { "use gpu"; /* ... */ })
  .$name("updateParticles");
```

:::tip[Auto-naming]
With `autoNamingEnabled` in the bundler plugin, many names are added automatically based on variable names.
:::

## Project Structure

```
project-root/
├── src/
│   ├── gpu/
│   │   ├── schemas/          # Data type definitions
│   │   ├── shaders/          # TGSL functions
│   │   │   ├── compute/
│   │   │   └── render/
│   │   └── pipelines/        # Pipeline configurations
│   └── main.ts
├── vite.config.ts
└── tsconfig.json
```

**Principles:**
- Separate GPU and CPU code
- Group shaders by purpose
- Centralize schema definitions
- Write reusable shader functions

## TGSL Requirements

:::caution[Required for TGSL]
TGSL functions require:
1. The `'use gpu'` directive as the first statement
2. The `unplugin-typegpu` bundler plugin configured
3. Buffer dependencies passed to `tgpu.fn()`
:::

```typescript title="Correct vs incorrect TGSL"
// ✓ Correct
const shader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  // Shader code
});

// ✗ Missing directive - won't transpile
const broken = tgpu.fn([buffer]).does(() => {
  // No 'use gpu' - this runs on CPU only
});
```

## Resources

:::note[Official Resources]
- **Documentation**: [docs.swmansion.com/TypeGPU](https://docs.swmansion.com/TypeGPU/)
- **GitHub**: [github.com/software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- **npm**: [npmjs.com/package/typegpu](https://www.npmjs.com/package/typegpu)
- **Discord**: Software Mansion Community Discord
:::
