---
title: TypeGPU Getting Started
sidebar:
  order: 60
---

## Overview

TypeGPU is a modular toolkit for WebGPU that brings type safety and developer-friendly abstractions to GPU programming. Developed by Software Mansion, it provides advanced type inference and enables writing GPU shaders directly in TypeScript through TGSL (TypeGPU Shading Language).

TypeGPU mirrors WGSL syntax in TypeScript while providing compile-time type checking, autocomplete support, and static analysis. The toolkit is designed to be non-opinionated, allowing incremental adoption and the ability to eject to vanilla WebGPU at any point.

Primary use cases:
- **Foundation for new projects**: Handles data serialization, buffer management, and shader composition
- **Integration with existing code**: Type-safe APIs can be adopted independently
- **Library interoperability**: Enables typed data sharing between WebGPU libraries

## Core Features

### Type-Safe Data Schemas

TypeGPU uses composable data schemas to manage data transfer between CPU and GPU. Every WGSL data type is represented as JavaScript schemas imported from `typegpu/data`:

```typescript
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

```typescript
const squareNumbers = tgpu
  .fn([inputBuffer, outputBuffer])
  .does(() => {
    "use gpu";
    const idx = builtin.globalInvocationId.x;
    outputBuffer[idx] = inputBuffer[idx] * inputBuffer[idx];
  })
  .$name("squareNumbers");
```

TGSL provides unified language development, full IDE support, code reuse between CPU and GPU, and gradual adoption alongside WGSL.

Note: JavaScript lacks operator overloading, so vector and matrix operations require functions from `typegpu/std` (like `add`, `mul`). As of version 0.7+, vectors and matrices also support method chaining: `v1.mul(2).add(v2)`.

### Bindless Resources

TypeGPU uses descriptive string keys instead of numeric binding indices:

```typescript
const resources = {
  particles: root.createBuffer(particleSchema).$usage("storage"),
  forces: root.createBuffer(forceSchema).$usage("storage"),
};
```

This improves code readability and reduces binding errors.

## Installation

```bash
npm install typegpu
```

For TGSL functionality, install the bundler plugin:

```bash
npm install --save-dev unplugin-typegpu
```

Add WebGPU type definitions:

```bash
npm install --save-dev @webgpu/types
```

## Bundler Configuration

TypeGPU's shader transpilation requires `unplugin-typegpu`. The plugin supports Vite, Webpack, Rollup, esbuild, and other bundlers via unplugin.

### Vite (Recommended)

```typescript
// vite.config.ts
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

Configuration options:
- `autoNamingEnabled` (default: `true`): Names resources based on variable names
- `earlyPruning` (default: `true`): Skips files without TypeGPU imports
- `include` (default: `/\.m?[jt]sx?$/`): File patterns to process
- `exclude`: Patterns to skip

### Webpack

```javascript
// webpack.config.js
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

### Babel (React Native)

```javascript
// babel.config.js
module.exports = (api) => {
  api.cache(true);
  return {
    presets: ["@babel/preset-typescript"],
    plugins: ["unplugin-typegpu/babel"],
  };
};
```

## First Program

A complete example that squares an array of numbers on the GPU:

```typescript
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

**Initialization**: `tgpu.init()` requests a GPU device and returns a root object managing all TypeGPU operations.

**Data Schemas**: `arrayOf(f32, 5)` defines an array of 5 floats. TypeGPU uses schemas to determine buffer sizes, validate data, and handle serialization.

**Buffer Creation**: `root.createBuffer(schema, data).$usage("storage")` creates a GPU buffer. The `$usage()` method sets buffer usage flags:
- `'storage'`: Read/write shader access
- `'uniform'`: Read-only constants
- `'vertex'`: Vertex data
- `'copy-src'` / `'copy-dst'`: Transfer operations

**TGSL Functions**: The `'use gpu'` directive marks functions for transpilation. Buffer dependencies are passed to `tgpu.fn()`.

**Pipeline Execution**: `makeComputePipeline()` creates a pipeline; `dispatchWorkgroups()` executes it.

**Reading Results**: `buffer.read()` asynchronously retrieves data from GPU to CPU.

## Initialization Patterns

### Basic

```typescript
const root = await tgpu.init();
```

### Custom Device Options

```typescript
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

### From Existing Device

When integrating with existing WebGPU code:

```typescript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const root = tgpu.initFromDevice(device);
```

## TypeScript Configuration

Required settings:

```json
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

For Vite, add to `src/vite-env.d.ts`:

```typescript
/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />
```

## Working with Schemas

### Primitive Types

```typescript
import { f32, i32, u32, bool, vec2f, vec3f, vec4f, mat4x4f } from "typegpu/data";
```

### Structs

```typescript
import { struct, f32, vec3f } from "typegpu/data";

const Material = struct({
  albedo: vec3f,
  roughness: f32,
  metallic: f32,
});
```

### Arrays

```typescript
import { arrayOf, f32 } from "typegpu/data";

const FloatArray = arrayOf(f32, 100);
```

### Type Extraction

```typescript
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

```typescript
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

```typescript
const particleBuffer = root
  .createBuffer(particleSchema)
  .$usage("storage")
  .$name("particlePositions");

const updateShader = tgpu
  .fn([particleBuffer])
  .does(() => { "use gpu"; /* ... */ })
  .$name("updateParticles");
```

With `autoNamingEnabled` in the bundler plugin, many names are added automatically based on variable names.

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

Principles:
- Separate GPU and CPU code
- Group shaders by purpose
- Centralize schema definitions
- Write reusable shader functions

## TGSL Requirements

TGSL functions require:

1. The `'use gpu'` directive as the first statement
2. The `unplugin-typegpu` bundler plugin configured
3. Buffer dependencies passed to `tgpu.fn()`

```typescript
// Correct
const shader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  // Shader code
});

// Missing directive - won't transpile
const broken = tgpu.fn([buffer]).does(() => {
  // No 'use gpu' - this runs on CPU only
});
```

## Resources

- **Documentation**: [docs.swmansion.com/TypeGPU](https://docs.swmansion.com/TypeGPU/)
- **GitHub**: [github.com/software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- **npm**: [npmjs.com/package/typegpu](https://www.npmjs.com/package/typegpu)
- **Discord**: Software Mansion Community Discord

---

TypeGPU brings type safety to WebGPU development, catching errors at compile time rather than runtime. Its modular design allows incremental adoption—start with buffer management, add TGSL shaders when comfortable, and expand as needed. The ability to eject to vanilla WebGPU ensures you're never locked in.
