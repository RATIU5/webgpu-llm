---
title: TypeGPU Tooling
sidebar:
  order: 25
---

## Overview

TypeGPU provides two official tools for enhanced development workflows:
- **unplugin-typegpu**: Build plugin enabling JavaScript-to-WGSL transpilation
- **tgpu-gen**: CLI for generating TypeGPU definitions from existing WGSL files

## unplugin-typegpu

Build plugin that hooks into your bundler to enable TGSL (TypeGPU Shading Language) features.

### Installation

```bash
npm install --save-dev unplugin-typegpu
```

### Bundler Configuration

<details>
<summary>**Vite**</summary>

```javascript title="vite.config.js"
import { defineConfig } from "vite";
import typegpuPlugin from "unplugin-typegpu/vite";

export default defineConfig({
  plugins: [typegpuPlugin({})],
});
```

</details>

<details>
<summary>**Webpack**</summary>

```javascript title="webpack.config.js"
const typegpuPlugin = require("unplugin-typegpu/webpack");

module.exports = {
  plugins: [typegpuPlugin({})],
};
```

</details>

<details>
<summary>**Rollup**</summary>

```javascript title="rollup.config.js"
import typegpuPlugin from "unplugin-typegpu/rollup";

export default {
  plugins: [typegpuPlugin({})],
};
```

</details>

<details>
<summary>**esbuild**</summary>

```javascript title="esbuild.config.js"
import esbuild from "esbuild";
import typegpuPlugin from "unplugin-typegpu/esbuild";

esbuild.build({
  plugins: [typegpuPlugin({})],
});
```

</details>

<details>
<summary>**Babel** (React Native)</summary>

```javascript title="babel.config.js"
module.exports = {
  plugins: ["unplugin-typegpu/babel"],
};
```

</details>

### Plugin Options

```javascript title="Configuration options"
typegpuPlugin({
  include: ["**/*.ts", "**/*.tsx"],  // Files to process
  exclude: ["node_modules/**"],       // Files to skip
  autoNamingEnabled: true,            // Auto-name resources from variables
  earlyPruning: true,                 // Skip files without TypeGPU code
})
```

### Features

#### 1. "use gpu" Directive

Write shader functions in TypeScript that transpile to WGSL:

```typescript title="Shell-less GPU functions" {2}
const computeDistance = (a: Vec2f, b: Vec2f) => {
  "use gpu";
  const diff = std.sub(a, b);
  return std.length(diff);
};

// Works both on CPU and GPU
const cpuResult = computeDistance(vec2f(0, 0), vec2f(3, 4)); // 5.0

// Also generates WGSL for GPU execution
```

:::note[Migration from "kernel"]
The `"use gpu"` directive replaces the older `"kernel"` directive with the same functionality.
:::

#### 2. Automatic Resource Naming

Without plugin:
```typescript
const positionBuffer = root.createBuffer(d.arrayOf(d.vec3f, 100))
  .$name("positionBuffer");  // Manual naming required
```

With plugin:
```typescript
const positionBuffer = root.createBuffer(d.arrayOf(d.vec3f, 100));
// Automatically named "positionBuffer" from variable name
```

#### 3. Automatic External Detection

Without plugin:
```typescript
const shader = tgpu.fn([d.u32], d.void).does`(idx: u32) {
  positions[idx] = velocities[idx];
}`.$uses({ positions: posBuffer, velocities: velBuffer });  // Manual
```

With plugin:
```typescript
const shader = tgpu.fn([d.u32], d.void).does`(idx: u32) {
  ${posBuffer}[idx] = ${velBuffer}[idx];
}`;  // Externals detected automatically
```

## tgpu-gen CLI

Generates TypeGPU TypeScript definitions from existing WGSL shader files.

### Installation

```bash
# Use directly with npx
npx tgpu-gen shader.wgsl

# Or install globally
npm install -g tgpu-gen
```

### Basic Usage

```bash title="Generate from single file"
tgpu-gen path/to/shader.wgsl
# Creates shader.ts in same directory
```

```bash title="Batch processing with globs"
tgpu-gen "shaders/*.wgsl" -o "generated/*.ts"

# Recursive
tgpu-gen "src/**/*.wgsl" -o "types/**/*.ts"
```

### Watch Mode

```bash title="Continuous generation"
tgpu-gen "shaders/*.wgsl" --output "generated/*.ts" --watch
```

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output path pattern |
| `--keep` | Skip existing files |
| `--overwrite` | Replace existing files |
| `--commonjs` | Generate CommonJS instead of ES modules |

### Supported Extensions

Input: `.wgsl`
Output: `.ts`, `.js`, `.mjs`, `.cjs`, `.mts`, `.cts`

### Example

```wgsl title="input.wgsl"
struct Particle {
  position: vec3f,
  velocity: vec3f,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
  particles[id.x].position += particles[id.x].velocity;
}
```

```bash
tgpu-gen input.wgsl
```

```typescript title="input.ts (generated)"
import * as d from "typegpu/data";
import tgpu from "typegpu";

export const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
});

export const updateParticles = tgpu
  .computeFn({ workgroupSize: [64] })
  .does(/* ... */);
```

## When to Use Each Tool

| Scenario | Tool |
|----------|------|
| New TypeGPU project | unplugin-typegpu |
| Migrating existing WGSL | tgpu-gen |
| React Native | unplugin-typegpu/babel |
| Maximum type safety | Both together |

:::tip[Recommended Setup]
For new projects, use unplugin-typegpu from the start. For existing WebGPU projects with WGSL shaders, use tgpu-gen to generate initial TypeGPU code, then switch to unplugin-typegpu for ongoing development.
:::
