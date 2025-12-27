---
title: Debugging GPU Code
sidebar:
  order: 30
---

## Overview

GPU debugging differs fundamentally from CPU debugging due to massive parallelism, separate execution contexts, and limited introspection capabilities. This guide covers validation layers, TypeGPU debugging features, visual debugging techniques, and browser developer tools.

:::note[Why GPU Debugging is Different]
- **No breakpoints**: Hardware doesn't support pausing individual shader invocations
- **Parallel execution**: Thousands of threads run simultaneously in lockstep
- **Separate process**: GPU executes in a different process from JavaScript
- **Deferred output**: Logs appear after GPU execution completes, not during
:::

## Validation Layers

WebGPU validates all operations and reports detailed errors through the browser console:

```typescript title="Error scope pattern" {1,6-8}
device.pushErrorScope("validation");

const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.UNIFORM,  // Missing COPY_DST
});
device.queue.writeBuffer(buffer, 0, data);  // Will fail

const error = await device.popErrorScope();
if (error) {
  console.error("Validation error:", error.message);
}
```

### Common Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Missing usage flag | Buffer created without required usage | Add `GPUBufferUsage.COPY_DST` for writes |
| Binding mismatch | Shader bindings don't match bind group | Ensure `@group`/`@binding` match JavaScript |
| Format incompatible | Texture view format differs from texture | Use matching formats |
| Size mismatch | Buffer too small for shader expectations | Check minimum binding sizes |

## TypeGPU Console.log

TypeGPU enables `console.log` directly in GPU shaders:

```typescript title="GPU console.log" {5,8}
import tgpu from "typegpu";
import * as d from "typegpu/data";

const processData = tgpu.fn([d.f32], d.f32).does((input) => {
  console.log("Processing input:", input);

  const result = input * 2.0;
  console.log("Computed result:", result);

  return result;
});
```

:::caution[Console.log Limitations]
- **Performance overhead**: Buffer writes consume GPU time
- **Fixed buffer size**: Excessive logging truncates output
- **Deferred output**: Messages appear after execution completes
- **Parallel flood**: Millions of invocations can overwhelm the log

Use conditional logging to limit output volume:

```typescript
if (pixelIndex < 10) {
  console.log("Processing pixel:", pixelIndex);
}
```
:::

## CPU Simulation

TypeGPU's `tgpu.simulate()` runs shader code on CPU with full debugging:

```typescript title="CPU simulation with debugging" {10-13}
import tgpu from "typegpu";
import * as d from "typegpu/data";

const computeGradient = tgpu.fn([d.f32, d.f32], d.f32).does((x, y) => {
  const dx = x * 2.0;
  const dy = y * 3.0;
  return Math.sqrt(dx * dx + dy * dy);
});

// Simulate on CPU - set breakpoints, inspect variables
const result = tgpu["~unstable"].simulate(() => {
  return computeGradient(5.0, 3.0);
});
console.log("Simulated result:", result);
```

:::tip[When to Use Simulation]
- **Algorithm verification**: Test with known inputs, verify outputs
- **Edge case testing**: Test boundary conditions systematically
- **Initial development**: Iterate quickly before GPU deployment
- **Step-through debugging**: Use browser DevTools breakpoints

**Limitations**: Simulation runs sequentially, so it won't catch race conditions, memory access patterns, or GPU-specific precision differences.
:::

## Visual Debugging

Encode shader values as colors to visualize program state:

```wgsl title="Visualize normals as RGB"
@fragment
fn debugNormals(@location(0) normal: vec3<f32>) -> @location(0) vec4<f32> {
  // Transform [-1,1] to [0,1] range
  let r = normal.x * 0.5 + 0.5;
  let g = normal.y * 0.5 + 0.5;
  let b = normal.z * 0.5 + 0.5;
  return vec4<f32>(r, g, b, 1.0);
}
```

### Common Visualization Techniques

<details>
<summary>**Depth Visualization**</summary>

```wgsl
@fragment
fn debugDepth(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let depth = pos.z;  // 0 = near, 1 = far
  return vec4<f32>(depth, depth, depth, 1.0);
}
```

</details>

<details>
<summary>**UV Coordinate Debugging**</summary>

```wgsl
@fragment
fn debugUVs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(uv.x, uv.y, 0.0, 1.0);
}
```

</details>

<details>
<summary>**Heat Map (Value Range)**</summary>

```wgsl
fn heatMap(value: f32) -> vec4<f32> {
  let t = clamp(value, 0.0, 1.0);

  // Blue -> Cyan -> Green -> Yellow -> Red
  var r = 0.0; var g = 0.0; var b = 0.0;

  if (t < 0.25) {
    b = 1.0; g = t * 4.0;
  } else if (t < 0.5) {
    g = 1.0; b = (0.5 - t) * 4.0;
  } else if (t < 0.75) {
    g = 1.0; r = (t - 0.5) * 4.0;
  } else {
    r = 1.0; g = (1.0 - t) * 4.0;
  }

  return vec4<f32>(r, g, b, 1.0);
}
```

</details>

## Shader Compilation Errors

### Error Types

| Type | Example | Fix |
|------|---------|-----|
| Parser | Missing semicolon, invalid token | Check WGSL syntax |
| Type | Returning `f32` from `i32` function | Use explicit casts |
| Semantic | Duplicate binding, invalid address space | Check resource declarations |

### Debugging Strategy

1. **Isolate**: Comment out code sections until error disappears
2. **Minimize**: Create smallest shader that reproduces the error
3. **Verify types**: Add explicit type annotations
4. **Check bindings**: Ensure `@group`/`@binding` are unique and match JavaScript

:::tip[Inspect Generated WGSL]
For TypeGPU, inspect the generated WGSL to trace errors:

```typescript
const wgsl = tgpu.resolve(myShaderFunction);
console.log(wgsl);
```
:::

## Runtime Errors

### Device Lost

```typescript title="Handle device loss" {1-7}
device.lost.then((info) => {
  console.error("Device lost:", info.reason, info.message);

  if (info.reason !== "destroyed") {
    // Attempt recovery
    reinitializeWebGPU();
  }
});
```

**Common causes**: Driver crash, GPU timeout, resource exhaustion

### Out of Memory

```typescript title="Handle OOM errors" {1,9-11}
device.pushErrorScope("out-of-memory");

const largeTexture = device.createTexture({
  size: { width: 8192, height: 8192 },
  format: "rgba16float",
  usage: GPUTextureUsage.STORAGE_BINDING,
});

const error = await device.popErrorScope();
if (error) {
  // Use smaller texture
  createFallbackTexture();
}
```

## Browser Developer Tools

### WebGPU Inspector

Available for Chrome, Firefox, and Safari:
- **Inspection Mode**: View live GPU objects
- **Capture Mode**: Record GPU commands per frame
- **Shader Editing**: Edit and reload shaders live
- **Performance**: Plot frame times and object counts

:::note[Installation]
- [Chrome Web Store](https://chromewebstore.google.com/detail/webgpu-inspector/holcbbnljhkpkjkhgkagjkhhpeochfal)
- [Firefox Add-ons](https://addons.mozilla.org/en-US/firefox/addon/webgpu-inspector/)
:::

### Platform-Specific Tools

| Browser | Tools |
|---------|-------|
| Chrome | DevTools Performance tab, `chrome://gpu` |
| Firefox | Graphics inspector, about:support |
| Safari | Web Inspector, Metal frame capture |

## Debugging Workflow

```typescript title="Debug vs Release pattern"
const DEBUG = import.meta.env.DEV;

if (DEBUG) {
  device.pushErrorScope("validation");
  // Enable detailed logging
}

// Create resources with labels
const buffer = device.createBuffer({
  label: "Particle Position Buffer",  // Shows in error messages
  size: particleCount * 16,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
```

:::tip[Label Everything]
Labels appear in error messages and developer tools, making it immediately clear which resource failed:

```typescript
const pipeline = device.createComputePipeline({
  label: "Physics Update Pipeline",
  // ...
});
```
:::

### Incremental Development

1. Start with minimal working shader
2. Add one feature at a time
3. Test after each addition
4. Use simulation for algorithm verification
5. Validate on GPU with visual debugging
6. Optimize only after confirming correctness

:::danger[Asynchronous Error Handling]
GPU errors may appear later than the causative code. Always use error scopes:

```typescript
device.pushErrorScope("validation");
// ... operations that might fail ...
const error = await device.popErrorScope();
```
:::

## Resources

:::note[Official Documentation]
- [WebGPU Inspector](https://github.com/brendan-duncan/webgpu_inspector)
- [WebGPU Error Handling](https://toji.dev/webgpu-best-practices/error-handling.html)
- [WebGPU Debugging Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-debugging.html)
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/)
:::
