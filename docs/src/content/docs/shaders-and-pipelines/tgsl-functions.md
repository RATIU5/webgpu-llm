---
title: TGSL Functions
sidebar:
  order: 30
---

## Overview

TGSL (TypeGPU Shading Language) enables writing GPU shader code in TypeScript rather than WGSL. Functions written in TGSL execute on both CPU and GPU, providing type safety, IDE support, and code reusability.

:::note[Transpilation Pipeline]
TGSL code is transformed at build time:
1. **JavaScript → Tinyest AST** via `unplugin-typegpu` build plugin
2. **Tinyest → WGSL** at runtime by TypeGPU

The `'use gpu'` directive marks functions for GPU execution.
:::

## The 'use gpu' Directive

```typescript title="Basic TGSL function"
import tgpu from "typegpu";
import * as d from "typegpu/data";

const addVectors = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((a, b) => {
  "use gpu";  // First statement in function body
  return a.add(b);
});
```

### Dual-Mode Execution

Functions with `'use gpu'` run on both CPU (for testing) and GPU (as shaders):

```typescript title="Test on CPU, run on GPU"
const clampValue = tgpu.fn([d.f32, d.f32, d.f32], d.f32).does((value, min, max) => {
  "use gpu";
  if (value < min) return min;
  if (value > max) return max;
  return value;
});

// Test on CPU
console.assert(clampValue(5, 0, 10) === 5);
console.assert(clampValue(-5, 0, 10) === 0);

// Use in GPU shader
const shader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  buffer.value[idx] = clampValue(buffer.value[idx], 0.0, 1.0);
});
```

### Supported JavaScript Features

<details>
<summary>**Supported**</summary>

- Variable declarations with `const`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `<`, `>`, `<=`, `>=`, `===`, `!==`
- Logical: `&&`, `||`, `!`
- Control flow: `if/else`, `for`, `while`
- Ternary operator: `condition ? a : b`
- Function calls to other TGSL functions
- Array and struct member access
- Return statements

</details>

<details>
<summary>**Not Supported**</summary>

- Dynamic typing or type coercion
- Closures and higher-order functions
- Objects and classes (use structs)
- String manipulation
- Async/await, Promises
- Try/catch exception handling
- Most JavaScript standard library

</details>

:::danger[Build Plugin Required]
TGSL requires `unplugin-typegpu` to transpile to WGSL. Without the plugin, functions run only on CPU.
:::

## Defining Functions with tgpu.fn()

### Basic Functions

```typescript title="Function definitions"
// Scalar function
const add = tgpu.fn([d.f32, d.f32], d.f32).does((a, b) => {
  "use gpu";
  return a + b;
});

// Vector function
const distance = tgpu.fn([d.vec3f, d.vec3f], d.f32).does((a, b) => {
  "use gpu";
  const delta = b.sub(a);
  return Math.sqrt(delta.dot(delta));
});

// Void return (omit return type)
const updateParticle = tgpu.fn([particleBuffer, d.u32]).does((particles, index) => {
  "use gpu";
  particles.value[index].position = particles.value[index].position.add(velocity);
});
```

### Entry Points

<details>
<summary>**Vertex Entry Point**</summary>

```typescript
import { builtin } from "typegpu/builtin";

const VertexOutput = d.struct({
  position: builtin.position(d.vec4f),
  color: d.vec4f,
});

const vertexMain = tgpu.vertexFn([], VertexOutput).does(() => {
  "use gpu";
  const idx = builtin.vertexIndex;
  const positions = [d.vec2f(-0.5, -0.5), d.vec2f(0.5, -0.5), d.vec2f(0.0, 0.5)];
  return {
    position: d.vec4f(positions[idx].x, positions[idx].y, 0, 1),
    color: d.vec4f(1, 0, 0, 1),
  };
});
```

</details>

<details>
<summary>**Fragment Entry Point**</summary>

```typescript
const FragmentOutput = d.struct({
  color: builtin.fragColor(d.vec4f),
});

const fragmentMain = tgpu.fragmentFn([FragmentInput], FragmentOutput).does((input) => {
  "use gpu";
  return { color: input.color };
});
```

</details>

<details>
<summary>**Compute Entry Point**</summary>

```typescript
const computeMain = tgpu.computeFn([inputBuffer, outputBuffer]).does(() => {
  "use gpu";
  const id = builtin.globalInvocationId.x;
  outputBuffer.value[id] = inputBuffer.value[id] * 2.0;
}).$workgroupSize(64);
```

</details>

:::tip[Workgroup Size]
Use 64 as a default workgroup size. For 2D data, use `.$workgroupSize(8, 8)`.
:::

## Accessing GPU Resources

### The .value Property

Objects with different CPU and GPU representations (buffers, slots) must be accessed via `.value` (or `$` alias):

```typescript title="Resource access in TGSL" {8-10}
const particleBuffer = root.createBuffer(d.arrayOf(d.vec3f, 1000)).$usage("storage");
const deltaTime = root.createUniform(d.f32, 0.016);

const updateParticles = tgpu.fn([particleBuffer, deltaTime]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;

  // Use .value or $ to access GPU resources
  const position = particleBuffer.value[idx];  // or particleBuffer.$[idx]
  const dt = deltaTime.value;                   // or deltaTime.$

  particleBuffer.value[idx] = position.add(d.vec3f(0, -9.8 * dt, 0));
});
```

:::caution[Why .value?]
TypeGPU buffer objects are JavaScript proxies with CPU APIs. The `.value` property signals "access the GPU representation."
:::

## Standard Library (typegpu/std)

```typescript title="Standard library functions"
import { distance, length, normalize, dot, cross, clamp, mix, smoothstep } from "typegpu/std";

const lighting = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((normal, lightDir) => {
  "use gpu";
  const n = normalize(normal);
  const l = normalize(lightDir);
  const diffuse = clamp(dot(n, l), 0.0, 1.0);
  return d.vec3f(diffuse);
});
```

## Vector and Matrix Operations

### Chained Methods (TypeGPU 0.7+)

```typescript title="Vector operations"
const physics = tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f).does((pos, vel, dt) => {
  "use gpu";
  const gravity = d.vec3f(0, -9.8, 0);
  const newVel = vel.add(gravity.mul(dt));
  return pos.add(newVel.mul(dt));
});
```

| Operation | Method | WGSL Equivalent |
|-----------|--------|-----------------|
| Add | `a.add(b)` | `a + b` |
| Subtract | `a.sub(b)` | `a - b` |
| Multiply | `a.mul(b)` | `a * b` |
| Divide | `a.div(b)` | `a / b` |
| Length | `a.length()` | `length(a)` |
| Normalize | `a.normalize()` | `normalize(a)` |
| Dot product | `a.dot(b)` | `dot(a, b)` |
| Cross product | `a.cross(b)` | `cross(a, b)` |

### Matrix Operations

```typescript title="Matrix multiplication"
const transform = tgpu.fn([d.mat4x4f, d.vec3f], d.vec4f).does((mat, pos) => {
  "use gpu";
  return mat.mul(d.vec4f(pos.x, pos.y, pos.z, 1.0));
});
```

## Mixing WGSL and TGSL

### Call WGSL from TGSL

```typescript title="WGSL function called from TGSL"
const complexOp = tgpu.fn([d.mat4x4f, d.mat4x4f], d.mat4x4f)
  .implement(/* wgsl */ `
    fn complexOp(a: mat4x4f, b: mat4x4f) -> mat4x4f {
      return transpose(a * b);
    }
  `);

const useWgsl = tgpu.fn([d.mat4x4f, d.mat4x4f], d.mat4x4f).does((a, b) => {
  "use gpu";
  return complexOp(a, b);  // Call WGSL function
});
```

### Call TGSL from WGSL

```typescript title="TGSL function used in WGSL"
const clamp = tgpu.fn([d.f32, d.f32, d.f32], d.f32).does((val, min, max) => {
  "use gpu";
  if (val < min) return min;
  if (val > max) return max;
  return val;
});

const processValue = tgpu.fn([d.f32], d.f32)
  .implement(/* wgsl */ `
    fn processValue(input: f32) -> f32 {
      return clamp(input * 2.0, 0.0, 10.0);
    }
  `)
  .$uses({ clamp });
```

## GPU Console.log

```typescript title="Debug with console.log"
const debugShader = tgpu.computeFn([buffer]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;

  if (idx < 5) {  // Limit output
    console.log("Thread", idx, "value:", buffer.value[idx]);
  }

  buffer.value[idx] = buffer.value[idx] * 2;
}).$workgroupSize(64);
```

:::caution[Limitations]
- Buffer limit for logged values
- Performance overhead—remove for production
- Works with scalars and vectors
:::

## Common Patterns

### Function Composition

```typescript title="Reusable utilities"
const saturate = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";
  return clamp(x, 0, 1);
});

const attenuate = tgpu.fn([d.f32, d.f32], d.f32).does((dist, radius) => {
  "use gpu";
  const ratio = dist / radius;
  return saturate(1.0 - ratio * ratio);
});

const pointLight = tgpu.fn([d.vec3f, d.vec3f, d.f32], d.f32).does((fragPos, lightPos, radius) => {
  "use gpu";
  const dist = distance(fragPos, lightPos);
  return attenuate(dist, radius);
});
```

### Shared CPU/GPU Logic

```typescript title="Code reuse"
const applyGravity = tgpu.fn([d.vec3f, d.f32], d.vec3f).does((velocity, dt) => {
  "use gpu";
  return velocity.add(d.vec3f(0, -9.8, 0).mul(dt));
});

// CPU testing
function simulateOnCPU(particles, dt) {
  for (const p of particles) {
    p.velocity = applyGravity(p.velocity, dt);
  }
}

// GPU shader
const gpuUpdate = tgpu.computeFn([particleBuffer, deltaTime]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;
  particleBuffer.value[idx].velocity = applyGravity(
    particleBuffer.value[idx].velocity,
    deltaTime.value
  );
}).$workgroupSize(64);
```

## Common Pitfalls

:::danger[Forgetting .value]
```typescript
// WRONG
const value = buffer[0];

// CORRECT
const value = buffer.value[0];
```
:::

:::danger[Missing 'use gpu']
```typescript
// WRONG - runs CPU only
const shader = tgpu.fn([d.f32], d.f32).does((x) => {
  return x * 2;
});

// CORRECT
const shader = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";
  return x * 2;
});
```
:::

:::caution[Value vs Reference Semantics]
In JavaScript (CPU), vectors are references. In WGSL (GPU), they're values:

```typescript
// CPU: b references same object as a
const b = a;
b[0] = 10;  // Modifies a!

// GPU: b is a copy
const b = a;
b[0] = 10;  // a unchanged
```
:::

## Resources

:::note[Official Documentation]
- [TGSL Guide](https://docs.swmansion.com/TypeGPU/fundamentals/tgsl/)
- [Functions Documentation](https://docs.swmansion.com/TypeGPU/fundamentals/functions/)
- [typegpu/std API](https://docs.swmansion.com/TypeGPU/api/typegpu-std/)
:::
