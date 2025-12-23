---
title: TGSL Functions
sidebar:
  order: 30
---

## Overview

TGSL (TypeGPU Shading Language) is one of the most distinctive and powerful features of TypeGPU, enabling developers to write GPU shader code in TypeScript rather than WGSL (WebGPU Shading Language). This revolutionary approach eliminates the context-switching between JavaScript application code and shader code, providing a unified development experience with full IDE support, type safety, and code reusability across CPU and GPU execution contexts.

Unlike traditional WebGPU development where shaders are written as WGSL strings—divorced from TypeScript's type system and tooling—TGSL brings GPU programming into the TypeScript ecosystem. Functions written in TGSL can operate in three distinct ways: they execute as regular JavaScript on the CPU, generate WGSL code for shader compilation, and run directly on the GPU via compute and render pipelines. This triple-mode execution unlocks unprecedented possibilities for code sharing, testing, and maintainability.

TGSL is not a complete reimplementation of WGSL in JavaScript. Rather, it's a carefully designed subset of JavaScript that transpiles cleanly to WGSL, maintaining semantic equivalence while leveraging TypeScript's powerful type inference and validation capabilities. The transformation happens at build time through the `unplugin-typegpu` bundler plugin, ensuring zero runtime overhead and producing readable, debuggable WGSL output.

## What is TGSL?

TGSL (TypeGPU Shading Language) is a subset of JavaScript designed specifically for GPU execution via TypeGPU. It enables developers to write shader functions in TypeScript that are automatically transpiled to WebGPU Shading Language (WGSL) while maintaining full type safety and IDE support.

### Transpilation: JavaScript → Tinyest AST → WGSL

The TGSL transpilation pipeline is a two-stage process that transforms JavaScript code into efficient WGSL shaders:

1. **JavaScript to Tinyest**: When you write a TGSL function, the `unplugin-typegpu` build plugin analyzes the JavaScript Abstract Syntax Tree (AST) and converts it into a compact intermediate representation called "tinyest." This format is specifically designed for GPU shader code generation and strips away JavaScript-specific features that don't translate to WGSL.

2. **Tinyest to WGSL**: At runtime, TypeGPU processes the tinyest representation and generates equivalent WGSL code. This generated code is then compiled by the WebGPU driver and executed on the GPU. The generated WGSL is readable and can be inspected for debugging purposes.

The beauty of this pipeline is that it happens automatically. You write TypeScript, mark it with `'use gpu'`, and the rest is handled for you. The generated WGSL maintains semantic equivalence to your original JavaScript code while adhering to all of WGSL's syntax and type requirements.

### Type-Safe Function Signatures

One of TGSL's most valuable features is complete type safety across the CPU-GPU boundary. When you define a TGSL function using `tgpu.fn()`, you specify typed arguments and return values using TypeGPU schemas. These signatures are visible to TypeScript's type checker, enabling:

- **Compile-time validation**: Type errors are caught during development, not at runtime
- **Autocomplete and IntelliSense**: Your IDE provides suggestions for function arguments and return types
- **Refactoring support**: Rename operations and other refactorings work seamlessly across shader code
- **Documentation**: Function signatures serve as self-documenting interfaces

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Fully typed function signature
const addVectors = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((a, b) => {
  "use gpu";
  return a.add(b);
});

// TypeScript knows the signature:
// (a: vec3f, b: vec3f) => vec3f
```

### Automatic Dependency Resolution

TGSL functions can call other TGSL functions or reference external resources like buffers, textures, and constants. TypeGPU automatically resolves these dependencies when generating WGSL code, ensuring:

- **No duplication**: Shared utility functions are included exactly once in the generated shader
- **No name clashes**: TypeGPU manages WGSL identifier naming to prevent conflicts
- **Proper ordering**: Functions are declared in the correct order for WGSL compilation
- **Resource binding**: External resources are automatically mapped to appropriate bind groups

```typescript
// Helper function
const normalize = tgpu.fn([d.vec3f], d.vec3f).does((v) => {
  "use gpu";
  const len = v.length();
  return v.div(len);
});

// Main function that uses the helper
const computeDirection = tgpu
  .fn([d.vec3f, d.vec3f], d.vec3f)
  .does((from, to) => {
    "use gpu";
    const direction = to.sub(from);
    return normalize(direction); // Automatically included in WGSL
  });
```

### Supported JavaScript Features

TGSL supports a carefully curated subset of JavaScript that maps cleanly to WGSL:

**Supported:**

- Variable declarations with `const` (WGSL `let` and `var` where appropriate)
- Arithmetic operations: `+`, `-`, `*`, `/`, `%`
- Comparison operations: `<`, `>`, `<=`, `>=`, `===`, `!==`
- Logical operations: `&&`, `||`, `!`
- Control flow: `if/else`, `for` loops, `while` loops
- Function calls to other TGSL functions
- Array and struct member access
- Return statements
- Ternary operator: `condition ? valueIfTrue : valueIfFalse`

**Not Supported:**

- Dynamic typing or type coercion
- Closures and higher-order functions (except predefined patterns)
- Objects and classes (use structs instead)
- String manipulation
- Asynchronous operations (`async`/`await`, Promises)
- Most JavaScript standard library functions
- `try`/`catch` exception handling

### Build Plugin Requirement

For TGSL functions to work, you must use the dedicated build plugin—`unplugin-typegpu`. This plugin integrates with various bundlers (Vite, Webpack, Rollup, esbuild) and performs the JavaScript-to-tinyest transformation during the build process.

Without the plugin, TGSL functions will execute on the CPU as regular JavaScript but cannot be transpiled to WGSL or executed on the GPU.

## The 'use gpu' Directive

The `'use gpu'` directive is the magic incantation that marks a JavaScript function for GPU execution. Similar to JavaScript's `'use strict'` directive, it must appear as the first statement in the function body.

### Marking Functions for GPU Execution

When TypeGPU encounters a function containing `'use gpu'`, it signals the build plugin to transpile that function to WGSL:

```typescript
const computeShader = tgpu.fn([inputBuffer, outputBuffer]).does(() => {
  "use gpu"; // This function will be transpiled to WGSL

  const idx = builtin.globalInvocationId.x;
  outputBuffer[idx] = inputBuffer[idx] * 2;
});
```

Only functions marked with `'use gpu'` can be called from within a shader. An exception to this rule is `console.log`, which allows tracking runtime behavior of shaders in a familiar way.

### Dual-Mode Execution

Functions marked with `'use gpu'` can be executed in two contexts:

1. **CPU Execution**: Called as regular JavaScript functions for testing, simulation, or host-side computation
2. **GPU Execution**: Transpiled to WGSL and executed on the GPU as part of a shader pipeline

This dual-mode capability is transformative for testing and debugging. You can write unit tests for your shader logic that run on the CPU, validating behavior before deploying to GPU:

```typescript
const clampValue = tgpu
  .fn([d.f32, d.f32, d.f32], d.f32)
  .does((value, min, max) => {
    "use gpu";
    if (value < min) return min;
    if (value > max) return max;
    return value;
  });

// Test on CPU
console.assert(clampValue(5, 0, 10) === 5);
console.assert(clampValue(-5, 0, 10) === 0);
console.assert(clampValue(15, 0, 10) === 10);

// Use in GPU shader
const shader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  const value = buffer[idx];
  buffer[idx] = clampValue(value, 0.0, 1.0);
});
```

### Restrictions and Constraints

When using `'use gpu'`, you must adhere to TGSL's constraints:

- Only use supported JavaScript features
- Avoid closures and capturing external variables (except through function parameters or `$uses`)
- Use TypeGPU data types (via `typegpu/data`) rather than raw JavaScript numbers for vectors and matrices
- Call only other functions marked with `'use gpu'` (or `console.log`)

```typescript
// INCORRECT: Captures external variable
const multiplier = 2.5;
const badShader = tgpu.fn([d.f32], d.f32).does((value) => {
  "use gpu";
  return value * multiplier; // ERROR: multiplier not accessible
});

// CORRECT: Pass as parameter or use $uses
const goodShader = tgpu
  .fn([d.f32], d.f32)
  .does((value) => {
    "use gpu";
    return value * multiplier.value;
  })
  .$uses({ multiplier: 2.5 });
```

## Defining TGSL Functions

TGSL functions are defined using the `tgpu.fn()` API, which creates type-safe function shells that can be transpiled to WGSL and executed on the GPU.

### Basic Functions with tgpu.fn()

The fundamental pattern for defining TGSL functions uses `tgpu.fn()` to create a function shell with typed arguments and return values:

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Basic arithmetic function
const add = tgpu.fn([d.f32, d.f32], d.f32).does((a, b) => {
  "use gpu";
  return a + b;
});

// Vector operation
const addVectors = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((a, b) => {
  "use gpu";
  return a.add(b);
});

// More complex function with multiple operations
const computeDistance = tgpu
  .fn([d.vec3f, d.vec3f], d.f32)
  .does((pointA, pointB) => {
    "use gpu";
    const delta = pointB.sub(pointA);
    const distanceSquared = delta.dot(delta);
    return Math.sqrt(distanceSquared);
  });
```

The `tgpu.fn()` constructor accepts:

- An array of argument types (using TypeGPU schemas from `typegpu/data`)
- An optional return type (defaults to `void` if omitted)
- Returns a `TgpuFnShell` that you chain `.does()` on to provide the implementation

### Void Return Type

For functions that don't return a value, omit the return type:

```typescript
const updateParticle = tgpu
  .fn([particleBuffer, d.u32])
  .does((particles, index) => {
    "use gpu";
    particles[index].velocity = particles[index].velocity.add(gravity);
    particles[index].position = particles[index].position.add(
      particles[index].velocity,
    );
  });
```

### Entry Points

Entry points are shader functions that serve as the main entry to compute, vertex, or fragment shaders. TypeGPU provides dedicated constructors for defining entry points.

**Note**: Entry functions are an unstable feature, and the API may be subject to change in future TypeGPU versions.

#### Vertex Entry Points

Vertex shaders process individual vertices, transforming positions and passing data to fragment shaders:

```typescript
import { builtin } from "typegpu/builtin";

const VertexOutput = d.struct({
  position: builtin.position(d.vec4f),
  color: d.vec4f,
});

const vertexMain = tgpu.vertexFn([], VertexOutput).does(() => {
  "use gpu";
  const vertexIndex = builtin.vertexIndex;

  // Simple triangle vertices
  const positions = [
    d.vec2f(-0.5, -0.5),
    d.vec2f(0.5, -0.5),
    d.vec2f(0.0, 0.5),
  ];

  const colors = [
    d.vec4f(1, 0, 0, 1), // Red
    d.vec4f(0, 1, 0, 1), // Green
    d.vec4f(0, 0, 1, 1), // Blue
  ];

  return {
    position: d.vec4f(positions[vertexIndex].x, positions[vertexIndex].y, 0, 1),
    color: colors[vertexIndex],
  };
});
```

#### Fragment Entry Points

Fragment shaders determine the color of individual pixels:

```typescript
const FragmentInput = d.struct({
  color: d.vec4f,
});

const FragmentOutput = d.struct({
  color: builtin.fragColor(d.vec4f),
});

const fragmentMain = tgpu
  .fragmentFn([FragmentInput], FragmentOutput)
  .does((input) => {
    "use gpu";
    return { color: input.color };
  });
```

#### Compute Entry Points with @workgroup_size

Compute shaders perform general-purpose GPU computations. They require a workgroup size specification:

```typescript
const computeMain = tgpu
  .computeFn([inputBuffer, outputBuffer])
  .does(() => {
    "use gpu";
    const globalId = builtin.globalInvocationId.x;
    outputBuffer[globalId] = inputBuffer[globalId] * inputBuffer[globalId];
  })
  .$workgroupSize(64); // 64 threads per workgroup
```

**Workgroup Size Considerations**:

- A `@workgroup_size` of 64 is a good all-around default if you don't have specific requirements
- Common values are powers of 2: 64, 128, 256
- Can specify 1D, 2D, or 3D workgroups: `.$workgroupSize(8, 8)` or `.$workgroupSize(4, 4, 4)`
- Larger workgroup sizes can improve GPU utilization but have hardware limits

**Dispatch Calculation**:
If a compute shader has `@workgroup_size(4, 4)` and you dispatch with `dispatchWorkgroups(8, 8)`, the entry point will be invoked 1,024 times total (4 × 4 × 8 × 8 = 1,024).

### Function Shells

Function shells are objects that hold type information about function inputs and outputs. The shell constructor `tgpu.fn` relies on TypeGPU schemas to represent WGSL data types and assist in generating shader code at runtime.

#### Type Constraints

Shells enforce type constraints at compile time, preventing type mismatches:

```typescript
// Define a shell for a specific signature
const Vec3Operation = tgpu.fn([d.vec3f, d.vec3f], d.vec3f);

// Implementations must match the signature
const add = Vec3Operation.does((a, b) => {
  "use gpu";
  return a.add(b);
});

const multiply = Vec3Operation.does((a, b) => {
  "use gpu";
  return a.mul(b);
});

// TypeScript error: wrong types
const wrong = Vec3Operation.does((a, b) => {
  "use gpu";
  return a + b; // ERROR: returns number, not vec3f
});
```

#### Reusable Patterns

Function shells enable creating reusable patterns and interfaces:

```typescript
// Define a transformation interface
const TransformFn = tgpu.fn([d.vec3f], d.vec3f);

// Multiple implementations of the same interface
const translate = TransformFn.does((point) => {
  "use gpu";
  return point.add(d.vec3f(1, 0, 0));
});

const scale = TransformFn.does((point) => {
  "use gpu";
  return point.mul(2);
});

const normalize = TransformFn.does((point) => {
  "use gpu";
  const len = point.length();
  return point.div(len);
});
```

#### Named Parameters with IORecords

For entry point functions, you can use IORecords (JavaScript objects mapping argument names to types) to describe inputs and outputs:

```typescript
const VertexInput = {
  position: d.vec3f,
  normal: d.vec3f,
  uv: d.vec2f,
};

const VertexOutput = {
  position: builtin.position(d.vec4f),
  worldPos: d.vec3f,
  normal: d.vec3f,
  uv: d.vec2f,
};

const vertexShader = tgpu
  .vertexFn([VertexInput], VertexOutput)
  .does((input) => {
    "use gpu";
    const worldPos = modelMatrix.value.mul(d.vec4f(input.position, 1.0));
    const clipPos = viewProjectionMatrix.value.mul(worldPos);

    return {
      position: clipPos,
      worldPos: worldPos.xyz,
      normal: input.normal,
      uv: input.uv,
    };
  })
  .$uses({ modelMatrix, viewProjectionMatrix });
```

## Accessing GPU Resources

GPU resources like buffers, textures, and samplers have different representations on the CPU and GPU. TGSL provides the `.value` property (or `$` alias) to access these resources within shader code.

### The .value Property (or $ Alias)

Objects that have different types on the CPU and GPU—such as buffers, textures, slots, and layouts—must be accessed via the `.value` property in TGSL functions. This differs from how they appear in WGSL-implemented functions where they're accessed directly.

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

const root = await tgpu.init();

// Create a buffer
const particleBuffer = root
  .createBuffer(d.arrayOf(d.vec3f, 1000))
  .$usage("storage");

// Create a uniform value
const deltaTime = root.createUniform(d.f32, 0.016);

// TGSL function accessing resources
const updateParticles = tgpu.fn([particleBuffer, deltaTime]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;

  // Use .value to access GPU resources
  const position = particleBuffer.value[idx];
  const dt = deltaTime.value;

  const velocity = d.vec3f(0, -9.8, 0).mul(dt);
  particleBuffer.value[idx] = position.add(velocity);
});
```

The `$` property is an alias for `.value`, providing a shorter syntax:

```typescript
const updateParticles = tgpu.fn([particleBuffer, deltaTime]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;

  // $ is shorthand for .value
  const position = particleBuffer.$[idx];
  const dt = deltaTime.$;

  particleBuffer.$[idx] = position.add(d.vec3f(0, -9.8, 0).mul(dt));
});
```

### Why .value is Needed

The `.value` property exists because TypeGPU buffer and texture objects are JavaScript proxies that provide rich APIs for CPU-side operations (reading, writing, mapping, etc.). On the GPU, these resources become simple bindings to memory locations or texture samplers.

The `.value` property signals to the TGSL transpiler: "access the GPU representation of this resource, not the CPU object."

**On CPU** (JavaScript):

```typescript
const buffer = root.createBuffer(d.arrayOf(d.f32, 10));
await buffer.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]); // CPU API
const data = await buffer.read(); // CPU API
```

**On GPU** (TGSL):

```typescript
const shader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  buffer.value[0] = buffer.value[1] * 2; // GPU memory access
});
```

### Examples of Correct Usage

#### Accessing Buffer Elements

```typescript
const positions = root.createBuffer(d.arrayOf(d.vec3f, 100)).$usage("storage");

const shader = tgpu.fn([positions]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;

  // Read from buffer
  const currentPos = positions.value[idx];

  // Modify
  const newPos = currentPos.add(d.vec3f(0, 0.1, 0));

  // Write to buffer
  positions.value[idx] = newPos;
});
```

#### Accessing Texture Samples

```typescript
const colorTexture = root.createTexture(...);
const textureSampler = root.createSampler(...);

const fragmentShader = tgpu
  .fragmentFn([input])
  .does((input) => {
    'use gpu';
    // Sample texture using .value
    const color = colorTexture.value.sample(textureSampler.value, input.uv);
    return { color };
  })
  .$uses({ colorTexture, textureSampler });
```

#### Accessing Uniform Values

```typescript
const modelMatrix = root.createUniform(d.mat4x4f);
const viewProjection = root.createUniform(d.mat4x4f);

const vertexShader = tgpu
  .vertexFn([input])
  .does((input) => {
    "use gpu";
    // Access uniforms with .value
    const worldPos = modelMatrix.value.mul(d.vec4f(input.position, 1));
    const clipPos = viewProjection.value.mul(worldPos);

    return { position: clipPos };
  })
  .$uses({ modelMatrix, viewProjection });
```

## Standard Library (typegpu/std)

TypeGPU provides a comprehensive standard library under `typegpu/std` that includes functions for arithmetic, comparison, and other common operations. These functions work consistently on both CPU and GPU, enabling shared business logic and shader unit testing.

### Arithmetic Operations

Since JavaScript doesn't support operator overloading, TypeGPU provides standard library functions for vector and matrix arithmetic:

```typescript
import { add, sub, mul, div, mod } from "typegpu/std";
import * as d from "typegpu/data";

const shader = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((a, b) => {
  "use gpu";

  // Arithmetic operations
  const sum = add(a, b); // a + b
  const difference = sub(a, b); // a - b
  const product = mul(a, d.f32(2)); // a * 2
  const quotient = div(a, d.f32(2)); // a / 2

  return sum;
});
```

**Note**: As of TypeGPU 0.7+, vectors and matrices have operator methods, allowing chained operations (covered in the next section).

### Comparison Operations

Standard library comparison functions work with scalars, vectors, and other types:

```typescript
import { eq, ne, lt, gt, le, ge } from "typegpu/std";

const compareValues = tgpu.fn([d.f32, d.f32], d.bool).does((a, b) => {
  "use gpu";

  const isEqual = eq(a, b); // a === b
  const notEqual = ne(a, b); // a !== b
  const lessThan = lt(a, b); // a < b
  const greaterThan = gt(a, b); // a > b
  const lessOrEqual = le(a, b); // a <= b
  const greaterOrEqual = ge(a, b); // a >= b

  return lessThan;
});
```

### WGSL Standard Library Functions

TypeGPU provides access to WGSL's built-in standard library functions:

```typescript
import {
  distance,
  length,
  normalize,
  dot,
  cross,
  reflect,
  clamp,
  mix,
  step,
  smoothstep,
  arrayLength,
  workgroupBarrier,
} from "typegpu/std";

const advancedOperations = tgpu
  .fn([d.vec3f, d.vec3f], d.vec3f)
  .does((pointA, pointB) => {
    "use gpu";

    // Distance between points
    const dist = distance(pointA, pointB);

    // Normalize vector
    const direction = normalize(sub(pointB, pointA));

    // Dot product
    const dotProduct = dot(pointA, pointB);

    // Cross product
    const perpendicular = cross(pointA, pointB);

    // Clamp values
    const clamped = clamp(pointA, d.f32(0), d.f32(1));

    // Linear interpolation
    const interpolated = mix(pointA, pointB, d.f32(0.5));

    return direction;
  });
```

### Synchronization and Utility Functions

For compute shaders, TypeGPU provides synchronization primitives:

```typescript
import { workgroupBarrier, storageBarrier } from "typegpu/std";

const computeShader = tgpu
  .computeFn([sharedData])
  .does(() => {
    "use gpu";

    // Perform some computation
    const localId = builtin.localInvocationId.x;
    sharedData.value[localId] = computeSomething();

    // Synchronize all threads in workgroup
    workgroupBarrier();

    // Now all threads can safely read shared data
    const sum = sharedData.value[0] + sharedData.value[1];

    return sum;
  })
  .$workgroupSize(64);
```

### CPU and GPU Consistency

The goal of `typegpu/std` is for all functions to have matching behavior on CPU and GPU. This unlocks powerful workflows:

```typescript
import { normalize, dot } from "typegpu/std";
import * as d from "typegpu/data";

const computeReflection = tgpu
  .fn([d.vec3f, d.vec3f], d.vec3f)
  .does((incident, normal) => {
    "use gpu";
    const normalizedNormal = normalize(normal);
    const dotProduct = dot(incident, normalizedNormal);
    const reflection = sub(
      incident,
      mul(normalizedNormal, mul(dotProduct, d.f32(2))),
    );
    return reflection;
  });

// Test on CPU
const incident = d.vec3f(1, -1, 0);
const normal = d.vec3f(0, 1, 0);
const result = computeReflection(incident, normal);
console.log("Reflection:", result); // Runs on CPU

// Use on GPU
const shader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  const idx = builtin.globalInvocationId.x;
  buffer.value[idx] = computeReflection(incident, normal); // Runs on GPU
});
```

## Vector and Matrix Operations

Starting with TypeGPU 0.7, vectors and matrices gained operator methods, enabling more readable and chainable operations compared to using standalone `typegpu/std` functions.

### Chained Operations

Vectors and matrices now have methods that return new instances, allowing natural chaining:

```typescript
import * as d from "typegpu/data";

const physicsUpdate = tgpu
  .fn([d.vec3f, d.vec3f, d.f32], d.vec3f)
  .does((position, velocity, deltaTime) => {
    "use gpu";

    // Chained vector operations
    const gravity = d.vec3f(0, -9.8, 0);
    const acceleration = gravity.mul(deltaTime);
    const newVelocity = velocity.add(acceleration);
    const newPosition = position.add(newVelocity.mul(deltaTime));

    return newPosition;
  });
```

### Vector Operations

Vectors support arithmetic, comparison, and geometric operations:

```typescript
const vectorOps = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((a, b) => {
  "use gpu";

  // Arithmetic (chained methods since 0.7+)
  const sum = a.add(b); // Vector addition
  const difference = a.sub(b); // Vector subtraction
  const scaled = a.mul(2.0); // Scalar multiplication
  const divided = a.div(2.0); // Scalar division

  // Component-wise operations
  const componentMul = a.mul(b); // Component-wise multiply
  const componentDiv = a.div(b); // Component-wise divide

  // Geometric operations
  const len = a.length(); // Vector length
  const normalized = a.normalize(); // Normalize (unit vector)
  const dist = a.distance(b); // Distance between vectors

  return normalized;
});
```

### Dot Product and Cross Product

Essential operations for graphics and physics:

```typescript
const geometricOps = tgpu.fn([d.vec3f, d.vec3f], d.f32).does((a, b) => {
  "use gpu";

  // Dot product: measure of alignment
  const dotProduct = a.dot(b);

  // Cross product: perpendicular vector (3D only)
  const perpendicular = a.cross(b);

  // Angle between vectors (using dot product)
  const cosAngle = a.normalize().dot(b.normalize());
  const angle = Math.acos(cosAngle);

  return dotProduct;
});
```

### Matrix Multiplication

Matrices support multiplication with vectors and other matrices:

```typescript
const transformVertex = tgpu
  .fn([d.mat4x4f, d.vec3f], d.vec4f)
  .does((transform, position) => {
    "use gpu";

    // Matrix-vector multiplication
    const position4 = d.vec4f(position.x, position.y, position.z, 1.0);
    const transformed = transform.mul(position4);

    return transformed;
  });

const combineTransforms = tgpu
  .fn([d.mat4x4f, d.mat4x4f], d.mat4x4f)
  .does((matrixA, matrixB) => {
    "use gpu";

    // Matrix-matrix multiplication
    const combined = matrixA.mul(matrixB);

    return combined;
  });
```

### Complex Chained Examples

The true power of operator methods shines in complex expressions:

```typescript
const advancedPhysics = tgpu
  .fn([particleBuffer, d.f32])
  .does((particles, dt) => {
    "use gpu";
    const idx = builtin.globalInvocationId.x;
    const particle = particles.value[idx];

    // Complex chained operations
    const gravity = d.vec3f(0, -9.8, 0);
    const drag = particle.velocity.mul(-0.1);
    const totalForce = gravity.add(drag);
    const acceleration = totalForce.div(particle.mass);

    // Update with chaining
    particle.velocity = particle.velocity.add(acceleration.mul(dt)).mul(0.99); // Apply damping

    particle.position = particle.position.add(particle.velocity.mul(dt));

    // Boundary check with chaining
    if (particle.position.y < 0) {
      particle.position = d.vec3f(particle.position.x, 0, particle.position.z);
      particle.velocity = particle.velocity.mul(d.vec3f(1, -0.8, 1));
    }

    particles.value[idx] = particle;
  });
```

### Integration with wgpu-matrix

TypeGPU vectors and matrices work seamlessly with the `wgpu-matrix` library because elements can be accessed with the `[]` operator:

```typescript
import { mat4, vec3 } from "wgpu-matrix";
import * as d from "typegpu/data";

// Create TypeGPU matrix
const modelMatrix = d.mat4x4f();

// Use wgpu-matrix utilities
mat4.identity(modelMatrix);
mat4.rotateY(modelMatrix, Math.PI / 4, modelMatrix);
mat4.translate(modelMatrix, [1, 2, 3], modelMatrix);

// Create TypeGPU vector
const direction = d.vec3f(1, 0, 0);

// Use wgpu-matrix for normalization
vec3.normalize(direction, direction);
```

## Mixing WGSL and TGSL

TypeGPU doesn't force you to choose between TGSL and WGSL—you can mix and match both approaches freely, using each where it makes the most sense.

### When to Use Each

**Use TGSL when:**

- You want full IDE support with autocomplete and type checking
- You need to share code between CPU and GPU (testing, simulation)
- The shader logic is relatively straightforward
- You're building reusable utility functions
- You want automatic dependency resolution

**Use WGSL when:**

- Complex matrix operations are more readable in shader syntax
- You need features or functions not yet supported by TGSL
- You're porting existing WGSL shaders
- You prefer the familiar shader language syntax
- You need specific WGSL features or attributes

### Calling WGSL from TGSL

You can define functions in WGSL and call them from TGSL:

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Define a WGSL function
const complexMatrixOp = tgpu.fn([d.mat4x4f, d.mat4x4f], d.mat4x4f)
  .implement(/* wgsl */ `
    fn complexMatrixOp(a: mat4x4f, b: mat4x4f) -> mat4x4f {
      // Complex WGSL-specific code
      let result = a * b;
      result[0][0] = result[0][0] * 2.0;
      return transpose(result);
    }
  `);

// Call from TGSL
const useWgslFunction = tgpu
  .fn([d.mat4x4f, d.mat4x4f], d.mat4x4f)
  .does((matA, matB) => {
    "use gpu";

    // Call the WGSL function from TGSL
    const result = complexMatrixOp(matA, matB);

    return result;
  });
```

### Calling TGSL from WGSL

Similarly, TGSL functions can be called from WGSL implementations:

```typescript
// Define TGSL utility
const clamp = tgpu.fn([d.f32, d.f32, d.f32], d.f32).does((value, min, max) => {
  "use gpu";
  if (value < min) return min;
  if (value > max) return max;
  return value;
});

// Use in WGSL
const processValue = tgpu
  .fn([d.f32], d.f32)
  .implement(
    /* wgsl */ `
    fn processValue(input: f32) -> f32 {
      let processed = input * 2.0 + 1.0;
      return clamp(processed, 0.0, 10.0);
    }
  `,
  )
  .$uses({ clamp });
```

### Hybrid Approach

Real-world shaders often benefit from mixing both approaches:

```typescript
// TGSL utilities (shared with CPU)
const normalize = tgpu.fn([d.vec3f], d.vec3f).does((v) => {
  "use gpu";
  const len = v.length();
  return v.div(len);
});

const dot = tgpu.fn([d.vec3f, d.vec3f], d.f32).does((a, b) => {
  "use gpu";
  return a.x * b.x + a.y * b.y + a.z * b.z;
});

// WGSL for complex lighting
const calculateLighting = tgpu
  .fn([d.vec3f, d.vec3f, d.vec3f], d.vec3f)
  .implement(
    /* wgsl */ `
    fn calculateLighting(
      normal: vec3f,
      lightDir: vec3f,
      viewDir: vec3f
    ) -> vec3f {
      let N = normalize(normal);
      let L = normalize(lightDir);
      let V = normalize(viewDir);
      let H = normalize(L + V);

      let diffuse = max(dot(N, L), 0.0);
      let specular = pow(max(dot(N, H), 0.0), 32.0);

      return vec3f(diffuse) + vec3f(specular * 0.5);
    }
  `,
  )
  .$uses({ normalize, dot });

// TGSL main shader using WGSL function
const fragmentShader = tgpu
  .fragmentFn([input])
  .does((input) => {
    "use gpu";
    const lighting = calculateLighting(
      input.normal,
      lightDirection.value,
      input.viewDir,
    );
    const color = input.albedo.mul(lighting);
    return { color: d.vec4f(color, 1.0) };
  })
  .$uses({ calculateLighting, lightDirection });
```

### Operator Overloading Limitations in TGSL

JavaScript doesn't support operator overloading, which is why TypeGPU provides both operator methods (`.add()`, `.mul()`, etc.) and `typegpu/std` functions. In WGSL, you can use native operators:

```typescript
// TGSL: Must use methods
const tgslVersion = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does((a, b) => {
  "use gpu";
  return a.add(b).mul(2.0); // Method chaining
});

// WGSL: Can use operators
const wgslVersion = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).implement(/* wgsl */ `
    fn wgslVersion(a: vec3f, b: vec3f) -> vec3f {
      return (a + b) * 2.0;  // Native operators
    }
  `);
```

For complex mathematical expressions, WGSL's operator syntax can be more readable. For simple operations and code sharing with CPU, TGSL's methods work excellently.

## GPU Console.log

TypeGPU includes an innovative debugging feature: `console.log` that works inside GPU shaders. This allows you to inspect runtime behavior of your shaders without complex buffer-based debugging setups.

### Debugging with console.log in Shaders

Only functions marked with `'use gpu'` can be called from within a shader. An exception to this rule is `console.log`, which allows for tracking runtime behavior of shaders in a familiar way:

```typescript
const debugShader = tgpu
  .computeFn([buffer])
  .does(() => {
    "use gpu";
    const idx = builtin.globalInvocationId.x;
    const value = buffer.value[idx];

    // Debug output from GPU!
    console.log("Thread", idx, "processing value:", value);

    const result = value * value;
    console.log("Result:", result);

    buffer.value[idx] = result;
  })
  .$workgroupSize(64);
```

### How It Works

Usually when you write shaders, you have to set a pixel to a specific color to inspect what's happening, or write to a buffer (which requires knowing if you're writing correctly or if your logic is wrong). GPU `console.log` solves this by translating `console.log` calls into appropriate GPU instructions that:

1. Capture the logged values from shader execution
2. Transfer them back to CPU memory
3. Display them in your JavaScript console

This provides visibility into shader runtime logic without the complexity of manual debugging buffers.

### Limitations

While powerful, GPU `console.log` has important limitations:

**Volume Constraints**:

- You can log per every thread/pixel, but there's a buffer limit
- For small numbers of invocations, logging works well
- If you exceed the buffer limit, TypeGPU will warn you and stop logging
- It's safe—won't crash your application, just limits output

**Performance Impact**:

- Console logging from GPU has overhead
- Don't use in production or performance-critical shaders
- Remove logging statements after debugging

**Data Type Support**:

- Works with scalars (numbers, booleans)
- Works with vectors (logged component-wise)
- Complex structures may require logging individual fields

### Practical Debugging Example

```typescript
const particleSimulation = tgpu
  .computeFn([particles])
  .does(() => {
    "use gpu";
    const idx = builtin.globalInvocationId.x;

    // Only log for specific threads
    if (idx < 5) {
      console.log("=== Particle", idx, "===");

      const p = particles.value[idx];
      console.log("Position:", p.position);
      console.log("Velocity:", p.velocity);

      // Update particle
      const newPos = p.position.add(p.velocity);
      console.log("New position:", newPos);

      particles.value[idx].position = newPos;
    }
  })
  .$workgroupSize(64);
```

This selective logging (using `if (idx < 5)`) is a best practice—it limits output to manageable amounts while still providing insight into shader behavior.

### Alternative Debugging Approaches

Beyond `console.log`, TypeGPU offers other debugging options:

**CPU Simulation**: TypeGPU's experimental `tgpu.simulate` API simulates GPU behavior on the CPU, enabling:

- Breakpoints in shader code
- Step-through debugging
- Full access to CPU debugging tools

**Unit Testing**: Since TGSL functions run on both CPU and GPU, you can write unit tests:

```typescript
const clampValue = tgpu
  .fn([d.f32, d.f32, d.f32], d.f32)
  .does((val, min, max) => {
    "use gpu";
    if (val < min) return min;
    if (val > max) return max;
    return val;
  });

// Test on CPU
console.assert(clampValue(5, 0, 10) === 5);
console.assert(clampValue(-5, 0, 10) === 0);
console.assert(clampValue(15, 0, 10) === 10);
```

## Constant Folding

Constant folding is a compiler optimization technique where expressions involving constants are evaluated at compile time rather than runtime. While there's limited specific documentation about constant folding in TypeGPU, the WGSL specification and WebGPU drivers perform various compile-time optimizations.

### Compile-Time Optimizations

When TypeGPU generates WGSL code, the WebGPU driver's shader compiler applies optimizations including:

**Constant Expression Evaluation**:

```typescript
const shader = tgpu.fn([d.f32], d.f32).does((value) => {
  "use gpu";

  // This expression is evaluated at compile time
  const coefficient = (2.0 * 3.14159) / 180.0;

  return value * coefficient;
});

// Generated WGSL will likely have pre-computed coefficient
// const coefficient = 0.0349066;  // Pre-computed
```

**Dead Code Elimination**:

```typescript
const shader = tgpu.fn([d.f32], d.f32).does((value) => {
  "use gpu";

  const alwaysFalse = false;

  if (alwaysFalse) {
    // This code will be eliminated at compile time
    return value * 1000;
  }

  return value;
});
```

**Loop Unrolling**: Loops with constant bounds may be unrolled:

```typescript
const shader = tgpu.fn([d.vec3f], d.f32).does((v) => {
  "use gpu";

  let sum = 0;
  for (let i = 0; i < 3; i++) {
    // Constant bound
    sum += v[i];
  }

  return sum;
  // Might be optimized to: return v[0] + v[1] + v[2];
});
```

### Best Practices for Optimization

To help the compiler optimize your shaders:

**Use Constants**: Define constants for values that don't change:

```typescript
const MAX_ITERATIONS = 100;
const PI = 3.14159;
const GRAVITY = d.vec3f(0, -9.8, 0);

const shader = tgpu.fn([]).does(() => {
  "use gpu";
  // Compiler can optimize these constant references
  const angle = PI / 4.0;
  const force = GRAVITY.mul(2.0);
});
```

**Avoid Dynamic Branching**: When possible, use constant conditions:

```typescript
// Good: Constant condition
const USE_ADVANCED_LIGHTING = true;

const shader = tgpu.fn([]).does(() => {
  "use gpu";
  if (USE_ADVANCED_LIGHTING) {
    // Code path chosen at compile time
  }
});

// Less optimal: Dynamic branching
const shader2 = tgpu.fn([flags]).does(() => {
  "use gpu";
  if (flags.value.useAdvancedLighting) {
    // Runtime branch
  }
});
```

**Simplify Expressions**: Help the compiler by simplifying math:

```typescript
// Compiler-friendly
const shader = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";
  return x * 0.5; // Compiler can optimize to x >> 1 or special instruction
});
```

## Best Practices

### Function Composition

Build complex shaders from small, reusable functions:

```typescript
// Utility functions
const saturate = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";
  return clamp(x, 0, 1);
});

const lengthSquared = tgpu.fn([d.vec3f], d.f32).does((v) => {
  "use gpu";
  return v.dot(v);
});

const attenuate = tgpu.fn([d.f32, d.f32], d.f32).does((distance, radius) => {
  "use gpu";
  const ratio = distance / radius;
  const attenuation = 1.0 - saturate(ratio * ratio);
  return attenuation * attenuation;
});

// Compose into larger function
const calculatePointLight = tgpu
  .fn([d.vec3f, d.vec3f, d.vec3f, d.f32], d.vec3f)
  .does((fragPos, lightPos, lightColor, radius) => {
    "use gpu";
    const toLight = lightPos.sub(fragPos);
    const distSquared = lengthSquared(toLight);
    const distance = Math.sqrt(distSquared);
    const attenuation = attenuate(distance, radius);
    return lightColor.mul(attenuation);
  });
```

### Code Reuse Across CPU and GPU

Leverage TGSL's dual-mode execution for shared logic:

```typescript
// Physics function used on both CPU and GPU
const applyGravity = tgpu
  .fn([d.vec3f, d.f32], d.vec3f)
  .does((velocity, deltaTime) => {
    "use gpu";
    const gravity = d.vec3f(0, -9.8, 0);
    return velocity.add(gravity.mul(deltaTime));
  });

// CPU-side simulation for testing
function simulateOnCPU(particles: Particle[], dt: number) {
  for (const particle of particles) {
    particle.velocity = applyGravity(particle.velocity, dt);
    particle.position = particle.position.add(particle.velocity.mul(dt));
  }
}

// GPU shader using same function
const updateParticlesGPU = tgpu
  .computeFn([particleBuffer, deltaTime])
  .does(() => {
    "use gpu";
    const idx = builtin.globalInvocationId.x;
    const p = particleBuffer.value[idx];

    p.velocity = applyGravity(p.velocity, deltaTime.value);
    p.position = p.position.add(p.velocity.mul(deltaTime.value));

    particleBuffer.value[idx] = p;
  })
  .$workgroupSize(64);
```

### Naming Conventions

Use clear, descriptive names and the `$name()` method:

```typescript
const particleBuffer = root
  .createBuffer(particleSchema, particleCount)
  .$usage("storage")
  .$name("particlePositions"); // Shows in GPU profilers

const updateVelocity = tgpu
  .fn([particleBuffer])
  .does(() => {
    "use gpu"; /* ... */
  })
  .$name("updateParticleVelocity"); // Clear function name

const physicsPipeline = root
  .makeComputePipeline(updateVelocity)
  .$workgroupSize(64)
  .$name("physicsSimulation"); // Pipeline name for debugging
```

### Organize Shader Code

Structure your shader code for maintainability:

```typescript
// shaders/utils/math.ts
export const clamp = tgpu.fn([d.f32, d.f32, d.f32], d.f32).does(...);
export const saturate = tgpu.fn([d.f32], d.f32).does(...);

// shaders/utils/geometry.ts
export const normalize = tgpu.fn([d.vec3f], d.vec3f).does(...);
export const reflect = tgpu.fn([d.vec3f, d.vec3f], d.vec3f).does(...);

// shaders/compute/physics.ts
import { clamp, saturate } from '../utils/math';
import { normalize } from '../utils/geometry';

export const updateParticles = tgpu.computeFn([...]).does(...);

// shaders/render/vertex.ts
export const vertexMain = tgpu.vertexFn([...]).does(...);
```

## Common Pitfalls

### Unsupported JS Features

Not all JavaScript features work in TGSL:

```typescript
// WRONG: Using unsupported features
const badShader = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";

  // ERROR: String manipulation not supported
  const message = `Value: ${x}`;

  // ERROR: Object literals (use structs)
  const obj = { value: x, doubled: x * 2 };

  // ERROR: Arrow functions as values
  const multiply = (a, b) => a * b;

  // ERROR: Try-catch
  try {
    return x / 0;
  } catch (e) {
    return 0;
  }
});

// CORRECT: Use TGSL-supported features
const goodShader = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";

  // Use structs instead of objects
  const result = MyStruct({ value: x, doubled: x * 2 });

  // Use conditional logic instead of exceptions
  if (x === 0) return 0;
  return 1 / x;
});
```

### Value vs Reference Semantics

TGSL is developed to work on GPU the same as on CPU as much as possible. However, because of fundamental differences between JavaScript and WGSL, vectors, matrices, and structs are treated as reference types in JavaScript and value types in WGSL:

```typescript
// On CPU (JavaScript): Reference semantics
const a = d.vec3f(1, 2, 3);
const b = a; // b references same object as a
b[0] = 10;
console.log(a[0]); // 10 - a was modified!

// On GPU (WGSL): Value semantics
const shader = tgpu.fn([]).does(() => {
  "use gpu";
  const a = d.vec3f(1, 2, 3);
  const b = a; // b is a COPY of a
  b[0] = 10;
  // a[0] is still 1 - assignment copied the value
});
```

This difference can cause confusion when testing shaders on CPU versus GPU. Be aware that assignment behavior differs between execution contexts.

### Forgetting .value

One of the most common mistakes is forgetting to use `.value` when accessing GPU resources:

```typescript
const buffer = root.createBuffer(d.arrayOf(d.f32, 100)).$usage("storage");

// WRONG: Accessing buffer without .value
const wrongShader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  const value = buffer[0]; // ERROR: buffer is the JS object
  buffer[0] = value * 2; // ERROR
});

// CORRECT: Use .value to access GPU resource
const correctShader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  const value = buffer.value[0]; // Accesses GPU memory
  buffer.value[0] = value * 2; // Writes to GPU memory
});

// ALTERNATIVE: Use $ alias
const alternativeShader = tgpu.fn([buffer]).does(() => {
  "use gpu";
  const value = buffer.$[0];
  buffer.$[0] = value * 2;
});
```

### Type Mismatches

Ensure argument types match function signatures:

```typescript
// Function expects vec3f
const transform = tgpu.fn([d.vec3f], d.vec3f).does((v) => {
  "use gpu";
  return v.mul(2);
});

// WRONG: Passing wrong type
const wrongUsage = tgpu.fn([d.vec2f]).does((v) => {
  "use gpu";
  return transform(v); // ERROR: vec2f passed to vec3f parameter
});

// CORRECT: Match types
const correctUsage = tgpu.fn([d.vec2f]).does((v) => {
  "use gpu";
  const v3 = d.vec3f(v.x, v.y, 0);
  return transform(v3); // Now types match
});
```

### Missing 'use gpu' Directive

Always include `'use gpu'` as the first statement:

```typescript
// WRONG: Missing directive
const wrongShader = tgpu.fn([d.f32], d.f32).does((x) => {
  return x * 2; // Will execute only on CPU, cannot be transpiled to WGSL
});

// CORRECT: Include 'use gpu'
const correctShader = tgpu.fn([d.f32], d.f32).does((x) => {
  "use gpu";
  return x * 2; // Can execute on both CPU and GPU
});
```

### Workgroup Size Optimization

Don't use arbitrary workgroup sizes—choose power-of-2 values:

```typescript
// SUBOPTIMAL: Odd workgroup size
const slowPipeline = root.makeComputePipeline(shader).$workgroupSize(37); // Poor GPU utilization

// GOOD: Power-of-2 workgroup size
const fastPipeline = root.makeComputePipeline(shader).$workgroupSize(64); // Better GPU utilization

// ALSO GOOD: 2D workgroup for 2D data
const imagePipeline = root
  .makeComputePipeline(imageShader)
  .$workgroupSize(8, 8); // 64 threads total, good for image processing
```

---

## Conclusion

TGSL Functions represent a paradigm shift in GPU programming, bringing the power of TypeScript to shader development. By enabling code sharing between CPU and GPU, providing comprehensive type safety, and offering modern developer tooling, TGSL eliminates many traditional pain points of GPU programming while maintaining the performance characteristics of hand-written WGSL.

Whether you're building particle systems, implementing ray tracers, creating physics simulations, or developing complex rendering pipelines, TGSL provides the tools to write maintainable, testable, and performant GPU code. The ability to mix TGSL and WGSL freely ensures you're never locked into one approach, while features like GPU `console.log` and dual-mode execution make debugging and testing dramatically easier.

As you build with TypeGPU, remember that TGSL is designed to grow with your needs—start simple with basic compute shaders, expand to complex rendering pipelines, and always have the option to drop down to vanilla WGSL when specialized optimizations are needed. The future of GPU programming is typed, composable, and written in the language you already know: TypeScript.

---

## Sources

- [TGSL | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/tgsl/)
- [Functions | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/functions/)
- [GitHub - software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- [tgpu | TypeGPU](https://docs.swmansion.com/TypeGPU/api/typegpu/variables/tgpu/)
- [Data Schemas | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/)
- [Working with wgpu-matrix | TypeGPU](https://docs.swmansion.com/TypeGPU/integration/working-with-wgpu-matrix/)
- [TypeGPU – Type-safe WebGPU toolkit](https://docs.swmansion.com/TypeGPU/)
- [TypeScript on the GPU – Bridging the Gap Between Realms by Iwo Plaza](https://gitnation.com/contents/typescript-on-the-gpu-bridging-the-gap-between-realms)
- [typegpu - npm](https://www.npmjs.com/package/typegpu)
