---
title: Data Schemas in TypeGPU
sidebar:
  order: 10
---

## Overview

TypeGPU data schemas provide type-safe GPU data definitions with automatic serialization. Similar to how [Zod](https://zod.dev/) provides runtime type validation, TypeGPU schemas serve as both type definitions and runtime validators for GPU data structures.

:::note[Why Schemas?]
In traditional WebGPU, data transfer involves manual byte manipulation, alignment calculations, and padding management. TypeGPU eliminates this by leveraging [typed-binary](https://github.com/iwoplaza/typed-binary)—you never think about bytes when writing GPU programs.
:::

Schemas serve a dual purpose:
1. **Define TypeScript types** for compile-time checking
2. **Validate and serialize data** when moving between JavaScript and GPU memory

## Scalar Types

Scalar schemas represent single numeric or boolean values, mapping directly to WGSL primitives:

| TypeGPU | WGSL | Description |
|---------|------|-------------|
| `d.f32` | `f32` | 32-bit float |
| `d.f16` | `f16` | 16-bit float |
| `d.u32` | `u32` | Unsigned 32-bit integer |
| `d.i32` | `i32` | Signed 32-bit integer |
| `d.bool` | `bool` | Boolean |

### Using Scalars as Constructors

```typescript title="Creating scalar values"
import * as d from "typegpu/data";

const radius = d.f32(3.14159);
const particleCount = d.u32(1000);
const temperature = d.i32(-40);
const isActive = d.bool(true);
```

### Scalars in Structs

```typescript title="Scalar fields in structs"
const Configuration = d.struct({
  timeStep: d.f32,
  gravity: d.f32,
  damping: d.f32,
  maxParticles: d.u32,
  simulationEnabled: d.bool,
});
```

## Vector Types

Vectors represent 2, 3, or 4 elements of the same scalar type—positions, velocities, colors.

<details>
<summary>**Float vectors (32-bit)**</summary>

| TypeGPU | WGSL | Components |
|---------|------|------------|
| `d.vec2f` | `vec2<f32>` | 2 |
| `d.vec3f` | `vec3<f32>` | 3 |
| `d.vec4f` | `vec4<f32>` | 4 |

</details>

<details>
<summary>**Integer vectors**</summary>

| TypeGPU | WGSL | Type |
|---------|------|------|
| `d.vec2i`, `d.vec3i`, `d.vec4i` | `vec2<i32>`, etc. | Signed |
| `d.vec2u`, `d.vec3u`, `d.vec4u` | `vec2<u32>`, etc. | Unsigned |

</details>

<details>
<summary>**Half-precision and boolean vectors**</summary>

| TypeGPU | WGSL |
|---------|------|
| `d.vec2h`, `d.vec3h`, `d.vec4h` | `vec2<f16>`, etc. |
| `d.vec2b`, `d.vec3b`, `d.vec4b` | `vec2<bool>`, etc. |

</details>

### Constructor Patterns

```typescript title="Vector construction"
// Zero initialization
const origin = d.vec3f();           // (0.0, 0.0, 0.0)

// Uniform initialization
const ones = d.vec3f(1);            // (1.0, 1.0, 1.0)

// Component-wise
const position = d.vec3f(1, 2, 3.5);
const color = d.vec4f(1, 0, 0, 1);  // RGBA red
```

### Element Access

```typescript title="Accessing vector components"
const velocity = d.vec3f(1.5, 2.0, -0.5);

const x = velocity[0];  // 1.5
const y = velocity[1];  // 2.0
const z = velocity[2];  // -0.5

velocity[0] = 3.0;      // Modify x
```

### Practical Examples

```typescript title="Common vector use cases"
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

const ColorRGBA = d.struct({
  color: d.vec4f,
});

const GridCell = d.struct({
  indices: d.vec3u,
  value: d.f32,
});
```

## Matrix Types

Matrices handle transformations. TypeGPU supports square matrices of `f32`:

| TypeGPU | WGSL | Size |
|---------|------|------|
| `d.mat2x2f` | `mat2x2<f32>` | 2×2 |
| `d.mat3x3f` | `mat3x3<f32>` | 3×3 |
| `d.mat4x4f` | `mat4x4<f32>` | 4×4 |

:::caution[Column-Major Order]
Like WGSL and most graphics APIs, TypeGPU matrices use column-major order:

```typescript
// Matrix stored as [column0, column1, column2]
// [m00, m10, m20, m01, m11, m21, m02, m12, m22]
```
:::

### Integration with wgpu-matrix

TypeGPU works seamlessly with [wgpu-matrix](https://github.com/greggman/wgpu-matrix) via the `[]` operator:

```typescript title="Using wgpu-matrix with TypeGPU"
import { mat4, vec3 } from "wgpu-matrix";
import * as d from "typegpu/data";

const transform = d.mat4x4f();
mat4.identity(transform);

const modelMatrix = d.mat4x4f();
mat4.translation([1, 2, 3], modelMatrix);
mat4.rotateY(modelMatrix, Math.PI / 4, modelMatrix);

const direction = d.vec3f(1, 2, 3);
vec3.normalize(direction, direction);
```

:::tip[wgpu-matrix Destination Parameter]
Always pass the destination parameter when using wgpu-matrix:

```typescript
// ✗ Wrong: creates new Float32Array, doesn't modify v
const normalized = vec3.normalize(v);

// ✓ Correct: modifies v in-place
vec3.normalize(v, v);
```
:::

## Struct Schemas

Structs compose complex types from primitives and other structs, automatically handling alignment and padding.

### Basic Definition

```typescript title="Struct definitions"
import * as d from "typegpu/data";

const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

const SimulationConfig = d.struct({
  deltaTime: d.f32,
  gravity: d.f32,
  damping: d.f32,
  particleCount: d.u32,
});
```

### Nested Structs

```typescript title="Hierarchical data structures"
const Material = d.struct({
  albedo: d.vec3f,
  metallic: d.f32,
  roughness: d.f32,
  ao: d.f32,
});

const MeshInstance = d.struct({
  transform: d.mat4x4f,
  material: Material,
  isVisible: d.bool,
});

const SceneData = d.struct({
  camera: d.struct({
    viewMatrix: d.mat4x4f,
    projectionMatrix: d.mat4x4f,
    position: d.vec3f,
  }),
  ambientLight: d.vec3f,
  time: d.f32,
});
```

### Automatic Alignment

:::danger[vec3 Padding]
`vec3<f32>` requires 16-byte alignment despite being 12 bytes. TypeGPU handles this automatically:

```typescript
const ExampleStruct = d.struct({
  a: d.f32,     // 4 bytes at offset 0
  // 12 bytes padding inserted automatically
  b: d.vec3f,   // 12 bytes at offset 16 (aligned to 16)
  c: d.f32,     // 4 bytes at offset 28
  // Padding to align struct size to 16
});
// Total: 48 bytes (not 20)
```
:::

### Custom Alignment

Override defaults when needed:

```typescript title="Manual alignment control"
const CustomAlignedStruct = d.struct({
  a: d.align(16, d.f32),     // Force 16-byte alignment
  b: d.size(32, d.vec4f),    // Explicit size
});
```

## Array Schemas

Arrays are created with `d.arrayOf()`:

### Fixed-Size Arrays

```typescript title="Fixed-size arrays"
const FloatArray = d.arrayOf(d.f32, 10);

const ParticleArray = d.arrayOf(
  d.struct({
    position: d.vec3f,
    velocity: d.vec3f,
  }),
  100
);

const Grid = d.struct({
  cells: d.arrayOf(d.f32, 64),
  width: d.u32,
  height: d.u32,
});
```

### Runtime-Sized Arrays

Runtime-sized arrays omit the length parameter:

```typescript title="Runtime-sized arrays"
const DynamicParticles = d.arrayOf(
  d.struct({
    position: d.vec3f,
    velocity: d.vec3f,
  })
  // No size - determined at runtime
);
```

:::caution[Position Requirement]
Runtime-sized arrays must be the last field in a struct:

```typescript
// ✗ Error: runtime-sized array not at end
const Wrong = d.struct({
  particles: d.arrayOf(d.vec3f),
  particleCount: d.u32,
});

// ✓ Correct: runtime-sized array at end
const Correct = d.struct({
  particleCount: d.u32,
  particles: d.arrayOf(d.vec3f),
});
```
:::

### Nested Arrays

```typescript title="Multi-dimensional data"
const HeightMap = (width: number, height: number) =>
  d.arrayOf(d.arrayOf(d.f32, height), width);

const BoneTransforms = d.struct({
  bones: d.arrayOf(d.mat4x4f, 64),
});
```

## Type Inference

Extract TypeScript types from schemas:

```typescript title="Type inference"
import * as d from "typegpu/data";

const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Infer TypeScript type
type ParticleType = d.Infer<typeof Particle>;

// Use in application code
const particles: ParticleType[] = [
  { position: [0, 0, 0], velocity: [1, 0, 0], lifetime: 1.0 },
  { position: [1, 1, 1], velocity: [0, 1, 0], lifetime: 0.5 },
];
```

### Schema Factory Functions

```typescript title="Dynamic schema generation"
const HeightMap = (width: number, height: number) =>
  d.arrayOf(d.arrayOf(d.f32, height), width);

type HeightMapType = ReturnType<typeof HeightMap>;
```

## Memory Layout

### Alignment Rules

| Type | Size | Alignment |
|------|------|-----------|
| `f32`, `i32`, `u32` | 4 bytes | 4 bytes |
| `vec2<f32>` | 8 bytes | 8 bytes |
| `vec3<f32>` | 12 bytes | **16 bytes** |
| `vec4<f32>` | 16 bytes | 16 bytes |
| `mat4x4<f32>` | 64 bytes | 16 bytes |

TypeGPU automatically inserts padding to satisfy these requirements.

### Loose Schemas for Vertex Data

When strict alignment isn't required (vertex buffers), use loose schemas to save memory:

```typescript title="Loose vs aligned schemas"
// Standard struct with alignment (32 bytes)
const ParticleGeometry = d.struct({
  position: d.vec3f,  // 16 bytes (aligned)
  color: d.vec4f,     // 16 bytes
});

// Loose struct without padding (28 bytes)
const LooseParticleGeometry = d.unstruct({
  position: d.vec3f,  // 12 bytes (no padding)
  color: d.vec4f,     // 16 bytes
});

// Use d.disarrayOf for loose arrays
const LooseVertexBuffer = d.disarrayOf(LooseParticleGeometry);
```

## Schema Design Guidelines

:::tip[Single Source of Truth]
Define schemas once and infer types from them:

```typescript
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
});
type Particle = d.Infer<typeof Particle>;
```
:::

:::tip[Match WGSL Definitions]
Ensure TypeGPU schemas match your WGSL struct definitions exactly:

```typescript
// TypeGPU
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// WGSL
// struct Particle {
//   position: vec3f,
//   velocity: vec3f,
//   lifetime: f32,
// }
```
:::

:::tip[Leverage Nested Structs]
Organize schemas hierarchically for reusability:

```typescript
const Material = d.struct({
  albedo: d.vec3f,
  roughness: d.f32,
});

const Mesh = d.struct({
  transform: d.mat4x4f,
  material: Material,  // Reuse Material schema
});
```
:::

## Resources

:::note[Official Documentation]
- [Data Schemas Guide](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/)
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/)
- [Buffers Guide](https://docs.swmansion.com/TypeGPU/fundamentals/buffers/)
- [WebGPU Memory Layout](https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html)
:::
