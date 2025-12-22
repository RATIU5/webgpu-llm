# Data Schemas in TypeGPU

## Overview

Data schemas in TypeGPU provide type-safe GPU data definitions with automatic serialization, allowing developers to work with GPU buffers without manually handling bytes. Similar to how [Zod](https://zod.dev/) provides runtime type validation for JavaScript objects, TypeGPU's data schemas serve as both type definitions and runtime validators for GPU data structures.

In traditional WebGPU development, data communication between JavaScript and WGSL (WebGPU Shading Language) involves reading and writing raw bytes to buffers. Even though data is strongly typed in WGSL shaders, the JavaScript side requires manual byte manipulation, alignment calculations, and padding management. Any misalignments or misinterpretations can lead to faulty code that is extremely difficult to debug.

TypeGPU's data schemas eliminate this complexity by leveraging [typed-binary](https://github.com/iwoplaza/typed-binary), enabling developers to "never think about bytes when writing GPU programs ever again." The library provides data schemas for scalar, vector, and matrix types, as well as constructors for struct and array schemas, all with automatic binary serialization and TypeScript type inference.

## Key Concepts

### Schemas as Both Type Definitions and Runtime Validators

TypeGPU data schemas serve a dual purpose: they define the structure of GPU data at compile time while also providing runtime validation and serialization. When you define a schema like `d.vec3f` or create a struct with `d.struct()`, you're creating an object that:

1. **Defines TypeScript types** that can be inferred and used in your application code
2. **Validates data** when moving between JavaScript and GPU memory
3. **Handles serialization** automatically, converting JavaScript values to properly aligned binary data
4. **Provides constructors** for creating instances of these types in both JavaScript and TGSL (TypeGPU Shading Language)

### Automatic Binary Serialization

One of TypeGPU's most powerful features is its automatic handling of binary serialization. Complex data types like structs and arrays require particular byte alignment and padding between fields according to WebGPU's memory layout rules. TypeGPU handles this automatically during buffer writes and reads.

For example, a `vec3<f32>` in WGSL has an alignment requirement of 16 bytes, meaning it takes up 16 bytes of memory even though it only contains 12 bytes of data (three 32-bit floats). TypeGPU understands these alignment rules and applies proper padding automatically, so developers no longer need to know that "vec3fs have to be aligned to multiples of 16 bytes."

### TypeScript Type Inference

TypeGPU schemas support full TypeScript type inference, allowing you to maintain a single source of truth for data structures. You can define schemas programmatically and then infer both runtime and compile-time type information from them.

The primary usage for data schemas is to define buffer and function signatures, ensuring type safety across the JavaScript-GPU boundary. This type safety extends throughout the entire GPU computation pipeline, enabling you to pass typed buffer values between different libraries without copying data back to CPU memory.

## Scalar Schemas

Scalar schemas represent single numeric or boolean values. TypeGPU provides schemas that directly correspond to WGSL primitive types:

### Available Scalar Types

- **`d.f32`**: 32-bit floating-point number (maps to WGSL `f32`)
- **`d.f16`**: 16-bit floating-point number (maps to WGSL `f16`)
- **`d.u32`**: Unsigned 32-bit integer (maps to WGSL `u32`)
- **`d.i32`**: Signed 32-bit integer (maps to WGSL `i32`)
- **`d.bool`**: Boolean value (maps to WGSL `bool`)

### Using Scalars as Constructors

Scalar schemas can be called as functions to create or cast values to their corresponding types:

```typescript
import * as d from 'typegpu/data';

// Creating scalar values
const radius = d.f32(3.14159);
const particleCount = d.u32(1000);
const temperature = d.i32(-40);
const isActive = d.bool(true);

// Type casting
const pi = d.f32(Math.PI);
const maxIterations = d.u32(100);
```

When used in TGSL shader code, these constructors ensure type safety:

```typescript
import { tgsl } from 'typegpu/tgsl';

const shader = tgsl.fn([], d.f32)
  .does(() => {
    const halfPi = d.f32(1.5708);
    return halfPi;
  });
```

### Scalar Schemas in Buffer Definitions

Scalars are commonly used as fields in struct definitions:

```typescript
const Configuration = d.struct({
  timeStep: d.f32,
  gravity: d.f32,
  damping: d.f32,
  maxParticles: d.u32,
  simulationEnabled: d.bool,
});
```

## Vector Schemas

Vector schemas represent collections of 2, 3, or 4 elements of the same scalar type. They map directly to WGSL vector types and are essential for graphics programming, representing positions, velocities, colors, and more.

### Available Vector Types

**Float vectors (32-bit):**
- `d.vec2f` - 2-component vector of `f32` (maps to WGSL `vec2<f32>` or `vec2f`)
- `d.vec3f` - 3-component vector of `f32` (maps to WGSL `vec3<f32>` or `vec3f`)
- `d.vec4f` - 4-component vector of `f32` (maps to WGSL `vec4<f32>` or `vec4f`)

**Integer vectors (signed 32-bit):**
- `d.vec2i` - 2-component vector of `i32` (maps to WGSL `vec2<i32>` or `vec2i`)
- `d.vec3i` - 3-component vector of `i32` (maps to WGSL `vec3<i32>` or `vec3i`)
- `d.vec4i` - 4-component vector of `i32` (maps to WGSL `vec4<i32>` or `vec4i`)

**Unsigned integer vectors:**
- `d.vec2u` - 2-component vector of `u32` (maps to WGSL `vec2<u32>` or `vec2u`)
- `d.vec3u` - 3-component vector of `u32` (maps to WGSL `vec3<u32>` or `vec3u`)
- `d.vec4u` - 4-component vector of `u32` (maps to WGSL `vec4<u32>` or `vec4u`)

**Half-precision float vectors (16-bit):**
- `d.vec2h` - 2-component vector of `f16` (maps to WGSL `vec2<f16>` or `vec2h`)
- `d.vec3h` - 3-component vector of `f16` (maps to WGSL `vec3<f16>` or `vec3h`)
- `d.vec4h` - 4-component vector of `f16` (maps to WGSL `vec4<f16>` or `vec4h`)

**Boolean vectors:**
- `d.vec2b` - 2-component vector of `bool` (maps to WGSL `vec2<bool>`)
- `d.vec3b` - 3-component vector of `bool` (maps to WGSL `vec3<bool>`)
- `d.vec4b` - 4-component vector of `bool` (maps to WGSL `vec4<bool>`)

### Constructor Patterns

Just like scalar schemas, vector schemas can be called as functions to create vector values. They support multiple constructor patterns:

```typescript
// Zero initialization - all components set to 0
const origin = d.vec3f();           // (0.0, 0.0, 0.0)
const zeroVec = d.vec2i();          // (0, 0)

// Uniform initialization - all components set to the same value
const ones = d.vec3f(1);            // (1.0, 1.0, 1.0)
const allFive = d.vec4u(5);         // (5, 5, 5, 5)

// Component-wise initialization
const position = d.vec3f(1, 2, 3.5);      // (1.0, 2.0, 3.5)
const color = d.vec4f(1, 0, 0, 1);        // (1.0, 0.0, 0.0, 1.0) - red
const gridCoord = d.vec2i(10, 20);        // (10, 20)
```

### Vector Operations

Vectors created with TypeGPU support element access using the `[]` operator, making them compatible with existing WebGPU utilities:

```typescript
const velocity = d.vec3f(1.5, 2.0, -0.5);

// Accessing components
const x = velocity[0];  // 1.5
const y = velocity[1];  // 2.0
const z = velocity[2];  // -0.5

// Modifying components
velocity[0] = 3.0;
```

In TGSL shader code, you can use standard WGSL vector operations:

```typescript
import { tgsl } from 'typegpu/tgsl';

const computeVelocity = tgsl.fn([d.vec3f, d.f32], d.vec3f)
  .does((acceleration, deltaTime) => {
    const initialVelocity = d.vec3f(0, 0, 0);
    return initialVelocity + acceleration * deltaTime;
  });
```

### Practical Vector Examples

Vectors are fundamental in graphics programming. Here are common use cases:

```typescript
// Defining a particle with position and velocity
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// RGB color representation
const Color = d.struct({
  rgb: d.vec3f,  // Red, green, blue components
  alpha: d.f32,
});

// Alternative: RGBA as a single vector
const ColorRGBA = d.struct({
  color: d.vec4f,  // Red, green, blue, alpha
});

// 2D texture coordinates
const TexCoord = d.struct({
  uv: d.vec2f,
});

// Grid coordinates for compute shaders
const GridCell = d.struct({
  indices: d.vec3u,  // x, y, z indices
  value: d.f32,
});
```

## Matrix Schemas

Matrix schemas represent 2D arrays of floating-point values, essential for transformations in graphics programming. TypeGPU provides schemas for square matrices of `f32` values.

### Available Matrix Types

- **`d.mat2x2f`**: 2×2 matrix of `f32` (maps to WGSL `mat2x2<f32>` or `mat2x2f`)
- **`d.mat3x3f`**: 3×3 matrix of `f32` (maps to WGSL `mat3x3<f32>` or `mat3x3f`)
- **`d.mat4x4f`**: 4×4 matrix of `f32` (maps to WGSL `mat4x4<f32>` or `mat4x4f`)

**Note:** Matrices of other sizes (e.g., `mat4x3f`, `mat3x2f`) and matrices of `f16` are currently not supported by TypeGPU.

### Column-Major Order

Like WGSL and most graphics APIs, TypeGPU matrices are stored in column-major order. This means the matrix data is organized as an array of column vectors:

```typescript
// A 3x3 matrix in column-major order:
// [ m00  m01  m02 ]
// [ m10  m11  m12 ]
// [ m20  m21  m22 ]
//
// Is stored as: [m00, m10, m20, m01, m11, m21, m02, m12, m22]
// Which is: [column0, column1, column2]
```

### Matrix Constructors

Matrices can be created and initialized in various ways:

```typescript
// Identity matrix initialization (typically done with helper libraries)
const identityMatrix = d.mat4x4f();

// Creating transformation matrices
const Transform = d.struct({
  modelMatrix: d.mat4x4f,
  viewMatrix: d.mat4x4f,
  projectionMatrix: d.mat4x4f,
});
```

### Integration with wgpu-matrix

The [wgpu-matrix](https://github.com/greggman/wgpu-matrix) library provides utilities for matrix and vector math, designed from the ground up to be compatible with WebGPU. TypeGPU works seamlessly with wgpu-matrix because elements in TypeGPU vectors and matrices can be accessed with the `[]` operator.

```typescript
import { mat4, vec3 } from 'wgpu-matrix';
import * as d from 'typegpu/data';

// Create a TypeGPU matrix
const transform = d.mat4x4f();

// Initialize as identity using wgpu-matrix
mat4.identity(transform);

// Create transformation matrices
const modelMatrix = d.mat4x4f();
mat4.translation([1, 2, 3], modelMatrix);
mat4.rotateY(modelMatrix, Math.PI / 4, modelMatrix);

// Create a vector and normalize it in-place
const direction = d.vec3f(1, 2, 3);
vec3.normalize(direction, direction);  // Pass same vector as input and dst

// Matrix multiplication
const viewProjection = d.mat4x4f();
const view = d.mat4x4f();
const projection = d.mat4x4f();

mat4.multiply(projection, view, viewProjection);
```

Since you can use TypeGPU primitives directly with wgpu-matrix functions, migration from existing WebGPU code is relatively simple. The matrices and vectors maintain their type information while being compatible with the computational utilities.

### Practical Matrix Examples

```typescript
// Camera transformation
const Camera = d.struct({
  viewMatrix: d.mat4x4f,
  projectionMatrix: d.mat4x4f,
  position: d.vec3f,
  target: d.vec3f,
});

// Model instance with transform
const ModelInstance = d.struct({
  transform: d.mat4x4f,
  normalMatrix: d.mat3x3f,  // For transforming normals
  color: d.vec4f,
});

// 2D transformation (useful for UI or 2D games)
const Transform2D = d.struct({
  matrix: d.mat3x3f,
  position: d.vec2f,
  rotation: d.f32,
  scale: d.vec2f,
});
```

## Struct Schemas

Struct schemas allow you to compose complex data types from primitive and other struct types. They are one of TypeGPU's most powerful features, automatically handling the complex alignment and padding requirements that WebGPU's memory layout rules impose.

### Basic Struct Definition

Structs are created using `d.struct()` with an object defining the fields:

```typescript
import * as d from 'typegpu/data';

// Simple struct with scalar fields
const CircleStruct = d.struct({
  radius: d.f32,
  pos: d.vec3f,
});

// Particle with position, velocity, and lifetime
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Configuration parameters
const SimulationConfig = d.struct({
  deltaTime: d.f32,
  gravity: d.f32,
  damping: d.f32,
  particleCount: d.u32,
});
```

### Nested Structs

Structs can contain other structs, allowing you to build hierarchical data structures:

```typescript
// Material properties
const Material = d.struct({
  albedo: d.vec3f,
  metallic: d.f32,
  roughness: d.f32,
  ao: d.f32,  // Ambient occlusion
});

// Mesh instance combining transform and material
const MeshInstance = d.struct({
  transform: d.mat4x4f,
  material: Material,  // Nested struct
  isVisible: d.bool,
});

// Light source
const Light = d.struct({
  position: d.vec3f,
  color: d.vec3f,
  intensity: d.f32,
  radius: d.f32,
});

// Complete scene data
const SceneData = d.struct({
  camera: d.struct({
    viewMatrix: d.mat4x4f,
    projectionMatrix: d.mat4x4f,
    position: d.vec3f,
  }),
  ambientLight: d.vec3f,
  directionalLight: Light,
  time: d.f32,
});
```

### Automatic Alignment and Padding

WebGPU has strict alignment requirements for struct fields. For example:

- `f32` must be aligned to 4 bytes
- `vec2<f32>` must be aligned to 8 bytes
- `vec3<f32>` and `vec4<f32>` must be aligned to 16 bytes
- Structs must be aligned to the largest alignment of their fields

TypeGPU's `d.struct()` automatically adjusts padding and alignment so that structs comply with WebGPU's memory alignment rules. This eliminates a major source of bugs in GPU programming.

```typescript
// This struct will have automatic padding
const ExampleStruct = d.struct({
  a: d.f32,       // 4 bytes at offset 0
  // 12 bytes of padding inserted here
  b: d.vec3f,     // 12 bytes at offset 16 (aligned to 16)
  c: d.f32,       // 4 bytes at offset 28
  // 12 bytes of padding here to align struct size to 16
});

// Total size: 48 bytes (not 20 bytes as you might expect)
```

### Custom Alignment

While automatic alignment is usually what you want, TypeGPU allows you to override default byte alignment and size for particular fields using `d.align()` and `d.size()`:

```typescript
const CustomAlignedStruct = d.struct({
  // Ensure this field is aligned to 16 bytes
  a: d.align(16, d.f32),

  // Explicitly set the size of this field
  b: d.size(32, d.vec4f),
});
```

### Struct Usage Examples

```typescript
// Ray tracing data structure
const Ray = d.struct({
  origin: d.vec3f,
  direction: d.vec3f,
  tMin: d.f32,
  tMax: d.f32,
});

const HitRecord = d.struct({
  position: d.vec3f,
  normal: d.vec3f,
  t: d.f32,
  materialIndex: d.u32,
});

// Physics simulation
const RigidBody = d.struct({
  position: d.vec3f,
  rotation: d.vec4f,  // Quaternion
  linearVelocity: d.vec3f,
  angularVelocity: d.vec3f,
  mass: d.f32,
  inverseMass: d.f32,
});

// Vertex data for rendering
const Vertex = d.struct({
  position: d.vec3f,
  normal: d.vec3f,
  uv: d.vec2f,
  color: d.vec4f,
});
```

## Array Schemas

Array schemas represent fixed-size or runtime-sized arrays of elements. TypeGPU provides the `d.arrayOf()` function to create array schemas.

### Fixed-Size Arrays

Fixed-size arrays have a known length at schema definition time:

```typescript
import * as d from 'typegpu/data';

// Array of 10 floats
const FloatArray = d.arrayOf(d.f32, 10);

// Array of 100 particles
const ParticleArray = d.arrayOf(
  d.struct({
    position: d.vec3f,
    velocity: d.vec3f,
  }),
  100
);

// Fixed-size array in a struct
const Grid = d.struct({
  cells: d.arrayOf(d.f32, 64),  // 8x8 grid
  width: d.u32,
  height: d.u32,
});
```

### Runtime-Sized Arrays

Runtime-sized arrays don't specify a length, allowing them to be sized dynamically. These are typically used in storage buffers:

```typescript
// Runtime-sized array (for storage buffers)
const DynamicParticleArray = d.arrayOf(
  d.struct({
    position: d.vec3f,
    velocity: d.vec3f,
    lifetime: d.f32,
  })
  // No size specified - runtime-sized
);

// Struct with runtime-sized array (must be last field)
const ParticleSystem = d.struct({
  particleCount: d.u32,
  deltaTime: d.f32,
  // Runtime-sized array must be the last field
  particles: d.arrayOf(d.struct({
    position: d.vec3f,
    velocity: d.vec3f,
  })),
});
```

**Important:** In structs, runtime-sized arrays must be the last field, as per WGSL requirements.

### Nested Arrays

Arrays can contain other arrays, enabling multi-dimensional data structures:

```typescript
// 2D height map
const HeightMap = (width: number, height: number) =>
  d.arrayOf(d.arrayOf(d.f32, height), width);

// 3D voxel grid
const VoxelGrid = (x: number, y: number, z: number) =>
  d.arrayOf(
    d.arrayOf(
      d.arrayOf(d.u32, z),
      y
    ),
    x
  );

// Array of matrices for skeletal animation
const BoneTransforms = d.struct({
  bones: d.arrayOf(d.mat4x4f, 64),  // Up to 64 bones
});
```

### Practical Array Examples

```typescript
// Compute shader workgroup shared memory
const SharedMemoryData = d.struct({
  // Tile of data shared within workgroup
  tile: d.arrayOf(d.vec4f, 256),
});

// Light array for deferred rendering
const LightBuffer = d.struct({
  lightCount: d.u32,
  lights: d.arrayOf(
    d.struct({
      position: d.vec3f,
      color: d.vec3f,
      radius: d.f32,
    }),
    1024  // Max 1024 lights
  ),
});

// Vertex buffer data
const VertexBuffer = d.arrayOf(
  d.struct({
    position: d.vec3f,
    normal: d.vec3f,
    uv: d.vec2f,
  })
  // Runtime-sized based on mesh
);
```

## Type Inference

TypeGPU provides powerful type inference capabilities, allowing you to extract TypeScript types from schemas and use them throughout your application.

### Basic Type Inference with d.Infer

While the exact API details may vary, TypeGPU enables type extraction from schemas:

```typescript
import * as d from 'typegpu/data';

// Define a schema
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Infer the TypeScript type (exact syntax may vary)
type ParticleType = d.Infer<typeof Particle>;

// Now you can use this type in your code
const particles: ParticleType[] = [
  { position: [0, 0, 0], velocity: [1, 0, 0], lifetime: 1.0 },
  { position: [1, 1, 1], velocity: [0, 1, 0], lifetime: 0.5 },
];
```

### Using ReturnType for Schema Functions

You can define schemas as functions and infer types from them, maintaining a single source of truth:

```typescript
// Schema factory function
const HeightMap = (width: number, height: number) =>
  d.arrayOf(d.arrayOf(d.f32, height), width);

// Infer the type
type HeightMap = ReturnType<typeof HeightMap>;

// Use in your application
function generateTerrain(width: number, height: number): HeightMap {
  // Implementation
}
```

### Type Inference in Practice

```typescript
// Define reusable schemas
const Vec3 = d.vec3f;
const Mat4 = d.mat4x4f;

const Transform = d.struct({
  position: Vec3,
  rotation: Vec3,
  scale: Vec3,
  matrix: Mat4,
});

// Extract the type
type TransformData = d.Infer<typeof Transform>;

// Use in your application logic
class GameObject {
  transform: TransformData;

  constructor() {
    this.transform = {
      position: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      matrix: new Float32Array(16),
    };
  }

  updateTransform(newPosition: [number, number, number]) {
    this.transform.position = newPosition;
  }
}
```

### Partial Inference

Recent TypeGPU releases have expanded inference capabilities with features like `InferPartial` and `InferGPU`, which allow more flexible type extraction for different use cases. These enable you to work with partial data structures or GPU-specific type representations.

## Memory Layout

Understanding memory layout is crucial for GPU programming, but TypeGPU handles most of the complexity automatically.

### Automatic Byte Alignment

WebGPU follows strict alignment rules based on the WGSL specification. Every type has alignment requirements - data must be placed at memory addresses that are multiples of certain byte counts:

| Type | Size | Alignment |
|------|------|-----------|
| `f32`, `i32`, `u32` | 4 bytes | 4 bytes |
| `vec2<f32>` | 8 bytes | 8 bytes |
| `vec3<f32>` | 12 bytes | 16 bytes |
| `vec4<f32>` | 16 bytes | 16 bytes |
| `mat4x4<f32>` | 64 bytes | 16 bytes |

TypeGPU automatically ensures proper alignment by inserting padding where necessary.

### Padding Rules

The most common source of confusion is the `vec3` padding rule. A `vec3<f32>`, despite only containing 12 bytes of data, is aligned to 16 bytes:

```typescript
// Without TypeGPU, manual padding calculation:
const manualStruct = {
  a: new Float32Array(1),        // 4 bytes at offset 0
  // 12 bytes of padding
  b: new Float32Array(3),        // 12 bytes at offset 16
  // 4 bytes to next aligned position
};

// With TypeGPU, automatic handling:
const AutoStruct = d.struct({
  a: d.f32,       // 4 bytes
  b: d.vec3f,     // 12 bytes (but aligned to 16)
});
// Padding inserted automatically!
```

### Size Calculations

TypeGPU's schemas automatically measure and hold information about their memory layout parameters. You can query the size of a schema:

```typescript
const Particle = d.struct({
  position: d.vec3f,  // 16 bytes (12 + 4 padding)
  velocity: d.vec3f,  // 16 bytes (12 + 4 padding)
  lifetime: d.f32,    // 4 bytes
  // Struct size rounded up to alignment (16)
});

// Total size: 48 bytes (16 + 16 + 16, rounded to next multiple of 16)
```

### Loose Schemas for Vertex Data

When working with vertex buffers that don't need to function as storage or uniform buffers, you can use "loose schemas" to avoid alignment restrictions and save memory:

```typescript
// Standard struct with alignment (32 bytes)
const ParticleGeometry = d.struct({
  position: d.vec3f,  // 16 bytes (aligned)
  color: d.vec4f,     // 16 bytes
});

// Loose struct without alignment padding (28 bytes)
const LooseParticleGeometry = d.unstruct({
  position: d.vec3f,  // 12 bytes (no padding)
  color: d.vec4f,     // 16 bytes
});

// Use d.disarrayOf instead of d.arrayOf for loose arrays
const LooseVertexBuffer = d.disarrayOf(LooseParticleGeometry);
```

The size of `LooseParticleGeometry` is 28 bytes compared to the aligned version's larger footprint. This can provide significant memory savings when working with large amounts of vertex data.

## Best Practices and Common Pitfalls

### Best Practices

**1. Use Type Inference Everywhere**
```typescript
// Good: Single source of truth
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
});
type Particle = d.Infer<typeof Particle>;

// Avoid: Duplicate type definitions
```

**2. Leverage Nested Structs for Organization**
```typescript
// Good: Organized, reusable schemas
const Material = d.struct({
  albedo: d.vec3f,
  roughness: d.f32,
});

const Mesh = d.struct({
  transform: d.mat4x4f,
  material: Material,
});

// Avoid: Flat, monolithic structs
```

**3. Use Descriptive Field Names**
```typescript
// Good
const Camera = d.struct({
  viewMatrix: d.mat4x4f,
  projectionMatrix: d.mat4x4f,
  nearPlane: d.f32,
  farPlane: d.f32,
});

// Avoid
const Camera = d.struct({
  m1: d.mat4x4f,
  m2: d.mat4x4f,
  n: d.f32,
  f: d.f32,
});
```

**4. Match WGSL Shader Definitions**

Ensure your TypeGPU schemas exactly match your WGSL struct definitions:

```typescript
// TypeGPU schema
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Matching WGSL definition
// struct Particle {
//   position: vec3f,
//   velocity: vec3f,
//   lifetime: f32,
// }
```

**5. Use wgpu-matrix for Math Operations**
```typescript
import { mat4, vec3 } from 'wgpu-matrix';

const transform = d.mat4x4f();
mat4.identity(transform);
mat4.rotateY(transform, Math.PI / 4, transform);

const direction = d.vec3f(1, 0, 0);
vec3.normalize(direction, direction);
```

### Common Pitfalls

**1. Forgetting vec3 Alignment**

Remember that `vec3<f32>` is aligned to 16 bytes, not 12:

```typescript
// Pitfall: Assuming vec3f is 12 bytes
const wrongSize = 3 * 4;  // 12 bytes - WRONG!

// Correct: vec3f is aligned to 16 bytes
// TypeGPU handles this automatically
const Particle = d.struct({
  position: d.vec3f,  // Actually 16 bytes in buffer
});
```

**2. Runtime-Sized Arrays in Wrong Position**

Runtime-sized arrays must be the last field in a struct:

```typescript
// ERROR: Runtime-sized array not at end
const WrongBuffer = d.struct({
  particles: d.arrayOf(d.vec3f),  // Runtime-sized
  particleCount: d.u32,            // ERROR!
});

// Correct: Runtime-sized array at end
const CorrectBuffer = d.struct({
  particleCount: d.u32,
  particles: d.arrayOf(d.vec3f),  // OK!
});
```

**3. Mixing Loose and Regular Schemas**

Don't mix `d.unstruct` with regular `d.struct` in places where alignment matters:

```typescript
// Use loose schemas only for vertex buffers
const VertexData = d.unstruct({
  position: d.vec3f,
});

// Use regular structs for uniform/storage buffers
const UniformData = d.struct({
  viewMatrix: d.mat4x4f,
});
```

**4. Not Checking Matrix Format**

Be aware that TypeGPU matrices are column-major, like WGSL:

```typescript
// When manually creating matrix data, use column-major order
const mat = d.mat4x4f();
// Column 0, Column 1, Column 2, Column 3
```

**5. Forgetting to Import wgpu-matrix Utilities**

When using wgpu-matrix with TypeGPU, make sure to pass the destination parameter:

```typescript
import { vec3 } from 'wgpu-matrix';

const v = d.vec3f(1, 2, 3);

// Wrong: Creates Float32Array, doesn't modify v
const normalized = vec3.normalize(v);

// Correct: Modifies v in-place
vec3.normalize(v, v);
```

## Conclusion

TypeGPU's data schemas provide a robust, type-safe foundation for GPU programming in TypeScript. By abstracting away manual byte manipulation, alignment calculations, and padding management, they enable developers to focus on the logic of their GPU programs rather than low-level memory details.

The similarity to validation libraries like Zod makes schemas intuitive for TypeScript developers, while the automatic WGSL type mapping ensures correctness across the JavaScript-GPU boundary. Whether you're building particle systems, implementing ray tracers, or creating complex compute pipelines, TypeGPU's data schemas eliminate entire categories of bugs while maintaining excellent performance and full type safety.

For more information, explore the [official TypeGPU documentation](https://docs.swmansion.com/TypeGPU/) and the [Data Schemas guide](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/).

---

## Sources

- [Data Schemas | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/)
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/)
- [TypeGPU GitHub Repository](https://github.com/software-mansion/TypeGPU)
- [Working with wgpu-matrix | TypeGPU](https://docs.swmansion.com/TypeGPU/integration/working-with-wgpu-matrix/)
- [Buffers | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/buffers/)
- [Vertex Layouts | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/vertex-layouts/)
- [WebGPU Data Memory Layout](https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html)
