---
title: Buffers and Memory Management
sidebar:
  order: 20
---

## Overview

Buffers are contiguous regions of GPU memory that enable data transfer between JavaScript and shader programs. Understanding buffers and memory management is essential for building efficient WebGPU applications.

:::note[GPU Memory Model]
The GPU memory model differs significantly from CPU memory. WebGPU enforces strict separation between CPU-accessible and GPU-accessible memory, requiring explicit allocation, usage declaration, and synchronization between CPU and GPU operations.
:::

Buffers serve multiple purposes:
- **Vertex buffers** — Store geometry data for rendering
- **Index buffers** — Enable vertex reuse across triangles
- **Uniform buffers** — Hold read-only shader constants
- **Storage buffers** — Provide large read-write memory for compute

The WebGPU specification initializes all resources to zero, preventing information leakage from other applications or previous GPU operations.

## Buffer Types

### Vertex Buffers

Vertex buffers store per-vertex attribute data including positions, normals, texture coordinates, and colors. Unlike storage buffers accessed through array indexing, vertex buffers use an attribute system where WebGPU extracts data for each vertex automatically.

```javascript title="Creating a vertex buffer"
const vertices = new Float32Array([
  // x,    y,    r,    g,    b,    a
  -0.5, -0.5,  1.0,  0.0,  0.0,  1.0,
   0.5, -0.5,  0.0,  1.0,  0.0,  1.0,
   0.0,  0.5,  0.0,  0.0,  1.0,  1.0,
]);

const vertexBuffer = device.createBuffer({
  label: "triangle vertices",
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(vertexBuffer, 0, vertices);
```

Vertex buffers support per-vertex and per-instance stepping modes for efficient instanced rendering.

### Index Buffers

Index buffers reduce memory by allowing vertex reuse. Instead of duplicating vertices, indices point to vertices in the vertex buffer:

```javascript title="Creating an index buffer"
const indices = new Uint32Array([
  0, 1, 2,  // first triangle
  2, 1, 3,  // second triangle (reuses vertices 1 and 2)
]);

const indexBuffer = device.createBuffer({
  label: "quad indices",
  size: indices.byteLength,
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(indexBuffer, 0, indices);

// Usage in render pass
pass.setIndexBuffer(indexBuffer, "uint32");
pass.drawIndexed(6, 1);
```

| Format | Max Vertices | Use Case |
|--------|-------------|----------|
| `uint16` | 65,536 | Smaller meshes, memory savings |
| `uint32` | 4+ billion | Large geometries |

### Uniform Buffers

Uniform buffers hold small, read-only constants that remain unchanged during shader execution—transformation matrices, lighting parameters, material properties.

```javascript title="Creating a uniform buffer"
const uniformData = new Float32Array([
  1.0, 0.0, 0.0, 0.0,  // matrix row 1
  0.0, 1.0, 0.0, 0.0,  // matrix row 2
  0.0, 0.0, 1.0, 0.0,  // matrix row 3
  0.5, 0.5, 0.0, 1.0,  // matrix row 4 (translation)
]);

const uniformBuffer = device.createBuffer({
  label: "transform uniforms",
  size: uniformData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(uniformBuffer, 0, uniformData);
```

:::caution[Size Limit]
Uniform buffers are limited to 64 KiB (65,536 bytes). This constraint encourages efficient data packing and makes them unsuitable for large datasets—use storage buffers instead.
:::

Uniform buffers typically offer better performance than storage buffers for small, frequently accessed data due to aggressive GPU caching.

### Storage Buffers

Storage buffers provide large-capacity read-write memory with a guaranteed maximum of 128 MiB—over 2,000 times larger than uniform buffers:

```javascript title="Creating a storage buffer"
const particleCount = 10000;
const particleData = new Float32Array(particleCount * 4); // x, y, vx, vy

const storageBuffer = device.createBuffer({
  label: "particle data",
  size: particleData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

device.queue.writeBuffer(storageBuffer, 0, particleData);
```

Storage buffers support both `var<storage, read>` (read-only) and `var<storage, read_write>` (read-write) access in WGSL.

## Buffer Creation

```javascript title="Buffer descriptor options"
const buffer = device.createBuffer({
  label: "descriptive name",     // Aids debugging
  size: 1024,                     // Size in bytes (required)
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: false,        // Optional, default false
});
```

### Usage Flags

Usage flags declare all ways a buffer will be used. Combine multiple flags with bitwise OR (`|`).

<details>
<summary>**Primary Usage Flags**</summary>

| Flag | Purpose |
|------|---------|
| `VERTEX` | Vertex buffer in draw operations |
| `INDEX` | Index buffer for indexed drawing |
| `UNIFORM` | Uniform buffer in bind groups |
| `STORAGE` | Storage buffer in bind groups |
| `INDIRECT` | Indirect draw/dispatch arguments |

</details>

<details>
<summary>**Copy and Map Flags**</summary>

| Flag | Purpose |
|------|---------|
| `COPY_SRC` | Source of copy operations |
| `COPY_DST` | Destination of copy operations |
| `MAP_READ` | CPU read access via mapping |
| `MAP_WRITE` | CPU write access via mapping |

</details>

:::danger[Invalid Combinations]
`MAP_READ` and `MAP_WRITE` cannot be combined with `STORAGE` or `UNIFORM`. Mappable buffers reside in CPU-accessible memory, while storage/uniform buffers need fast GPU-local memory. Use staging buffers for transfers.
:::

```javascript title="Valid buffer configurations"
// Staging buffer for reading compute results
const readbackBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// Storage buffer for GPU operations
const computeBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
```

### Size and Alignment

| Buffer Type | Alignment Requirement |
|-------------|----------------------|
| Uniform bindings | 256 bytes |
| Storage bindings | 32 bytes (some implementations) |
| Copy operations | Source and destination offsets aligned |

Always ensure buffers accommodate data plus required padding. Misaligned access causes validation errors or incorrect data interpretation.

### mappedAtCreation

Efficient initialization without separate mapping operations:

```javascript title="Initialize buffer at creation" {1,5-8}
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});

// Buffer is immediately mapped for writing
const arrayBuffer = buffer.getMappedRange();
const float32View = new Float32Array(arrayBuffer);
float32View.set([1.0, 2.0, 3.0, 4.0]);

// Unmap to make available for GPU
buffer.unmap();
```

This pattern is ideal for static geometry and lookup tables initialized once.

## TypeGPU Buffer Creation

TypeGPU simplifies buffer management with typed schemas and automatic memory layout calculations.

### Basic Buffer Creation

```typescript title="TypeGPU buffer creation"
import * as d from "typegpu/data";

// Single primitive value
const counterBuffer = root.createBuffer(d.u32);

// Array of values
const positionsBuffer = root.createBuffer(d.arrayOf(d.vec3f, 100));

// Structured data
const particleBuffer = root.createBuffer(d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  health: d.f32,
  age: d.f32,
}));
```

TypeGPU automatically calculates buffer size, applies alignment rules, and generates corresponding WGSL type definitions.

### Usage Flags

```typescript title="Chaining usage flags"
const buffer = root
  .createBuffer(d.arrayOf(d.f32, 1000))
  .$usage("storage")
  .$usage("vertex");
```

| TypeGPU Usage | WebGPU Equivalent |
|---------------|-------------------|
| `'uniform'` | `GPUBufferUsage.UNIFORM` |
| `'storage'` | `GPUBufferUsage.STORAGE` |
| `'vertex'` | `GPUBufferUsage.VERTEX` |

### Initial Values

```typescript title="Type-safe initialization"
// Single value
const timeBuffer = root.createBuffer(d.f32, 0.0);

// Array initialization
const colorsBuffer = root.createBuffer(d.arrayOf(d.vec3f, 3), [
  d.vec3f(1, 0, 0),
  d.vec3f(0, 1, 0),
  d.vec3f(0, 0, 1),
]);

// Struct initialization
const cameraBuffer = root.createBuffer(
  d.struct({
    position: d.vec3f,
    fov: d.f32,
  }),
  {
    position: d.vec3f(0, 5, 10),
    fov: 45.0,
  }
);
```

The type system validates initial values against the schema at compile time.

### Automatic Serialization

TypeGPU handles binary serialization following WebGPU alignment rules:

```typescript title="Automatic binary conversion"
const transformBuffer = root.createBuffer(d.struct({
  scale: d.vec3f,
  rotation: d.vec4f,
  translation: d.vec3f,
}));

// Type-safe write with automatic serialization
transformBuffer.write({
  scale: d.vec3f(2, 2, 2),
  rotation: d.vec4f(0, 0, 0, 1),
  translation: d.vec3f(10, 0, 0),
});
```

This eliminates manual byte packing and ensures correct alignment.

## Reading and Writing Data

### WebGPU: queue.writeBuffer()

The simplest method for CPU-to-GPU data transfer:

```javascript title="Direct buffer write"
const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
device.queue.writeBuffer(buffer, 0, data);
```

Data is copied immediately—you can safely modify the source array after the call. The write executes before any subsequently queued GPU operations referencing the buffer.

:::tip[Use Cases]
- Per-frame uniform updates
- Uploading geometry data
- Initializing storage buffers
:::

### Buffer Mapping for Read Access

Reading GPU results requires a multi-step process with staging buffers:

```javascript title="Reading GPU results" {1-5,7-11,17,20-22}
// 1. GPU-local result buffer
const resultBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// 2. CPU-accessible staging buffer
const stagingBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// 3. Execute compute work...

// 4. Copy GPU results to staging buffer
const encoder = device.createCommandEncoder();
encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 256);
device.queue.submit([encoder.finish()]);

// 5. Map staging buffer for reading
await stagingBuffer.mapAsync(GPUMapMode.READ);
const data = new Float32Array(stagingBuffer.getMappedRange());
console.log("Results:", Array.from(data));

// 6. Unmap when done
stagingBuffer.unmap();
```

### Map/Unmap Lifecycle

| State | Description |
|-------|-------------|
| **Unmapped** | Default state; available for GPU commands |
| **Mapping Pending** | After `mapAsync()`, before promise resolves |
| **Mapped** | CPU can access via `getMappedRange()` |
| **Unmapped** | After `unmap()`, returns to GPU-accessible |

:::danger[Critical Rule]
A buffer can only be used in GPU operations while unmapped. Using a mapped buffer in commands causes validation errors.
:::

### TypeGPU: Type-Safe I/O

```typescript title="TypeGPU read/write"
const particleBuffer = root.createBuffer(d.struct({
  position: d.vec2f,
  velocity: d.vec2f,
  mass: d.f32,
}));

// Type-checked write
particleBuffer.write({
  position: d.vec2f(100, 200),
  velocity: d.vec2f(1.5, -2.0),
  mass: 1.0,
});

// Type-safe read
const result = await particleBuffer.read();
console.log("Position:", result.position);
console.log("Velocity:", result.velocity);
```

TypeGPU handles staging buffers, copies, mapping, and deserialization transparently.

## Memory Visibility

### GPU vs CPU Memory

Modern GPUs contain high-bandwidth memory (GDDR/HBM) optimized for parallel shader access. This GPU-local memory isn't directly CPU-accessible.

| Memory Type | Access | Speed | Use |
|-------------|--------|-------|-----|
| GPU-local | GPU only | Fastest | Storage, uniform, vertex, index |
| CPU-accessible | CPU + slow GPU | Slower | Mappable buffers |

### Staging Buffer Pattern

```javascript title="Complete staging buffer workflow"
// GPU-side storage buffer
const gpuBuffer = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

// CPU-accessible staging buffers
const uploadStaging = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
});

const downloadStaging = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// Upload: CPU -> Staging -> GPU
await uploadStaging.mapAsync(GPUMapMode.WRITE);
new Float32Array(uploadStaging.getMappedRange()).set(cpuData);
uploadStaging.unmap();

const encoder = device.createCommandEncoder();
encoder.copyBufferToBuffer(uploadStaging, 0, gpuBuffer, 0, 4096);
device.queue.submit([encoder.finish()]);

// Download: GPU -> Staging -> CPU
const downloadEncoder = device.createCommandEncoder();
downloadEncoder.copyBufferToBuffer(gpuBuffer, 0, downloadStaging, 0, 4096);
device.queue.submit([downloadEncoder.finish()]);

await downloadStaging.mapAsync(GPUMapMode.READ);
const results = new Float32Array(downloadStaging.getMappedRange()).slice();
downloadStaging.unmap();
```

## Buffer Update Patterns

### Frame-by-Frame Updates

```javascript title="Per-frame uniform update"
function frame(time) {
  const uniforms = new Float32Array([
    time * 0.001,
    Math.sin(time * 0.001),
    Math.cos(time * 0.001),
    0.0,
  ]);

  device.queue.writeBuffer(uniformBuffer, 0, uniforms);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass(renderPassDescriptor);
  // ... rendering commands ...
  pass.end();
  device.queue.submit([encoder.finish()]);

  requestAnimationFrame(frame);
}
```

### Double Buffering

For larger updates or when GPU work spans multiple frames:

```javascript title="Ring buffer pattern" {4,9-10,13}
const bufferSize = 65536;
const numBuffers = 2;
let currentBufferIndex = 0;

const buffers = Array.from({ length: numBuffers }, (_, i) =>
  device.createBuffer({
    label: `ring buffer ${i}`,
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
);

function frame() {
  const buffer = buffers[currentBufferIndex];
  const nextIndex = (currentBufferIndex + 1) % numBuffers;

  // Update NEXT buffer while GPU processes current
  device.queue.writeBuffer(buffers[nextIndex], 0, nextFrameData);

  // Render with current buffer...

  currentBufferIndex = nextIndex;
  requestAnimationFrame(frame);
}
```

:::danger[Write-While-In-Use]
Never write to a buffer currently in use by the GPU. Commands submitted via `queue.submit()` execute asynchronously—use separate buffers or wait for completion.
:::

## Buffer Pooling

Creating and destroying buffers frequently causes overhead and fragmentation. Maintain pools of reusable buffers:

```javascript title="Buffer pool implementation"
class BufferPool {
  constructor(device, size, usage) {
    this.device = device;
    this.size = size;
    this.usage = usage;
    this.available = [];
    this.inUse = new Set();
  }

  acquire() {
    let buffer = this.available.pop();
    if (!buffer) {
      buffer = this.device.createBuffer({
        size: this.size,
        usage: this.usage,
      });
    }
    this.inUse.add(buffer);
    return buffer;
  }

  release(buffer) {
    if (this.inUse.delete(buffer)) {
      this.available.push(buffer);
    }
  }

  destroy() {
    [...this.available, ...this.inUse].forEach(b => b.destroy());
  }
}
```

### Sub-allocation

Store multiple logical buffers in one physical buffer:

```javascript title="Large buffer with sub-allocations"
const largeBuffer = device.createBuffer({
  size: 1048576, // 1 MiB
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// Write different uniforms at different offsets
device.queue.writeBuffer(largeBuffer, 0, cameraUniforms);
device.queue.writeBuffer(largeBuffer, 256, lightUniforms);
device.queue.writeBuffer(largeBuffer, 512, materialUniforms);

// Bind with offsets
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{
    binding: 0,
    resource: { buffer: largeBuffer, offset: 0, size: 256 },
  }],
});
```

### Batching Updates

```javascript title="Batch vs individual writes"
// ✗ Inefficient: many small writes
for (let i = 0; i < 100; i++) {
  device.queue.writeBuffer(buffer, i * 16, objects[i].transform);
}

// ✓ Efficient: single batched write
const batchedData = new Float32Array(100 * 4);
for (let i = 0; i < 100; i++) {
  batchedData.set(objects[i].transform, i * 4);
}
device.queue.writeBuffer(buffer, 0, batchedData);
```

## Alignment Considerations

:::danger[WGSL Alignment Rules]
WGSL struct members have alignment requirements. `vec3<f32>` has 16-byte alignment despite being 12 bytes:

```wgsl
struct Uniforms {
  color: vec3<f32>,  // offset 0, takes 16 bytes
  intensity: f32,    // offset 16
}
```

When creating typed arrays manually, account for padding:

```javascript
// ✗ Wrong: only 4 floats
const data = new Float32Array([1, 0, 0, 0.5]);

// ✓ Correct: explicit padding for vec3
const data = new Float32Array([
  1, 0, 0, 0,     // vec3 color (4th element is padding)
  0.5, 0, 0, 0,   // f32 intensity (3 padding floats)
]);
```
:::

TypeGPU eliminates these issues by handling alignment automatically.

## Resource Cleanup

:::caution[Memory Leaks]
Buffers aren't automatically garbage collected. Destroy them explicitly:

```javascript
const tempBuffer = device.createBuffer({
  size: 1024,
  usage: GPUBufferUsage.STORAGE,
});

// ... use buffer ...

tempBuffer.destroy(); // Free GPU memory
```

Failing to destroy buffers leads to GPU memory exhaustion in long-running applications.
:::
