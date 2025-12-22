# Buffers and Memory Management

## Overview

Buffers are the fundamental building blocks of GPU memory management in WebGPU. They represent contiguous regions of memory accessible by both the CPU and GPU, enabling data transfer between JavaScript/TypeScript code and shader programs running on the graphics processor. Understanding buffers and their proper management is crucial for building efficient WebGPU applications.

The GPU memory model differs significantly from traditional CPU memory. While CPU programs can freely allocate and access memory through pointers, GPU memory requires explicit allocation, usage declaration, and careful synchronization between CPU and GPU operations. WebGPU enforces strict separation between CPU-accessible and GPU-accessible memory spaces, requiring developers to use specific APIs for data transfer.

Buffers in WebGPU serve multiple purposes: they can store vertex data for rendering geometry, hold uniform constants that remain unchanged during shader execution, provide large storage arrays for compute operations, or act as staging areas for transferring data between CPU and GPU. Each buffer must declare its intended usage upfront through usage flags, allowing the implementation to optimize memory placement and access patterns.

The WebGPU specification conceptually initializes all resources to zero, preventing information leakage from other applications or previous GPU operations. This security feature ensures that uninitialized buffers don't expose sensitive data, though implementations may optimize away explicit clearing when developers provide initial data.

TypeGPU builds upon WebGPU's buffer system by adding type safety and automatic serialization. While WebGPU requires manual calculation of buffer sizes and careful binary packing of data, TypeGPU uses TypeScript schemas to automatically handle these low-level details, reducing errors and improving developer productivity.

## Buffer Types in WebGPU

WebGPU defines several distinct buffer types, each optimized for specific use cases:

### Vertex Buffers

Vertex buffers store per-vertex attribute data that describes 3D geometry. This includes positions, normals, texture coordinates, colors, and any custom attributes your rendering pipeline requires. Unlike storage buffers which are accessed through array indexing in shaders, vertex buffers work through an indirect attribute system where WebGPU automatically extracts the appropriate data for each vertex.

When configuring a vertex buffer, you define a vertex layout that specifies how raw bytes map to shader attributes. This includes the stride (bytes between consecutive vertices), attribute formats (like `float32x2` for 2D positions), and byte offsets for each attribute within the vertex structure.

```javascript
// WebGPU vertex buffer example
const vertices = new Float32Array([
  // x,    y,     r,    g,    b,    a
  -0.5, -0.5,   1.0,  0.0,  0.0,  1.0,
   0.5, -0.5,   0.0,  1.0,  0.0,  1.0,
   0.0,  0.5,   0.0,  0.0,  1.0,  1.0,
]);

const vertexBuffer = device.createBuffer({
  label: 'triangle vertices',
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(vertexBuffer, 0, vertices);
```

Vertex buffers support per-vertex and per-instance stepping modes, enabling efficient instanced rendering where multiple copies of the same geometry share vertex data while varying per-instance attributes like transformation matrices or colors.

### Index Buffers

Index buffers reduce memory consumption by allowing vertices to be reused across multiple triangles. Instead of duplicating vertex data, an index buffer contains integer indices pointing to vertices in the vertex buffer. This is particularly valuable for mesh geometry where adjacent triangles share edges and vertices.

```javascript
// WebGPU index buffer example
const indices = new Uint32Array([
  0, 1, 2,  // first triangle
  2, 1, 3,  // second triangle (reuses vertices 1 and 2)
]);

const indexBuffer = device.createBuffer({
  label: 'quad indices',
  size: indices.byteLength,
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(indexBuffer, 0, indices);

// Usage in render pass
pass.setIndexBuffer(indexBuffer, 'uint32');
pass.drawIndexed(6, 1);  // 6 indices, 1 instance
```

Index buffers support 16-bit (`uint16`) and 32-bit (`uint32`) index formats, with 16-bit providing memory savings for smaller meshes and 32-bit supporting larger geometries with more than 65,536 vertices.

### Uniform Buffers

Uniform buffers hold small, read-only constant data that remains unchanged throughout shader execution. They're ideal for transformation matrices, lighting parameters, material properties, and other configuration data that applies uniformly to all shader invocations.

The key limitation of uniform buffers is their maximum size: the WebGPU specification guarantees only 64 kiB (65,536 bytes) of uniform buffer space. This constraint encourages efficient data packing and makes uniform buffers unsuitable for large datasets.

```javascript
// WebGPU uniform buffer example
const uniformData = new Float32Array([
  1.0, 0.0, 0.0, 0.0,  // matrix row 1
  0.0, 1.0, 0.0, 0.0,  // matrix row 2
  0.0, 0.0, 1.0, 0.0,  // matrix row 3
  0.5, 0.5, 0.0, 1.0,  // matrix row 4 (translation)
]);

const uniformBuffer = device.createBuffer({
  label: 'transform uniforms',
  size: uniformData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(uniformBuffer, 0, uniformData);
```

Uniform buffers typically offer better performance than storage buffers for small, frequently accessed constants because GPU hardware can cache uniform data more aggressively and provide faster access to shader cores.

### Storage Buffers

Storage buffers provide large-capacity read-write memory accessible from shaders. With a guaranteed maximum size of 128 MiB (134,217,728 bytes)—over 2,000 times larger than uniform buffers—they excel at handling substantial datasets for compute operations, particle systems, procedural generation, and advanced rendering techniques.

Storage buffers support both read-only (`var<storage, read>`) and read-write (`var<storage, read_write>`) access patterns in WGSL shaders. The read-write capability enables compute shaders to modify data in place, eliminating the need for separate input and output buffers in many scenarios.

```javascript
// WebGPU storage buffer example
const particleCount = 10000;
const particleData = new Float32Array(particleCount * 4);  // x, y, vx, vy per particle

const storageBuffer = device.createBuffer({
  label: 'particle data',
  size: particleData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

device.queue.writeBuffer(storageBuffer, 0, particleData);
```

While storage buffers offer tremendous flexibility, they may exhibit slightly lower performance than uniform buffers for small, constant data due to different caching characteristics and access patterns in GPU hardware.

## WebGPU Buffer Creation

Creating buffers in WebGPU requires calling `device.createBuffer()` with a `GPUBufferDescriptor` object that specifies the buffer's properties:

```javascript
const buffer = device.createBuffer({
  label: 'descriptive name',      // optional, aids debugging
  size: 1024,                      // size in bytes (required)
  usage: GPUBufferUsage.STORAGE |  // usage flags (required)
         GPUBufferUsage.COPY_DST,
  mappedAtCreation: false,         // optional, default false
});
```

### Usage Flags

Usage flags are a critical aspect of buffer creation, declaring all the ways a buffer will be used throughout its lifetime. These flags enable the WebGPU implementation to optimize memory placement and validate operations. Multiple flags can be combined using the bitwise OR operator (`|`).

**Primary Usage Flags:**

- `GPUBufferUsage.VERTEX`: Buffer can be used as a vertex buffer in draw operations
- `GPUBufferUsage.INDEX`: Buffer can be used as an index buffer for indexed drawing
- `GPUBufferUsage.UNIFORM`: Buffer can be bound as a uniform buffer in bind groups
- `GPUBufferUsage.STORAGE`: Buffer can be bound as a storage buffer in bind groups
- `GPUBufferUsage.INDIRECT`: Buffer can hold indirect draw/dispatch command arguments

**Copy and Map Flags:**

- `GPUBufferUsage.COPY_SRC`: Buffer can be the source of copy operations
- `GPUBufferUsage.COPY_DST`: Buffer can be the destination of copy operations
- `GPUBufferUsage.MAP_READ`: Buffer can be mapped for CPU read access
- `GPUBufferUsage.MAP_WRITE`: Buffer can be mapped for CPU write access

**Important Constraints:**

Not all usage flag combinations are valid. Notably, `MAP_READ` and `MAP_WRITE` cannot be combined with `STORAGE` or `UNIFORM`. This restriction exists because mappable buffers reside in CPU-accessible memory, while storage and uniform buffers need fast GPU-local memory. To transfer data to/from storage or uniform buffers, you must use intermediate staging buffers or the `queue.writeBuffer()` API.

```javascript
// Valid: Staging buffer for reading compute results
const readbackBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// Valid: Storage buffer for GPU operations
const computeBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

// INVALID: Cannot combine MAP_READ with STORAGE
// const invalid = device.createBuffer({
//   size: 256,
//   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE,
// });
```

### Size Requirements and Alignment

Buffer sizes are specified in bytes and must be non-negative integers. While you can create zero-sized buffers, they're rarely useful in practice.

Different buffer usages impose specific alignment requirements:

- Uniform buffer bindings must be aligned to 256 bytes
- Storage buffer bindings must be aligned to 32 bytes (on some implementations)
- Copy operations have alignment requirements for both source and destination offsets

When creating buffers, you're responsible for ensuring sufficient size to accommodate your data plus any padding required for alignment. Failing to account for alignment in data structures can lead to shader access violations or incorrect data interpretation.

### mappedAtCreation Option

The `mappedAtCreation` option provides an efficient way to initialize buffer contents without separate mapping operations:

```javascript
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});

// Buffer is immediately mapped for writing
const arrayBuffer = buffer.getMappedRange();
const float32View = new Float32Array(arrayBuffer);
float32View.set([1.0, 2.0, 3.0, 4.0]);

// Unmap to make available for GPU use
buffer.unmap();
```

This pattern is particularly useful for static geometry and lookup tables that are initialized once and never modified. It's more efficient than creating an unmapped buffer and subsequently mapping it because the implementation can optimize the initial allocation.

## TypeGPU Buffer Creation

TypeGPU simplifies buffer management through typed schemas and automatic memory layout calculations. Instead of manually computing byte sizes and packing data, you define the logical structure of your data and let TypeGPU handle the details.

### Basic Buffer Creation

Create buffers using `root.createBuffer()` with a data schema:

```typescript
import * as d from 'typegpu/data';

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

TypeGPU automatically calculates the required buffer size, applies proper alignment rules, and generates the corresponding WGSL type definitions for use in shaders.

### Usage Flags in TypeGPU

Declare buffer usage through the `$usage()` method, which can be chained for multiple usages:

```typescript
const buffer = root.createBuffer(d.arrayOf(d.f32, 1000))
  .$usage('storage')
  .$usage('vertex');
```

Supported usage flags in TypeGPU include:
- `'uniform'`: Uniform buffer binding
- `'storage'`: Storage buffer binding
- `'vertex'`: Vertex buffer usage

The type system prevents invalid combinations and ensures buffers are properly configured for their intended shader bindings.

### Initial Values

TypeGPU buffers accept typed initial values that match the schema:

```typescript
// Single value
const timeBuffer = root.createBuffer(d.f32, 0.0);

// Array initialization
const colorsBuffer = root.createBuffer(
  d.arrayOf(d.vec3f, 3),
  [
    d.vec3f(1, 0, 0),  // red
    d.vec3f(0, 1, 0),  // green
    d.vec3f(0, 0, 1),  // blue
  ]
);

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

The type system validates that initial values match the schema, catching type mismatches at compile time rather than runtime.

### Typed Buffers with Automatic Serialization

One of TypeGPU's most powerful features is automatic binary serialization. When you write JavaScript objects to buffers, TypeGPU converts them to the appropriate binary format following WebGPU alignment rules:

```typescript
const transformBuffer = root.createBuffer(d.struct({
  scale: d.vec3f,
  rotation: d.vec4f,  // quaternion
  translation: d.vec3f,
}));

// Type-safe write with automatic serialization
transformBuffer.write({
  scale: d.vec3f(2, 2, 2),
  rotation: d.vec4f(0, 0, 0, 1),
  translation: d.vec3f(10, 0, 0),
});
```

This eliminates the error-prone manual process of packing data into typed arrays while ensuring correct alignment and byte ordering.

### Fixed Resources

TypeGPU's "fixed resources" pattern enables buffer reuse across pipeline executions, improving performance by reducing allocation overhead:

```typescript
const fixedBuffer = root.createBuffer(d.arrayOf(d.f32, 1000))
  .$usage('storage');

// Use the same buffer across multiple compute passes
for (let i = 0; i < 100; i++) {
  // Update buffer contents
  fixedBuffer.write(newData);

  // Execute pipeline using the fixed buffer
  pipeline.execute({ data: fixedBuffer });
}
```

Fixed resources represent a more advanced usage pattern where you manually manage buffer lifecycles for optimization purposes.

### Wrapping Existing WebGPU Buffers

TypeGPU can wrap existing `GPUBuffer` objects to add type safety:

```typescript
const gpuBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const typedBuffer = root.createBuffer(d.arrayOf(d.f32, 64), gpuBuffer);
```

This interoperability allows gradual adoption of TypeGPU in existing WebGPU codebases or integration with libraries that provide their own buffer management.

## Reading and Writing Data

Moving data between CPU and GPU requires careful orchestration to maintain performance and correctness. Both WebGPU and TypeGPU provide APIs for data transfer, each with distinct tradeoffs.

### WebGPU Approach

WebGPU offers several mechanisms for buffer data transfer, each suited to different scenarios.

#### queue.writeBuffer()

The simplest and most commonly used method for updating buffer contents from the CPU:

```javascript
const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
device.queue.writeBuffer(buffer, 0, data);
```

This method queues a write operation that executes asynchronously. The data is copied immediately, so you can safely modify the source typed array after the call returns. The write occurs before any subsequently queued GPU operations that reference the buffer.

`queue.writeBuffer()` is ideal for:
- Per-frame uniform updates
- Uploading geometry data
- Initializing storage buffers

It's not suitable for reading data back from the GPU—for that, you need buffer mapping.

#### Buffer Mapping for Write Access

For initial buffer population or scenarios where you need more control, use `mappedAtCreation` or map an existing buffer:

```javascript
// Create mapped
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});

const arrayBuffer = buffer.getMappedRange();
const view = new Float32Array(arrayBuffer);
view[0] = 42.0;
buffer.unmap();

// Map existing buffer (requires MAP_WRITE flag)
const mappableBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
});

await mappableBuffer.mapAsync(GPUMapMode.WRITE);
const mappedRange = mappableBuffer.getMappedRange();
new Float32Array(mappedRange)[0] = 42.0;
mappableBuffer.unmap();
```

Remember that buffers with `MAP_WRITE` cannot be used as storage or uniform buffers directly—they serve as staging buffers that you copy to GPU-local buffers.

#### Buffer Mapping for Read Access

Reading GPU computation results requires a multi-step process using staging buffers:

```javascript
// 1. Create result buffer (GPU-local, not mappable)
const resultBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// 2. Create staging buffer (CPU-accessible for reading)
const stagingBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// 3. Execute compute work (omitted for brevity)

// 4. Copy GPU results to staging buffer
const encoder = device.createCommandEncoder();
encoder.copyBufferToBuffer(
  resultBuffer, 0,  // source
  stagingBuffer, 0, // destination
  256               // size
);
device.queue.submit([encoder.finish()]);

// 5. Map staging buffer for reading
await stagingBuffer.mapAsync(GPUMapMode.READ);
const data = new Float32Array(stagingBuffer.getMappedRange());
console.log('Results:', Array.from(data));

// 6. Unmap when done
stagingBuffer.unmap();
```

This pattern ensures separation between GPU-optimized memory and CPU-accessible memory, allowing both to operate at peak efficiency.

#### Buffer Lifecycle: map() and unmap()

The mapping lifecycle follows strict rules:

1. **Unmapped**: Default state after creation (unless `mappedAtCreation: true`)
2. **Mapping Pending**: After calling `mapAsync()`, before promise resolves
3. **Mapped**: After `mapAsync()` promise resolves, `getMappedRange()` available
4. **Unmapped**: After calling `unmap()`, returning to state 1

A buffer can only be used in GPU operations while unmapped. Attempting to use a mapped buffer in a command buffer results in validation errors. Similarly, calling `getMappedRange()` on an unmapped buffer fails.

```javascript
// Correct sequence
await buffer.mapAsync(GPUMapMode.READ);
const data = buffer.getMappedRange();
// Process data...
buffer.unmap();

// Now buffer can be used in GPU commands
```

### TypeGPU Approach

TypeGPU abstracts the complexities of buffer mapping and binary serialization behind simple, type-safe methods.

#### Type-Safe Writes

The `write()` method accepts JavaScript objects matching the buffer schema:

```typescript
const particleBuffer = root.createBuffer(d.struct({
  position: d.vec2f,
  velocity: d.vec2f,
  mass: d.f32,
}));

// Type-checked write operation
particleBuffer.write({
  position: d.vec2f(100, 200),
  velocity: d.vec2f(1.5, -2.0),
  mass: 1.0,
});
```

TypeGPU handles:
- Binary serialization of JavaScript objects
- Proper alignment and padding
- Queuing the write operation
- Type validation at compile time

This eliminates an entire class of bugs related to incorrect byte offsets, alignment violations, and data type mismatches.

#### Type-Safe Reads

Reading buffer data returns a promise resolving to a typed JavaScript object:

```typescript
const result = await particleBuffer.read();
console.log('Position:', result.position);
console.log('Velocity:', result.velocity);
console.log('Mass:', result.mass);
```

Under the hood, TypeGPU creates staging buffers, performs the necessary copies, maps the buffer, deserializes the binary data, and returns a properly typed object—all transparently to the developer.

#### Automatic Binary Conversion

TypeGPU's automatic conversion handles complex nested structures:

```typescript
const sceneBuffer = root.createBuffer(d.struct({
  lights: d.arrayOf(d.struct({
    position: d.vec3f,
    color: d.vec3f,
    intensity: d.f32,
  }), 4),
  ambientColor: d.vec3f,
}));

sceneBuffer.write({
  lights: [
    { position: d.vec3f(10, 20, 30), color: d.vec3f(1, 1, 1), intensity: 1.0 },
    { position: d.vec3f(-10, 20, 30), color: d.vec3f(1, 0.8, 0.6), intensity: 0.8 },
    // ... more lights
  ],
  ambientColor: d.vec3f(0.1, 0.1, 0.15),
});
```

The schema-driven approach ensures that regardless of complexity, serialization and deserialization remain correct and type-safe.

## Memory Visibility

Understanding memory visibility is crucial for correct WebGPU programming. The GPU and CPU operate in separate memory spaces with different performance characteristics and access patterns.

### GPU Memory vs CPU Memory

Modern GPUs contain their own high-bandwidth memory optimized for parallel access from thousands of shader cores. This GPU-local memory (often GDDR or HBM) provides substantially higher bandwidth than system RAM but isn't directly accessible from CPU code.

Conversely, system RAM is optimized for CPU access patterns and cache hierarchies. While the GPU can access system memory through PCIe or other interconnects, such access is significantly slower than GPU-local memory.

WebGPU's buffer model reflects this hardware reality:

- **GPU-optimized buffers**: Storage, uniform, vertex, and index buffers typically reside in GPU memory for fast shader access
- **CPU-accessible buffers**: Mappable buffers (with `MAP_READ` or `MAP_WRITE`) reside in system memory or a special shared memory region
- **Transfer mechanisms**: Copy operations and `queue.writeBuffer()` move data between these memory spaces

This separation means you cannot directly read from or write to storage buffers from JavaScript—you must use explicit transfer operations.

### Staging Buffers for Data Transfer

Staging buffers act as intermediaries between CPU and GPU memory spaces. The typical pattern involves:

```javascript
// GPU-side storage buffer
const gpuBuffer = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

// CPU-accessible staging buffer for uploads
const uploadStagingBuffer = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
});

// CPU-accessible staging buffer for downloads
const downloadStagingBuffer = device.createBuffer({
  size: 4096,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// Upload: CPU -> Staging -> GPU
await uploadStagingBuffer.mapAsync(GPUMapMode.WRITE);
new Float32Array(uploadStagingBuffer.getMappedRange()).set(cpuData);
uploadStagingBuffer.unmap();

const encoder = device.createCommandEncoder();
encoder.copyBufferToBuffer(uploadStagingBuffer, 0, gpuBuffer, 0, 4096);
device.queue.submit([encoder.finish()]);

// Download: GPU -> Staging -> CPU
const downloadEncoder = device.createCommandEncoder();
downloadEncoder.copyBufferToBuffer(gpuBuffer, 0, downloadStagingBuffer, 0, 4096);
device.queue.submit([downloadEncoder.finish()]);

await downloadStagingBuffer.mapAsync(GPUMapMode.READ);
const results = new Float32Array(downloadStagingBuffer.getMappedRange()).slice();
downloadStagingBuffer.unmap();
```

While verbose, this pattern provides explicit control over memory transfers and enables performance optimization through asynchronous operations.

### Map/Unmap Lifecycle

The map/unmap lifecycle manages the transition between CPU and GPU access:

- **Unmapped state**: Buffer is available for GPU commands but not CPU access
- **Map operation**: Requests CPU access, returns a promise that resolves when safe
- **Mapped state**: CPU can read/write through `getMappedRange()`, GPU operations forbidden
- **Unmap operation**: Returns buffer to GPU-accessible state, invalidates mapped ranges

This lifecycle prevents data races where CPU and GPU simultaneously access the same memory. WebGPU enforces strict validation: attempting to use a mapped buffer in GPU commands or map an already-mapped buffer results in errors.

The asynchronous nature of `mapAsync()` reflects that mapping may require waiting for pending GPU operations to complete. Always await the returned promise before accessing mapped data.

## Buffer Update Patterns

Efficient buffer updates are essential for achieving high frame rates and responsive applications.

### Frame-by-Frame Updates

Many applications need to update uniform or storage buffers every frame. The most straightforward pattern uses `queue.writeBuffer()`:

```javascript
function frame(time) {
  // Update uniforms
  const uniforms = new Float32Array([
    time * 0.001,  // time in seconds
    Math.sin(time * 0.001),
    Math.cos(time * 0.001),
    0.0,
  ]);

  device.queue.writeBuffer(uniformBuffer, 0, uniforms);

  // Render with updated uniforms
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass(renderPassDescriptor);
  // ... rendering commands ...
  pass.end();
  device.queue.submit([encoder.finish()]);

  requestAnimationFrame(frame);
}
```

This pattern works well for small updates (like transform matrices or time values) but becomes inefficient for large datasets updated frequently.

### Double Buffering / Ring Buffers

For larger updates or scenarios where GPU work takes multiple frames to complete, double buffering prevents write-after-read hazards:

```javascript
const bufferSize = 65536;
const numBuffers = 2;
let currentBufferIndex = 0;

// Create multiple buffers
const buffers = [];
for (let i = 0; i < numBuffers; i++) {
  buffers.push(device.createBuffer({
    label: `ring buffer ${i}`,
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }));
}

function frame() {
  // Use current buffer for GPU work
  const buffer = buffers[currentBufferIndex];

  // Update NEXT buffer while GPU processes current one
  const nextIndex = (currentBufferIndex + 1) % numBuffers;
  device.queue.writeBuffer(buffers[nextIndex], 0, nextFrameData);

  // Render with current buffer
  // ... rendering commands using 'buffer' ...

  // Advance to next buffer
  currentBufferIndex = nextIndex;

  requestAnimationFrame(frame);
}
```

This technique ensures the GPU always has data to process while the CPU prepares the next update, eliminating pipeline stalls.

### Avoiding Write-While-In-Use

A critical rule: never write to a buffer that's currently in use by the GPU. WebGPU's command queue model means that `queue.submit()` doesn't immediately execute commands—they're queued for asynchronous execution.

```javascript
// INCORRECT: Potential write-while-in-use
device.queue.writeBuffer(buffer, 0, data1);
device.queue.submit([renderCommands1]);

device.queue.writeBuffer(buffer, 0, data2);  // Might execute before GPU finishes!
device.queue.submit([renderCommands2]);

// CORRECT: Use different buffers or wait for completion
device.queue.writeBuffer(buffer1, 0, data1);
device.queue.submit([renderCommands1]);

device.queue.writeBuffer(buffer2, 0, data2);
device.queue.submit([renderCommands2]);
```

The safest approach is using separate buffers for concurrent work or ensuring operations complete before reuse through explicit synchronization (though WebGPU provides limited synchronization primitives, encouraging buffering strategies instead).

## Best Practices

### Buffer Pooling and Reuse

Creating and destroying buffers frequently causes performance overhead and memory fragmentation. Instead, maintain pools of reusable buffers:

```javascript
class BufferPool {
  constructor(device, size, usage) {
    this.device = device;
    this.size = size;
    this.usage = usage;
    this.available = [];
    this.inUse = new Set();
  }

  acquire() {
    let buffer;
    if (this.available.length > 0) {
      buffer = this.available.pop();
    } else {
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
    for (const buffer of this.available) {
      buffer.destroy();
    }
    for (const buffer of this.inUse) {
      buffer.destroy();
    }
  }
}

// Usage
const stagingPool = new BufferPool(
  device,
  65536,
  GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
);

const staging = stagingPool.acquire();
// ... use staging buffer ...
stagingPool.release(staging);
```

### Minimizing Allocations

Buffer allocation is expensive. Prefer these strategies:

1. **Preallocate at initialization**: Create all buffers needed during startup
2. **Use larger buffers with offsets**: Store multiple logical buffers in one physical buffer
3. **Grow buffers geometrically**: When expansion is necessary, double the size rather than adding fixed increments

```javascript
// Large buffer with sub-allocations
const largeBuffer = device.createBuffer({
  size: 1048576,  // 1 MiB
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// Write different uniforms at different offsets
device.queue.writeBuffer(largeBuffer, 0, cameraUniforms);
device.queue.writeBuffer(largeBuffer, 256, lightUniforms);
device.queue.writeBuffer(largeBuffer, 512, materialUniforms);

// Bind with offsets
bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{
    binding: 0,
    resource: {
      buffer: largeBuffer,
      offset: 0,
      size: 256,
    },
  }],
});
```

### Batching Updates

Group multiple small updates into larger operations:

```javascript
// INEFFICIENT: Many small writes
for (let i = 0; i < 100; i++) {
  device.queue.writeBuffer(buffer, i * 16, objects[i].transform);
}

// EFFICIENT: Single batched write
const batchedData = new Float32Array(100 * 4);
for (let i = 0; i < 100; i++) {
  batchedData.set(objects[i].transform, i * 4);
}
device.queue.writeBuffer(buffer, 0, batchedData);
```

Batching reduces command queue overhead and improves memory bandwidth utilization.

## Common Pitfalls

### Writing to In-Use Buffers

The most common mistake is updating a buffer while the GPU is still reading it:

```javascript
// WRONG
const buffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

function frame() {
  device.queue.writeBuffer(buffer, 0, frameUniforms);  // Update
  device.queue.submit([renderPass]);                    // Use
  // GPU hasn't necessarily processed the write before starting render!
}

// CORRECT: Use ring buffer
const buffers = [createBuffer(), createBuffer(), createBuffer()];
let index = 0;

function frame() {
  const buffer = buffers[index];
  device.queue.writeBuffer(buffer, 0, frameUniforms);
  device.queue.submit([renderPassUsingBuffer(buffer)]);
  index = (index + 1) % buffers.length;
}
```

### Memory Leaks

Buffers aren't automatically garbage collected—you must explicitly destroy them:

```javascript
// Create temporary buffer
const tempBuffer = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE });

// ... use buffer ...

// IMPORTANT: Destroy when done
tempBuffer.destroy();

// Further usage is invalid
// tempBuffer.getMappedRange();  // ERROR
```

Failing to destroy buffers leads to GPU memory exhaustion, especially in long-running applications or when creating many temporary buffers.

### Alignment Issues

Different data types and buffer usages have alignment requirements:

```javascript
// Uniform buffers require 256-byte aligned dynamic offsets
const uniformBuffer = device.createBuffer({
  size: 512,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// WRONG: Offset 100 is not 256-byte aligned
bindGroup = device.createBindGroup({
  entries: [{
    binding: 0,
    resource: { buffer: uniformBuffer, offset: 100, size: 64 },
  }],
});

// CORRECT: Use 256-byte aligned offsets
bindGroup = device.createBindGroup({
  entries: [{
    binding: 0,
    resource: { buffer: uniformBuffer, offset: 256, size: 64 },
  }],
});
```

Struct members in WGSL also have alignment requirements. For example, `vec3<f32>` has 16-byte alignment, so:

```wgsl
struct Uniforms {
  color: vec3<f32>,      // offset 0, takes 16 bytes due to alignment
  intensity: f32,        // offset 16
}
```

When creating typed arrays, account for this padding:

```javascript
// WRONG: Only 4 floats
const data = new Float32Array([1, 0, 0, /**/ 0.5]);

// CORRECT: Explicit padding for vec3
const data = new Float32Array([
  1, 0, 0, 0,   // vec3 color (4th element is padding)
  0.5, 0, 0, 0  // f32 intensity (3 padding floats)
]);
```

TypeGPU eliminates these issues by handling alignment automatically, but when using raw WebGPU, vigilance is required.

---

## Conclusion

Buffers form the foundation of WebGPU's memory model, enabling efficient data transfer between CPU and GPU. Understanding buffer types, creation patterns, and memory visibility is essential for building performant WebGPU applications.

WebGPU provides low-level control through explicit buffer creation, usage flags, and mapping operations, enabling optimization for specific use cases. TypeGPU builds upon this foundation with type-safe abstractions that reduce boilerplate and prevent common errors while maintaining access to the underlying WebGPU primitives when needed.

By following best practices—pooling buffers, batching updates, avoiding write-while-in-use, and respecting alignment requirements—developers can achieve optimal performance while maintaining code correctness. The explicit memory model may seem complex initially, but it provides the control necessary for high-performance graphics and compute applications that leverage the full power of modern GPUs.
