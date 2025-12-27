---
title: Performance Optimization
sidebar:
  order: 40
---

## Overview

WebGPU performance optimization requires understanding GPU architecture fundamentals: massive parallelism, memory bandwidth limitations, and CPU-GPU synchronization overhead.

:::note[Core Principle]
Minimize work, minimize transfers, maximize parallelism. GPUs excel at throughput, not latency.
:::

## GPU Architecture

### Memory Hierarchy

| Level | Speed | Scope |
|-------|-------|-------|
| Registers | Fastest | Per-thread |
| Shared/Workgroup Memory | Fast | Per-workgroup |
| L1/L2 Cache | Medium | Automatic |
| Global Memory | Slowest | All threads |

### Key Performance Factors

- **Parallelism**: GPUs run thousands of threads simultaneously
- **Latency hiding**: GPUs switch thread groups when one stalls on memory
- **Divergence**: Threads in same group taking different paths causes serialization
- **Memory bandwidth**: Often the primary bottleneck

## Buffer Management

### Buffer Pooling

:::tip[Preallocate and Reuse]
Buffer creation is expensive. Create once, reuse across frames:

```javascript title="Buffer pool pattern"
class BufferPool {
  constructor(device, size, usage) {
    this.available = [];
    this.device = device;
    this.size = size;
    this.usage = usage;
  }

  acquire() {
    return this.available.pop() || this.device.createBuffer({
      size: this.size,
      usage: this.usage,
    });
  }

  release(buffer) {
    this.available.push(buffer);
  }
}
```
:::

### Interleaved Vertex Attributes

:::tip[Single Buffer > Multiple Buffers]
Interleave all attributes in one buffer:

```javascript title="Interleaved vertex layout"
// Layout: [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, u, v] × N vertices
buffers: [{
  arrayStride: 32,  // 3+3+2 floats × 4 bytes
  attributes: [
    { shaderLocation: 0, offset: 0, format: "float32x3" },   // position
    { shaderLocation: 1, offset: 12, format: "float32x3" },  // normal
    { shaderLocation: 2, offset: 24, format: "float32x2" },  // uv
  ],
}]
```

Benefits:
- Fewer `setVertexBuffer()` calls
- Better cache utilization (all vertex data in one cache line)
- Up to 600% performance improvement for many models
:::

### Minimizing Buffer Updates

```javascript title="Batch uniform updates"
// Group uniforms by update frequency
const frameUniforms = device.createBuffer({ size: 256, ... });  // Per-frame
const materialUniforms = device.createBuffer({ size: 1024, ... });  // Per-material
const objectUniforms = device.createBuffer({ size: 65536, ... });  // Per-object with offsets

// Single write per group instead of many small writes
device.queue.writeBuffer(frameUniforms, 0, frameData);
```

:::caution[Avoid Per-Frame Allocations]
Creating new buffers every frame triggers garbage collection and wastes GPU resources. Update existing buffers instead.
:::

## Pipeline Optimization

### Pipeline Caching

```javascript title="Cache pipeline objects"
const pipelineCache = new Map();

function getPipeline(config) {
  const key = JSON.stringify(config);
  if (!pipelineCache.has(key)) {
    pipelineCache.set(key, device.createRenderPipeline(config));
  }
  return pipelineCache.get(key);
}
```

### Async Pipeline Creation

```javascript title="Background pipeline compilation"
// During initialization (prevents frame drops)
const pipelines = await Promise.all([
  device.createRenderPipelineAsync(desc1),
  device.createRenderPipelineAsync(desc2),
  device.createComputePipelineAsync(desc3),
]);
```

### Reducing State Changes

:::tip[Sort by Pipeline]
Group draw calls to minimize state changes:

1. **Pipeline** (most expensive to switch)
2. **Bind groups** (textures, uniforms)
3. **Vertex buffers**
4. **Distance** (for transparency sorting)
:::

## Memory Optimization

### Texture Compression

| Format | Platform | Compression |
|--------|----------|-------------|
| BC (DXT) | Desktop | ~75% smaller |
| ETC2/EAC | Mobile | ~75% smaller |
| ASTC | Mobile/Some Desktop | Variable |

```javascript title="Check compression support"
const hasBCTextures = adapter.features.has("texture-compression-bc");
const hasETC2 = adapter.features.has("texture-compression-etc2");
```

### Mipmaps

:::tip[Always Generate Mipmaps]
Mipmaps improve both quality and performance:
- Prevent aliasing at small sizes
- Improve cache efficiency
- GPU auto-selects appropriate level

```javascript
const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;
```
:::

### Staging Buffers

```javascript title="Ring buffer for streaming data"
const RING_SIZE = 3;
const stagingBuffers = Array.from({ length: RING_SIZE }, () =>
  device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
  })
);

let frameIndex = 0;
function getNextStagingBuffer() {
  return stagingBuffers[frameIndex++ % RING_SIZE];
}
```

## Compute Optimization

### Workgroup Size

:::tip[Default: 64 Threads]
Use 64 as a starting point:
- Aligns with common warp/wavefront sizes (32-64)
- Provides good latency hiding
- Divisible by common hardware sizes

```wgsl
@compute @workgroup_size(64)     // 1D data
@compute @workgroup_size(8, 8)   // 2D data
```
:::

### Memory Coalescing

```wgsl title="Coalesced vs scattered access"
// Good: Adjacent threads access adjacent memory
let value = input[global_id.x];

// Bad: Scattered access
let value = input[global_id.x * stride];
```

### Shared Memory

```wgsl title="Use workgroup memory for data reuse"
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256)
fn compute(@builtin(local_invocation_id) lid: vec3u) {
  // Load from global to shared
  cache[lid.x] = globalData[gid.x];
  workgroupBarrier();

  // Multiple reads from fast shared memory
  var sum = 0.0;
  for (var i = 0u; i < 256u; i++) {
    sum += cache[i];
  }
}
```

## Instancing

```javascript title="Render thousands with one draw call"
// Per-instance buffer
const instanceBuffer = device.createBuffer({
  size: instanceCount * 64,  // 4x4 matrix per instance
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

// Vertex layout with instance step mode
buffers: [
  { arrayStride: 12, stepMode: "vertex", ... },
  { arrayStride: 64, stepMode: "instance", ... },
]

// Single draw call for all instances
passEncoder.draw(vertexCount, instanceCount);
```

## Shader Optimization

### Avoid Branching

```wgsl title="Replace branches with math"
// Bad: Causes divergence
if (value > threshold) {
  result = a;
} else {
  result = b;
}

// Good: No divergence
result = mix(b, a, step(threshold, value));
```

### Use Built-in Functions

Built-ins like `dot()`, `normalize()`, `mix()`, `smoothstep()` map directly to hardware instructions and are significantly faster than manual implementations.

## Profiling

### Timestamp Queries

```javascript title="Measure GPU execution time"
const querySet = device.createQuerySet({
  type: "timestamp",
  count: 2,
});

const resolveBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});

// In render pass
passEncoder.writeTimestamp(querySet, 0);
// ... render commands ...
passEncoder.writeTimestamp(querySet, 1);

// Resolve and read
encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
```

### Browser DevTools

- **Chrome**: Performance panel, about://gpu
- **Firefox**: about:support, Performance profiler
- **Platform tools**: PIX (Windows), Instruments (macOS)

## Common Bottlenecks

<details>
<summary>**Overdraw**</summary>

Same pixel rendered multiple times.

**Fix**: Sort opaque geometry front-to-back, minimize transparent overlaps.

</details>

<details>
<summary>**Memory Bandwidth**</summary>

Large transfers, uncompressed textures, scattered access.

**Fix**: Compress textures, use mipmaps, coalesce memory access.

</details>

<details>
<summary>**CPU Overhead**</summary>

Too many draw calls, excessive API calls.

**Fix**: Batch geometry, use instancing, minimize state changes.

</details>

<details>
<summary>**Shader Complexity**</summary>

Heavy fragment shaders on high-resolution targets.

**Fix**: Move work to vertex shaders, reduce texture samples, simplify distant objects.

</details>

## Quick Checklist

| Area | Optimization |
|------|--------------|
| **Buffers** | Pool and reuse; interleave vertices; batch updates |
| **Pipelines** | Cache objects; use async creation; sort by pipeline |
| **Memory** | Compress textures; generate mipmaps; use staging buffers |
| **Draw Calls** | Use instancing; batch by material; minimize state changes |
| **Shaders** | Avoid branching; use built-ins; move work to vertex stage |
| **Compute** | Use workgroup size 64; coalesce access; leverage shared memory |
| **Profiling** | Measure before optimizing; use timestamp queries |

## Resources

:::note[Official Documentation]
- [WebGPU Best Practices](https://toji.dev/webgpu-best-practices/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [Chrome GPU Debugging](https://developer.chrome.com/docs/devtools/rendering/)
:::
