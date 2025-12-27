---
title: Workgroup Variables and Shared Memory
sidebar:
  order: 40
---

## Overview

Workgroup variables provide fast on-chip shared memory accessible by all threads in a workgroup. This memory is 10-20× faster than global storage buffers, enabling efficient parallel algorithms.

:::note[Memory Characteristics]
- **Scope**: Isolated per workgroup—each workgroup gets its own copy
- **Lifetime**: Exists only during workgroup execution
- **Speed**: ~20-40 cycles vs 400+ for uncached global memory
- **Size**: Minimum 16KB guaranteed (16,384 bytes)
:::

## When to Use Workgroup Memory

| Use Case | Benefit |
|----------|---------|
| Data reuse | Load once, read many times |
| Intermediate results | Share values between threads |
| Parallel reduction | Sum/max/min across threads |
| Tiled algorithms | Matrix multiplication, convolutions |
| Cooperative caching | Cache global data for neighborhood access |

## WGSL Declaration

```wgsl title="Workgroup variable declarations"
var<workgroup> sharedData: array<f32, 256>;
var<workgroup> tileCache: array<vec4f, 64>;
var<workgroup> hitCount: atomic<u32>;
```

:::caution[Compute Shaders Only]
Workgroup variables are only available in compute shaders—vertex and fragment shaders don't have workgroups.
:::

### Size Planning

```wgsl title="Memory budget calculation"
var<workgroup> cache: array<f32, 256>;        // 256 × 4 = 1,024 bytes
var<workgroup> positions: array<vec4f, 128>;  // 128 × 16 = 2,048 bytes
var<workgroup> indices: array<u32, 512>;      // 512 × 4 = 2,048 bytes
// Total: 5,120 bytes (within 16KB limit)
```

Query device limits:

```javascript title="Check available workgroup storage"
console.log("Max storage:", device.limits.maxComputeWorkgroupStorageSize);
```

## TypeGPU Declaration

```typescript title="TypeGPU workgroup variables"
import tgpu from "typegpu";
import * as d from "typegpu/data";

const sharedData = tgpu.workgroupVar(d.arrayOf(d.f32, 256));
const sharedVectors = tgpu.workgroupVar(d.arrayOf(d.vec4f, 128));

const ParticleData = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
});
const sharedParticles = tgpu.workgroupVar(d.arrayOf(ParticleData, 64));
```

## Synchronization

### workgroupBarrier()

All threads must reach the barrier before any can proceed. Also ensures memory visibility.

```wgsl title="Barrier usage" {7,12}
var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
  // Phase 1: Each thread writes
  sharedData[idx] = f32(idx) * 2.0;

  workgroupBarrier();  // Wait for all writes

  // Phase 2: Safe to read other threads' data
  let neighbor = sharedData[(idx + 1u) % 256u];
}
```

:::danger[Always Synchronize]
Without barriers, threads may read before others have written:

```wgsl
// BUG: Race condition!
sharedData[idx] = f32(idx);
let value = sharedData[(idx + 1u) % 256u];  // May read garbage!
```
:::

### storageBarrier()

Synchronizes storage buffer and atomic operations (but NOT execution):

```wgsl title="Storage barrier"
atomicAdd(&globalCounter, 1u);
storageBarrier();  // Ensure atomic completes
output[gid.x] = f32(value);
```

## Common Patterns

<details>
<summary>**Parallel Reduction (Sum)**</summary>

```wgsl
var<workgroup> reductionBuffer: array<f32, 256>;

@compute @workgroup_size(256)
fn parallelSum(
  @builtin(local_invocation_index) localIdx: u32,
  @builtin(global_invocation_id) globalId: vec3u,
  @builtin(workgroup_id) workgroupId: vec3u
) {
  // Load into shared memory
  reductionBuffer[localIdx] = input[globalId.x];
  workgroupBarrier();

  // Tree reduction: 8 steps for 256 values
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (localIdx < stride) {
      reductionBuffer[localIdx] += reductionBuffer[localIdx + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 writes final sum
  if (localIdx == 0u) {
    output[workgroupId.x] = reductionBuffer[0];
  }
}
```

</details>

<details>
<summary>**Tiled Matrix Multiplication**</summary>

```wgsl
var<workgroup> tileA: array<f32, 16 * 16>;
var<workgroup> tileB: array<f32, 16 * 16>;

@compute @workgroup_size(16, 16)
fn tiledMatMul(
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(global_invocation_id) globalId: vec3u
) {
  let row = globalId.y;
  let col = globalId.x;
  var sum = 0.0;
  let numTiles = (dims.z + 15u) / 16u;

  for (var t = 0u; t < numTiles; t++) {
    // Load tiles cooperatively
    tileA[localId.y * 16u + localId.x] = matrixA[...];
    tileB[localId.y * 16u + localId.x] = matrixB[...];
    workgroupBarrier();

    // Compute using cached tiles
    for (var k = 0u; k < 16u; k++) {
      sum += tileA[localId.y * 16u + k] * tileB[k * 16u + localId.x];
    }
    workgroupBarrier();
  }

  matrixC[row * dims.y + col] = sum;
}
```

Achieves 10-20× speedup by reducing global memory access.

</details>

<details>
<summary>**Image Blur with Caching**</summary>

```wgsl
var<workgroup> tileCache: array<vec4f, 18 * 18>;  // 16×16 + 1-pixel border

@compute @workgroup_size(16, 16)
fn imageBlur(
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(global_invocation_id) globalId: vec3u
) {
  // Load tile with border into shared memory
  let tileX = localId.x + 1u;
  let tileY = localId.y + 1u;
  tileCache[tileY * 18u + tileX] = textureLoad(inputTexture, globalId.xy, 0);

  // Edge threads load border pixels
  // ...

  workgroupBarrier();

  // Apply 3×3 blur using cached data
  var sum = vec4f(0.0);
  for (var dy = -1i; dy <= 1i; dy++) {
    for (var dx = -1i; dx <= 1i; dx++) {
      sum += tileCache[(i32(tileY) + dy) * 18 + i32(tileX) + dx];
    }
  }
  textureStore(outputTexture, globalId.xy, sum / 9.0);
}
```

</details>

<details>
<summary>**Histogram with Local Atomics**</summary>

```wgsl
var<workgroup> localHistogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn computeHistogram(@builtin(local_invocation_index) localIdx: u32) {
  // Initialize local histogram
  atomicStore(&localHistogram[localIdx], 0u);
  workgroupBarrier();

  // Process pixels
  while (pixelIdx < numPixels) {
    let value = imageData[pixelIdx] & 0xFFu;
    atomicAdd(&localHistogram[value], 1u);
    pixelIdx += 256u;
  }
  workgroupBarrier();

  // Merge into global histogram
  atomicAdd(&globalHistogram[localIdx], atomicLoad(&localHistogram[localIdx]));
}
```

</details>

## Performance Considerations

### Bank Conflicts

Workgroup memory uses banks—simultaneous access to the same bank serializes:

```wgsl title="Avoid bank conflicts"
// BAD: All threads access same element
let value = sharedData[0];  // Serialized!

// GOOD: Each thread accesses different element
let value = sharedData[idx];  // Parallel!
```

:::tip[Bank Layout]
Typical GPUs have 32 banks with 4-byte width. Consecutive elements map to consecutive banks. Avoid strides of 32.
:::

### Optimal Access Patterns

| Pattern | Performance |
|---------|-------------|
| Sequential access | Fast |
| Broadcast (same value) | Fast |
| Random access | Medium |
| Strided (every 32nd) | Slow (conflicts) |

### Workgroup Size

:::tip[Recommended Sizes]
- **General**: 64-256 threads
- **1D data**: `@workgroup_size(256)`
- **2D images**: `@workgroup_size(16, 16)` or `@workgroup_size(8, 8)`
- **3D volumes**: `@workgroup_size(8, 8, 4)`
:::

## Platform Limits

| Platform | Typical Limit |
|----------|---------------|
| Desktop NVIDIA | 48KB |
| Desktop AMD/Intel | 32KB |
| Mobile | 16-32KB |
| WebGPU Minimum | 16KB |

## Common Pitfalls

:::danger[Missing Barrier]
```wgsl
sharedData[idx] = f32(idx);
// Missing workgroupBarrier()!
let value = sharedData[(idx + 1u) % 256u];  // Race condition!
```
:::

:::danger[Race Condition with Non-Atomic]
```wgsl
// WRONG: Use atomic<u32> for shared counters
var<workgroup> counter: u32;
counter += 1u;  // Race condition!

// CORRECT
var<workgroup> counter: atomic<u32>;
atomicAdd(&counter, 1u);
```
:::

:::danger[Exceeding Size Limit]
```wgsl
// May fail on some devices (32KB)
var<workgroup> huge: array<vec4f, 2048>;  // 2048 × 16 = 32KB
```
:::

:::caution[Reading Uninitialized Data]
Workgroup memory starts uninitialized. Always write before reading:

```wgsl
// BUG: Only thread 0 writes, others read garbage
if (idx == 0u) {
  sharedData[0] = 42.0;
}
workgroupBarrier();
let value = sharedData[idx];  // data[1..255] uninitialized!
```
:::

## Resources

:::note[Official Documentation]
- [WGSL Memory Model](https://gpuweb.github.io/gpuweb/wgsl/#memory-model)
- [WebGPU Compute](https://gpuweb.github.io/gpuweb/#compute-pipeline)
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/)
:::
