# Workgroup Variables and Shared Memory

Workgroup variables represent one of the most powerful features in GPU compute programming, enabling efficient cooperation between parallel threads executing within the same workgroup. While storage buffers provide large-scale data access across an entire shader dispatch and private memory serves individual threads, workgroup memory occupies a unique middle ground—fast on-chip shared memory accessible by all threads in a workgroup, perfect for algorithms requiring inter-thread communication and data sharing.

In modern GPU architectures, workgroup memory (also called "shared memory" or "local memory") resides in specialized on-chip SRAM located close to the compute units. This proximity makes workgroup memory significantly faster than global VRAM access, with latency typically 10-20x lower than storage buffer reads. However, this speed comes with constraints: workgroup memory is limited in size (typically 16-32KB per workgroup) and requires careful synchronization to avoid race conditions.

Understanding workgroup variables is essential for implementing high-performance GPU algorithms. Techniques like parallel reduction, tiled matrix multiplication, image processing with neighborhood operations, and histogram computation all rely heavily on workgroup shared memory to achieve optimal performance. Without workgroup memory, these algorithms would be forced to repeatedly access slow global memory, resulting in performance degradation of 10x or more.

This guide explores workgroup variables in depth, covering their declaration and usage in both raw WGSL and TypeGPU, synchronization primitives, memory visibility guarantees, common algorithmic patterns, performance optimization strategies, and pitfalls to avoid.

## What is Workgroup Shared Memory?

Workgroup shared memory is a dedicated memory region allocated per workgroup in compute shaders. Unlike storage buffers that persist across shader dispatches and are shared globally, workgroup memory exists only for the duration of a single workgroup's execution and is isolated between different workgroups.

### Memory Characteristics

Workgroup memory has several defining characteristics that make it both powerful and unique:

**Scope and Lifetime**: Workgroup memory is allocated when a workgroup begins execution and deallocated when the workgroup completes. Each workgroup gets its own independent copy of all workgroup variables. If you dispatch 1,000 workgroups, each gets its own separate 16KB of workgroup memory—workgroups cannot see or interfere with each other's workgroup variables.

**Speed**: Workgroup memory typically resides in on-chip SRAM, making it 10-20x faster than storage buffer access. On modern GPUs, workgroup memory access latency is around 20-40 cycles compared to 400+ cycles for uncached global memory. This speed advantage makes workgroup memory ideal for data that's accessed repeatedly by multiple threads.

**Size Constraints**: The WebGPU specification guarantees a minimum of 16,384 bytes (16KB) of workgroup memory per shader. Some GPUs provide more (32KB or even 64KB), but portable code should assume only 16KB is available. You can query device limits to determine the actual available size.

**Visibility**: All threads (invocations) within a workgroup can read and write to the same workgroup variables. This shared access is both a powerful feature and a potential source of race conditions—proper synchronization is essential.

### When to Use Workgroup Memory

Workgroup shared memory shines in specific scenarios:

**Data Reuse**: When multiple threads need to access the same data, loading it once into workgroup memory and reusing it is much faster than having each thread independently access slow global memory.

**Intermediate Results**: Algorithms that produce intermediate values needed by other threads in the workgroup benefit from storing those values in shared memory.

**Communication and Cooperation**: Parallel reduction (summing values across threads), scan operations (prefix sum), and sorting all require threads to exchange data—workgroup memory provides the communication channel.

**Tiling and Blocking**: Breaking large problems into smaller tiles that fit in workgroup memory enables better cache utilization and reduced memory bandwidth consumption.

**Caching Global Data**: When accessing global memory with poor patterns (strided access, random access), caching data in workgroup memory can dramatically improve performance.

## WGSL Workgroup Variables

In WGSL, workgroup variables are declared at module scope using the `var<workgroup>` syntax. This explicit address space annotation tells the compiler to allocate the variable in workgroup shared memory.

### Declaration Syntax

Workgroup variables must be declared at module scope (outside any function) with the `<workgroup>` address space qualifier:

```wgsl
// Basic workgroup variable declarations
var<workgroup> sharedData: array<f32, 256>;
var<workgroup> tileCache: array<vec4f, 16, 16>;
var<workgroup> localSum: f32;
var<workgroup> hitCount: atomic<u32>;

// Workgroup variables can be arrays, scalars, vectors, or structs
var<workgroup> scratchpad: array<u32, 512>;
var<workgroup> reductionBuffer: array<f32, 1024>;
var<workgroup> sharedMatrix: array<mat4x4f, 8>;
```

### Scope and Lifetime

Workgroup variables are only valid in compute shaders—vertex and fragment shaders don't have the concept of workgroups, so workgroup memory is not available in those shader stages.

```wgsl
// Valid: Compute shader with workgroup variable
var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
    // Can access sharedData here
    sharedData[idx] = f32(idx);
}

// Invalid: Cannot use workgroup variables in fragment shaders
@fragment
fn fragmentMain() -> @location(0) vec4f {
    // var<workgroup> data: f32;  // COMPILATION ERROR!
    return vec4f(1.0);
}
```

The lifetime of workgroup variables spans the entire execution of a workgroup. When a workgroup begins executing, workgroup memory is allocated and variables are uninitialized (contain undefined values). All writes to workgroup variables must happen before reads to avoid accessing uninitialized data. When the workgroup completes, the memory is deallocated.

### Size Limits and Planning

The WGSL specification guarantees a minimum of 16,384 bytes (16KB) of workgroup memory. When designing data structures, you must carefully budget this limited resource:

```wgsl
// Calculate memory usage carefully
var<workgroup> cache: array<f32, 256>;        // 256 * 4 = 1,024 bytes
var<workgroup> positions: array<vec4f, 128>;  // 128 * 16 = 2,048 bytes
var<workgroup> indices: array<u32, 512>;      // 512 * 4 = 2,048 bytes
// Total: 5,120 bytes (well within 16KB limit)

// This would exceed the limit on some devices:
// var<workgroup> huge: array<vec4f, 2048>;   // 2048 * 16 = 32,768 bytes - TOO LARGE!
```

You can query the actual limit on a specific device:

```typescript
// JavaScript/TypeScript
const limits = device.limits;
console.log('Max compute workgroup storage size:', limits.maxComputeWorkgroupStorageSize);
```

### Complete WGSL Example

Here's a complete example showing workgroup variable usage in a parallel sum operation:

```wgsl
var<workgroup> sharedSums: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn parallelSum(
    @builtin(local_invocation_index) localIdx: u32,
    @builtin(global_invocation_id) globalId: vec3u,
    @builtin(workgroup_id) workgroupId: vec3u
) {
    // Each thread loads one value from global memory into shared memory
    sharedSums[localIdx] = input[globalId.x];

    // Synchronize to ensure all threads have finished writing
    workgroupBarrier();

    // Parallel reduction: sum values in a tree pattern
    var stride = 128u;
    while (stride > 0u) {
        if (localIdx < stride) {
            sharedSums[localIdx] += sharedSums[localIdx + stride];
        }
        workgroupBarrier();  // Synchronize after each reduction step
        stride = stride / 2u;
    }

    // Thread 0 writes the final sum for this workgroup
    if (localIdx == 0u) {
        output[workgroupId.x] = sharedSums[0];
    }
}
```

## TypeGPU Workgroup Variables

TypeGPU provides a type-safe, ergonomic API for working with workgroup variables through `tgpu.workgroupVar()`. This approach maintains full TypeScript type checking while generating correct WGSL code.

### Declaration with tgpu.workgroupVar()

TypeGPU workgroup variables are created using the `tgpu.workgroupVar()` function with a data schema describing the variable's type:

```typescript
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

// Scalar workgroup variable
const sharedCounter = tgpu.workgroupVar(d.u32);

// Array workgroup variable
const sharedData = tgpu.workgroupVar(d.arrayOf(d.f32, 256));

// Vector array
const sharedVectors = tgpu.workgroupVar(d.arrayOf(d.vec4f, 128));

// Struct array
const ParticleData = d.struct({
    position: d.vec3f,
    velocity: d.vec3f,
});
const sharedParticles = tgpu.workgroupVar(d.arrayOf(ParticleData, 64));
```

### Using in TGSL Functions

Workgroup variables can be accessed within TGSL function bodies just like storage buffers:

```typescript
import { tgsl } from 'typegpu/tgsl';

const sharedSums = tgpu.workgroupVar(d.arrayOf(d.f32, 256));
const inputBuffer = tgpu.storage(d.arrayOf(d.f32));
const outputBuffer = tgpu.storage(d.arrayOf(d.f32), 'readwrite');

const parallelSumKernel = tgsl
    .workgroupSize(256)
    .kernel([], () => {
        const localIdx = builtin.localInvocationIndex;
        const globalId = builtin.globalInvocationId.x;
        const workgroupId = builtin.workgroupId.x;

        // Load data into shared memory
        sharedSums[localIdx] = inputBuffer[globalId];

        // Synchronization (discussed in next section)
        workgroupBarrier();

        // Reduction logic
        let stride = 128;
        while (stride > 0) {
            if (localIdx < stride) {
                sharedSums[localIdx] = sharedSums[localIdx] + sharedSums[localIdx + stride];
            }
            workgroupBarrier();
            stride = stride / 2;
        }

        // Write result
        if (localIdx === 0) {
            outputBuffer[workgroupId] = sharedSums[0];
        }
    });
```

### Type Safety Benefits

TypeGPU's type system prevents common errors at compile time:

```typescript
// Type error: wrong element type
const sharedData = tgpu.workgroupVar(d.arrayOf(d.f32, 256));
// sharedData[0] = d.vec3f(1, 2, 3);  // TypeScript error!

// Correct: matching types
sharedData[0] = 3.14;

// Type error: index out of bounds (if TypeGPU supports this check)
// const data = tgpu.workgroupVar(d.arrayOf(d.f32, 10));
// data[100] = 1.0;  // Potential error or warning
```

## Synchronization Primitives

Workgroup memory is shared between threads, which creates the possibility of race conditions. WGSL provides synchronization primitives to coordinate thread execution and ensure memory consistency.

### workgroupBarrier()

The `workgroupBarrier()` function is the primary synchronization primitive. It creates an execution barrier that all threads in the workgroup must reach before any can proceed. It also provides memory visibility guarantees, ensuring that all writes to workgroup memory before the barrier are visible to all threads after the barrier.

```wgsl
var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
    // Phase 1: Each thread writes to its slot
    sharedData[idx] = f32(idx) * 2.0;

    // BARRIER: Wait for all writes to complete
    workgroupBarrier();

    // Phase 2: Now safe to read from any slot
    let neighbor = sharedData[(idx + 1u) % 256u];
    let sum = sharedData[idx] + neighbor;
}
```

**What workgroupBarrier() does:**

1. **Execution Barrier**: Halts each thread's execution at the barrier until all threads in the workgroup reach it
2. **Memory Fence**: Ensures all workgroup memory writes before the barrier are completed and visible
3. **Synchronization Point**: Establishes a "happens-before" relationship—all operations before the barrier are guaranteed to complete before any operations after the barrier begin

**When to use workgroupBarrier():**

- After writing to workgroup memory but before reading data written by other threads
- Between phases of an algorithm that depend on results from the previous phase
- Before and after critical sections where threads need a consistent view of shared data

### storageBarrier()

While `workgroupBarrier()` synchronizes workgroup memory, `storageBarrier()` synchronizes storage buffer access and atomic operations:

```wgsl
@group(0) @binding(0) var<storage, read_write> globalCounter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
    // Perform atomic operation
    let value = atomicAdd(&globalCounter, 1u);

    // Storage barrier ensures atomic operation completes
    storageBarrier();

    // Use the value
    output[gid.x] = f32(value);
}
```

**Difference from workgroupBarrier():**

- `workgroupBarrier()`: Synchronizes workgroup memory and execution within a workgroup
- `storageBarrier()`: Synchronizes storage buffer memory and atomic operations, but does NOT provide execution barrier

In practice, you often need both barriers for compute shaders that mix workgroup and storage buffer operations.

### Memory Visibility and Barriers

Barriers enforce memory visibility through the acquire-release memory model. Operations before a barrier "release" their writes, making them visible. Operations after a barrier "acquire" those writes, seeing the updated values.

```wgsl
var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
    // Thread 0 writes a value
    if (idx == 0u) {
        sharedData[100] = 42.0;
    }

    // WITHOUT barrier, this is a race condition:
    // let value = sharedData[100];  // Might see old value or garbage!

    // WITH barrier:
    workgroupBarrier();
    let value = sharedData[100];  // Guaranteed to see 42.0
}
```

## Common Patterns

Workgroup shared memory enables several fundamental parallel computing patterns. Understanding these patterns is key to writing efficient GPU code.

### Shared Reduction

Reduction combines many values into a single result (sum, max, min, etc.) using a tree-based parallel algorithm:

```wgsl
var<workgroup> reductionBuffer: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn parallelSum(
    @builtin(local_invocation_index) localIdx: u32,
    @builtin(global_invocation_id) globalId: vec3u,
    @builtin(workgroup_id) workgroupId: vec3u
) {
    // Load input into shared memory
    reductionBuffer[localIdx] = input[globalId.x];
    workgroupBarrier();

    // Tree reduction: each iteration halves active threads
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (localIdx < stride) {
            reductionBuffer[localIdx] += reductionBuffer[localIdx + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes final result
    if (localIdx == 0u) {
        output[workgroupId.x] = reductionBuffer[0];
    }
}
```

This pattern reduces 256 values to 1 using only 8 steps (log₂(256)), far more efficient than sequential summation.

### Tiled Matrix Multiplication

Matrix multiplication benefits enormously from tiling—loading submatrices into workgroup memory to reduce global memory traffic:

```wgsl
var<workgroup> tileA: array<f32, 16 * 16>;
var<workgroup> tileB: array<f32, 16 * 16>;

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3u;  // M, N, K dimensions

@compute @workgroup_size(16, 16)
fn tiledMatMul(
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(global_invocation_id) globalId: vec3u
) {
    let row = globalId.y;
    let col = globalId.x;
    let localRow = localId.y;
    let localCol = localId.x;

    var sum = 0.0;
    let numTiles = (dims.z + 15u) / 16u;  // K dimension, rounded up

    // Process matrix in 16x16 tiles
    for (var t = 0u; t < numTiles; t++) {
        // Cooperatively load tile of A into shared memory
        let aRow = row;
        let aCol = t * 16u + localCol;
        if (aRow < dims.x && aCol < dims.z) {
            tileA[localRow * 16u + localCol] = matrixA[aRow * dims.z + aCol];
        } else {
            tileA[localRow * 16u + localCol] = 0.0;
        }

        // Cooperatively load tile of B into shared memory
        let bRow = t * 16u + localRow;
        let bCol = col;
        if (bRow < dims.z && bCol < dims.y) {
            tileB[localRow * 16u + localCol] = matrixB[bRow * dims.y + bCol];
        } else {
            tileB[localRow * 16u + localCol] = 0.0;
        }

        // Wait for both tiles to be loaded
        workgroupBarrier();

        // Compute partial dot product using cached tiles
        for (var k = 0u; k < 16u; k++) {
            sum += tileA[localRow * 16u + k] * tileB[k * 16u + localCol];
        }

        // Wait before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < dims.x && col < dims.y) {
        matrixC[row * dims.y + col] = sum;
    }
}
```

This approach can achieve 10-20x speedup by reducing global memory accesses.

### Data Caching

Loading frequently-accessed data into workgroup memory avoids redundant global memory reads:

```wgsl
var<workgroup> tileCache: array<vec4f, 18 * 18>;  // 16x16 + 1-pixel border

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn imageBlur(
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(global_invocation_id) globalId: vec3u
) {
    let texSize = vec2u(textureDimensions(inputTexture));

    // Load tile with 1-pixel border into shared memory
    let tileX = localId.x + 1u;
    let tileY = localId.y + 1u;

    // Central pixel
    if (all(globalId.xy < texSize)) {
        tileCache[tileY * 18u + tileX] = textureLoad(inputTexture, globalId.xy, 0);
    }

    // Border pixels (edge threads load extra pixels)
    if (localId.x == 0u && globalId.x > 0u) {
        tileCache[tileY * 18u] = textureLoad(inputTexture, globalId.xy - vec2u(1u, 0u), 0);
    }
    if (localId.x == 15u && globalId.x < texSize.x - 1u) {
        tileCache[tileY * 18u + 17u] = textureLoad(inputTexture, globalId.xy + vec2u(1u, 0u), 0);
    }
    if (localId.y == 0u && globalId.y > 0u) {
        tileCache[tileX] = textureLoad(inputTexture, globalId.xy - vec2u(0u, 1u), 0);
    }
    if (localId.y == 15u && globalId.y < texSize.y - 1u) {
        tileCache[17u * 18u + tileX] = textureLoad(inputTexture, globalId.xy + vec2u(0u, 1u), 0);
    }

    workgroupBarrier();

    // Apply 3x3 box blur using cached data
    var sum = vec4f(0.0);
    for (var dy = -1i; dy <= 1i; dy++) {
        for (var dx = -1i; dx <= 1i; dx++) {
            let tx = i32(tileX) + dx;
            let ty = i32(tileY) + dy;
            sum += tileCache[ty * 18 + tx];
        }
    }

    if (all(globalId.xy < texSize)) {
        textureStore(outputTexture, globalId.xy, sum / 9.0);
    }
}
```

### Histogram Building

Building histograms with atomic operations in workgroup memory, then writing to global memory:

```wgsl
var<workgroup> localHistogram: array<atomic<u32>, 256>;

@group(0) @binding(0) var<storage, read> imageData: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHistogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn computeHistogram(
    @builtin(local_invocation_index) localIdx: u32,
    @builtin(global_invocation_id) globalId: vec3u
) {
    // Initialize local histogram
    atomicStore(&localHistogram[localIdx], 0u);
    workgroupBarrier();

    // Process pixels, updating local histogram
    let numPixels = arrayLength(&imageData);
    var pixelIdx = globalId.x;
    while (pixelIdx < numPixels) {
        let pixelValue = imageData[pixelIdx] & 0xFFu;
        atomicAdd(&localHistogram[pixelValue], 1u);
        pixelIdx += 256u;
    }

    workgroupBarrier();

    // Merge local histogram into global histogram
    let localCount = atomicLoad(&localHistogram[localIdx]);
    atomicAdd(&globalHistogram[localIdx], localCount);
}
```

## Size Limits and Querying

Different GPUs have different workgroup memory limits. The WebGPU spec guarantees 16KB minimum, but devices may offer more.

### Maximum Workgroup Storage Size

Query device limits to determine available workgroup memory:

```typescript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter!.requestDevice();

const limits = device.limits;
console.log('Max workgroup storage size:', limits.maxComputeWorkgroupStorageSize);
console.log('Max workgroup size X:', limits.maxComputeWorkgroupSizeX);
console.log('Max workgroup size Y:', limits.maxComputeWorkgroupSizeY);
console.log('Max workgroup size Z:', limits.maxComputeWorkgroupSizeZ);
console.log('Max invocations per workgroup:', limits.maxComputeInvocationsPerWorkgroup);
```

### Platform Differences

Different platforms expose different limits:

| Platform | Typical Limit | Notes |
|----------|---------------|-------|
| Desktop (NVIDIA) | 49,152 bytes (48KB) | High-end GPUs |
| Desktop (AMD) | 32,768 bytes (32KB) | Standard |
| Desktop (Intel) | 32,768 bytes (32KB) | Integrated GPUs |
| Mobile (iOS) | 16,384 bytes (16KB) | Minimum spec |
| Mobile (Android) | 16,384-32,768 bytes | Varies by device |
| WebGPU Minimum | 16,384 bytes (16KB) | Guaranteed |

Always design for the minimum 16KB if you want portable code, or query limits and adapt.

## Performance Considerations

Workgroup memory is fast, but achieving optimal performance requires understanding hardware architecture and access patterns.

### Bank Conflicts

Workgroup memory is typically organized into banks—separate memory modules that can be accessed simultaneously. If multiple threads access the same bank simultaneously, a bank conflict occurs, serializing the accesses and reducing performance.

```wgsl
// BAD: All threads access same bank
var<workgroup> sharedData: array<f32, 256>;
@compute @workgroup_size(256)
fn badPattern(@builtin(local_invocation_index) idx: u32) {
    // All threads access element 0—serialized!
    let value = sharedData[0];
}

// GOOD: Each thread accesses different bank
@compute @workgroup_size(256)
fn goodPattern(@builtin(local_invocation_index) idx: u32) {
    // Each thread accesses its own element—parallel!
    let value = sharedData[idx];
}
```

Typical GPUs have 32 banks with 4-byte width, meaning consecutive 4-byte elements map to consecutive banks. Accessing elements with stride 32 (or multiples) can cause conflicts.

### Optimal Access Patterns

Best practices for workgroup memory access:

1. **Sequential Access**: Access consecutive elements in consecutive threads
2. **Broadcast**: All threads reading the same value is fast (broadcast)
3. **Avoid Strides**: Strided access (e.g., every 32nd element) can cause bank conflicts
4. **Padding**: Sometimes adding padding to arrays can avoid conflicts

```wgsl
// Without padding: stride causes conflicts
var<workgroup> data: array<vec3f, 64>;  // Each vec3f is 12 bytes

// With padding: better alignment
var<workgroup> data: array<vec4f, 64>;  // Each vec4f is 16 bytes
```

### Workgroup Size Impact

Choosing the right workgroup size affects performance:

- **Too Small**: Underutilizes GPU, may not saturate memory bandwidth
- **Too Large**: May exceed occupancy limits, reducing parallel workgroups
- **Optimal**: Typically 64-256 threads per workgroup for most GPUs

```typescript
// Common workgroup size choices
const config1D = tgsl.workgroupSize(256);      // 1D: linear data
const config2D = tgsl.workgroupSize(16, 16);   // 2D: image processing (256 total)
const config3D = tgsl.workgroupSize(8, 8, 4);  // 3D: volume processing (256 total)
```

## TypeGPU Integration

Complete example using TypeGPU for a parallel reduction:

```typescript
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import { tgsl, builtin } from 'typegpu/tgsl';

// Define workgroup variable
const sharedSums = tgpu.workgroupVar(d.arrayOf(d.f32, 256));

// Define buffers
const inputBuffer = tgpu.storage(d.arrayOf(d.f32));
const outputBuffer = tgpu.storage(d.arrayOf(d.f32), 'readwrite');

// Create kernel
const reduceKernel = tgsl
    .workgroupSize(256)
    .kernel([], () => {
        const localIdx = builtin.localInvocationIndex;
        const globalIdx = builtin.globalInvocationId.x;
        const workgroupIdx = builtin.workgroupId.x;

        // Load into shared memory
        sharedSums[localIdx] = inputBuffer[globalIdx];
        workgroupBarrier();

        // Reduction tree
        for (let stride = 128; stride > 0; stride = stride / 2) {
            if (localIdx < stride) {
                sharedSums[localIdx] = sharedSums[localIdx] + sharedSums[localIdx + stride];
            }
            workgroupBarrier();
        }

        // Write result
        if (localIdx === 0) {
            outputBuffer[workgroupIdx] = sharedSums[0];
        }
    });

// Usage
const device = await navigator.gpu.requestDevice();
const root = await tgpu.init(device);

const input = root.storage(d.arrayOf(d.f32), new Float32Array(1024));
const output = root.storage(d.arrayOf(d.f32), new Float32Array(4));

root.execute(
    reduceKernel
        .with({ input, output })
        .dispatch([4]) // 4 workgroups of 256 threads each
);
```

## Best Practices

1. **Always Synchronize**: Use `workgroupBarrier()` after writes and before reads
2. **Budget Memory Carefully**: Keep total usage under 16KB for portability
3. **Minimize Barriers**: Barriers serialize execution—use them when necessary, not excessively
4. **Avoid Bank Conflicts**: Use sequential access patterns when possible
5. **Choose Appropriate Workgroup Size**: 64-256 threads typically optimal
6. **Initialize Before Use**: Workgroup memory starts uninitialized—write before reading
7. **Profile**: Test on target hardware to validate performance assumptions

## Common Pitfalls

### Forgetting Barriers

```wgsl
// BUG: Race condition!
var<workgroup> shared: array<f32, 256>;
@compute @workgroup_size(256)
fn buggyKernel(@builtin(local_invocation_index) idx: u32) {
    shared[idx] = f32(idx);
    // Missing workgroupBarrier()!
    let value = shared[(idx + 1u) % 256u];  // May read uninitialized data!
}
```

### Exceeding Shared Memory Limits

```wgsl
// May exceed limit on some devices!
var<workgroup> huge: array<vec4f, 2048>;  // 2048 * 16 = 32KB
```

### Race Conditions

```wgsl
// BUG: Multiple threads writing to same location
var<workgroup> counter: u32;  // Should be atomic<u32>!
@compute @workgroup_size(256)
fn racyKernel() {
    counter += 1u;  // RACE CONDITION!
}

// FIX: Use atomic operations
var<workgroup> counter: atomic<u32>;
@compute @workgroup_size(256)
fn safeKernel() {
    atomicAdd(&counter, 1u);  // Thread-safe
}
```

### Uninitialized Reads

```wgsl
// BUG: Reading before writing
var<workgroup> data: array<f32, 256>;
@compute @workgroup_size(256)
fn uninitializedRead(@builtin(local_invocation_index) idx: u32) {
    if (idx == 0u) {
        data[0] = 42.0;
    }
    workgroupBarrier();
    // BUG: data[1..255] are uninitialized!
    let sum = data[idx] + data[idx + 1u];
}
```

---

## Conclusion

Workgroup variables and shared memory are essential tools for high-performance GPU compute programming. By providing fast on-chip memory accessible to all threads in a workgroup, they enable efficient implementation of parallel algorithms like reduction, scan, tiling, and cooperative caching. While workgroup memory requires careful synchronization and memory management, the performance benefits—often 10-20x speedups—make mastering these concepts worthwhile.

TypeGPU simplifies workgroup variable usage through type-safe APIs and automatic WGSL generation, reducing boilerplate while maintaining full control over synchronization and memory layout. Whether implementing image filters, physics simulations, or machine learning kernels, workgroup shared memory will be a cornerstone of your GPU acceleration strategy.

Remember the key principles: synchronize carefully with barriers, budget your 16KB wisely, avoid bank conflicts through sequential access, and always test on target hardware. With these practices, you'll unlock the full potential of parallel computing on the GPU.
