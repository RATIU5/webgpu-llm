---
title: Compute Pipelines
sidebar:
  order: 20
---

## Overview

Compute pipelines enable general-purpose GPU computation (GPGPU) in WebGPU, executing arbitrary parallel algorithms without rendering operations. Unlike render pipelines that produce visual output, compute pipelines process data through storage buffers and textures.

:::note[Why GPU Compute?]
Modern GPUs contain thousands of cores optimized for parallel execution. Tasks like machine learning inference, physics simulations, image processing, and data analysis that would take seconds on CPU can complete in milliseconds on GPU.
:::

## GPU Architecture

### Parallelism Model

| CPU | GPU |
|-----|-----|
| 4-16 high-performance cores | 2,000-10,000+ simpler cores |
| Optimized for sequential performance | Optimized for throughput |
| Low latency per operation | High throughput across operations |

GPUs execute the same instruction on different data simultaneously—the **SIMT (Single Instruction, Multiple Threads)** model. All threads in a group run the same instruction in lockstep.

:::caution[Thread Divergence]
When threads in a group take different branches, the GPU must execute both paths sequentially:

```wgsl
if (data[id] > threshold) {
  result[id] = expensiveCalculation(data[id]);
} else {
  result[id] = 0.0;
}
```

If half the threads take each branch, execution time equals the sum of both paths, not the faster one.
:::

### When GPUs Excel

GPUs become advantageous when:
- Workload is highly parallel (thousands+ independent operations)
- Data transfers to GPU memory are efficient
- Algorithm doesn't require frequent CPU-GPU synchronization
- Memory access patterns align with GPU architecture

:::tip[Size Threshold]
For matrix multiplication, GPUs become faster than CPUs when dimensions exceed ~256×256. Below this, data transfer overhead exceeds computational benefits.
:::

## Workgroups and Invocations

### Workgroup Structure

A **workgroup** is a collection of shader invocations that execute together and can cooperate through shared memory.

```wgsl title="Workgroup size declaration"
@compute @workgroup_size(8, 8, 1)
fn computeMain(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  // 8×8×1 = 64 invocations per workgroup
}
```

### Built-in Identifiers

| Built-in | Type | Description |
|----------|------|-------------|
| `local_invocation_id` | `vec3<u32>` | Position within workgroup (0 to size-1) |
| `workgroup_id` | `vec3<u32>` | Which workgroup this invocation belongs to |
| `global_invocation_id` | `vec3<u32>` | Unique position across all workgroups |
| `local_invocation_index` | `u32` | Linearized index within workgroup |
| `num_workgroups` | `vec3<u32>` | Total workgroups dispatched |

The `global_invocation_id` is computed as: `workgroup_id × workgroup_size + local_invocation_id`

### Choosing Workgroup Size

:::tip[Recommended Size: 64]
For general-purpose compute, use workgroup size of 64:
- Aligns with common GPU warp/wavefront sizes (32-64 threads)
- Provides enough parallelism to hide latency
- Works efficiently across diverse GPU architectures

For image processing, 2D workgroups like `@workgroup_size(8, 8)` are more intuitive.
:::

Query device limits:

```javascript title="Check workgroup limits"
const limits = device.limits;
console.log("Max workgroup size X:", limits.maxComputeWorkgroupSizeX);
console.log("Max invocations per workgroup:", limits.maxComputeInvocationsPerWorkgroup);
```

## Dispatching Work

### dispatchWorkgroups()

```javascript title="Dispatch compute work" {5-6}
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();

passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

passEncoder.end();
device.queue.submit([commandEncoder.finish()]);
```

### Calculating Dispatch Dimensions

**1D Data** (10,000 elements, workgroup size 64):

```javascript title="1D dispatch calculation"
const dataSize = 10000;
const workgroupSize = 64;
const workgroupsNeeded = Math.ceil(dataSize / workgroupSize);
passEncoder.dispatchWorkgroups(workgroupsNeeded, 1, 1);
// Dispatches 157 workgroups (157 × 64 = 10,048 invocations)
```

**2D Data** (1920×1080 image, 8×8 workgroups):

```javascript title="2D dispatch calculation"
const workgroupsX = Math.ceil(1920 / 8);
const workgroupsY = Math.ceil(1080 / 8);
passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
```

:::danger[Always Add Bounds Checks]
Dispatching more invocations than data elements causes out-of-bounds access:

```wgsl
@compute @workgroup_size(64)
fn process(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&data)) {
    return;  // Guard against overflow
  }
  // Process data[id.x]
}
```
:::

### Dispatch Strategies

**Multi-pass for cross-workgroup dependencies:**

When results from one workgroup affect another, use separate dispatches:

```javascript title="Multi-pass reduction"
// Pass 1: Partial sums within workgroups
pass1.dispatchWorkgroups(Math.ceil(dataSize / 256));
pass1.end();

// Pass 2: Final reduction of partial results
const pass2 = encoder.beginComputePass();
pass2.setPipeline(reductionPipeline);
pass2.setBindGroup(0, partialResultsBindGroup);
pass2.dispatchWorkgroups(1);  // Single workgroup for final sum
pass2.end();
```

**Indirect dispatch for GPU-driven workloads:**

Let the GPU determine dispatch size:

```javascript title="GPU-driven dispatch"
// Buffer holds dispatch dimensions (3 × u32)
const indirectBuffer = device.createBuffer({
  size: 12,
  usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

// Compute shader writes workgroup counts
// Then dispatch using those counts
pass.dispatchWorkgroupsIndirect(indirectBuffer, 0);
```

```wgsl title="Write dispatch args in shader"
struct DispatchArgs {
  x: u32,
  y: u32,
  z: u32,
}

@group(0) @binding(0) var<storage, read_write> dispatch: DispatchArgs;
@group(0) @binding(1) var<storage, read> activeCount: u32;

@compute @workgroup_size(1)
fn prepareDispatch() {
  dispatch.x = (activeCount + 63u) / 64u;  // Workgroups needed
  dispatch.y = 1u;
  dispatch.z = 1u;
}
```

## Creating Compute Pipelines

### GPUComputePipeline

```javascript title="Create compute pipeline" {2-4}
const computePipeline = device.createComputePipeline({
  label: "data-processing-pipeline",
  layout: "auto",
  compute: {
    module: shaderModule,
    entryPoint: "main",
  },
});
```

Use async creation to avoid blocking:

```javascript title="Async pipeline creation"
const computePipeline = await device.createComputePipelineAsync({
  layout: "auto",
  compute: { module: shaderModule, entryPoint: "main" },
});
```

### Storage Buffers

Storage buffers are the primary data mechanism for compute shaders:

```javascript title="Create storage buffers" {3-4,9-10}
const inputBuffer = device.createBuffer({
  size: dataArray.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});
new Float32Array(inputBuffer.getMappedRange()).set(dataArray);
inputBuffer.unmap();

const outputBuffer = device.createBuffer({
  size: dataArray.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
```

In WGSL:

```wgsl title="Storage buffer declarations"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
```

| Access Mode | WGSL | Use Case |
|-------------|------|----------|
| Read-only | `var<storage, read>` | Input data |
| Read-write | `var<storage, read_write>` | Output or in-place updates |

### Complete Example

```javascript title="Full compute pipeline setup" {19-28}
const shaderCode = `
  @group(0) @binding(0) var<storage, read> input: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&input)) {
      output[i] = sqrt(input[i]);
    }
  }
`;

const shaderModule = device.createShaderModule({ code: shaderCode });
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: shaderModule, entryPoint: "main" },
});

// Create buffers and bind group
const inputData = new Float32Array(1000).map((_, i) => i);
const inputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(inputBuffer, 0, inputData);

const outputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: inputBuffer } },
    { binding: 1, resource: { buffer: outputBuffer } },
  ],
});

// Execute
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(inputData.length / 64));
pass.end();

device.queue.submit([encoder.finish()]);
```

## TypeGPU Compute Pipelines

TypeGPU provides type-safe compute pipeline abstractions:

```typescript title="TypeGPU compute pipeline" {6-15}
import tgpu from "typegpu";
import * as d from "typegpu/data";

const root = await tgpu.init();

const doubleValues = tgpu.computeFn([], () => {
  "use gpu";

  const id = builtin.globalInvocationId.x;
  if (id < inputBuffer.value.length) {
    outputBuffer.value[id] = inputBuffer.value[id] * 2.0;
  }
}).$workgroupSize(64);

const pipeline = root["~unstable"].withCompute(doubleValues).createPipeline();
pipeline.with(bindGroup).dispatchWorkgroups(Math.ceil(1000 / 64));
```

## Memory Access Patterns

### Coalesced Access

:::tip[Coalesce Memory Accesses]
Adjacent threads should access adjacent memory locations. GPUs combine coalesced accesses into single, larger transactions:

```wgsl
// Good: Coalesced access
output[id.x] = input[id.x];

// Bad: Scattered access
output[id.x] = input[id.x * stride];
```
:::

### Shared Memory

Workgroup shared memory (`var<workgroup>`) is fast on-chip memory shared by all invocations in a workgroup:

```wgsl title="Using shared memory" {1,8,11,14}
var<workgroup> sharedCache: array<f32, 64>;

@compute @workgroup_size(64)
fn process(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  // Load into shared memory
  sharedCache[local_id.x] = input[global_id.x];

  // Wait for all threads to finish loading
  workgroupBarrier();

  // Perform operations using cached data
  var sum = 0.0;
  for (var i = 0u; i < 64u; i++) {
    sum += sharedCache[i];
  }

  output[global_id.x] = sum;
}
```

:::tip[Learn More]
See [Workgroup Variables](/advanced/workgroup-variables/) for advanced patterns: parallel reductions, prefix sums, and histogram computation.
:::

## Synchronization

### workgroupBarrier()

Synchronizes all invocations in a workgroup—all threads must reach the barrier before any can proceed:

```wgsl title="Barrier synchronization"
var<workgroup> data: array<f32, 64>;

@compute @workgroup_size(64)
fn compute(@builtin(local_invocation_id) local_id: vec3<u32>) {
  // Phase 1: Each thread writes
  data[local_id.x] = computeValue();

  // Wait for all writes to complete
  workgroupBarrier();

  // Phase 2: Safely read other threads' data
  let sum = data[0] + data[local_id.x];
}
```

### storageBarrier()

Ensures all storage buffer writes within a workgroup are visible before proceeding:

```wgsl title="Storage barrier"
@group(0) @binding(0) var<storage, read_write> buffer: array<f32>;

@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
  // Write to storage buffer
  buffer[id.x] = computeValue(id.x);

  // Ensure writes are visible within workgroup
  storageBarrier();

  // Now reads see updated values (within same workgroup)
  let neighbor = buffer[id.x + 1u];
}
```

:::danger[Workgroup Scope Only]
Both `workgroupBarrier()` and `storageBarrier()` only synchronize invocations **within the same workgroup**. They cannot synchronize across different workgroups.

For cross-workgroup synchronization:
- Use multiple dispatch calls (separate passes)
- Use atomics for coordination
:::

### Barrier Comparison

| Barrier | Scope | Memory Affected |
|---------|-------|-----------------|
| `workgroupBarrier()` | Workgroup | `var<workgroup>` |
| `storageBarrier()` | Workgroup | `var<storage>` |
| `textureBarrier()` | Workgroup | Storage textures |

:::danger[Race Conditions]
Without barriers, some threads may read before others have written:

```wgsl
// BAD: Race condition!
counter[0] = counter[0] + 1;  // Multiple threads write simultaneously

// GOOD: Use atomics
atomicAdd(&counter[0], 1u);
```
:::

## Common Use Cases

<details>
<summary>**Physics Simulation**</summary>

```wgsl
@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= particleCount) { return; }

  particles[i].velocity += gravity * deltaTime;
  particles[i].position += particles[i].velocity * deltaTime;
}
```

</details>

<details>
<summary>**Image Processing**</summary>

```wgsl
@compute @workgroup_size(8, 8)
fn applyFilter(@builtin(global_invocation_id) id: vec3<u32>) {
  let coord = vec2<i32>(id.xy);
  if (any(coord >= textureDimensions(inputTexture))) { return; }

  let color = textureLoad(inputTexture, coord, 0);
  let processed = filterFunction(color);
  textureStore(outputTexture, coord, processed);
}
```

</details>

<details>
<summary>**Matrix Multiplication**</summary>

```wgsl
@compute @workgroup_size(8, 8)
fn matrixMultiply(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let col = id.y;

  var sum = 0.0;
  for (var k = 0u; k < matrixSize; k++) {
    sum += matrixA[row * matrixSize + k] * matrixB[k * matrixSize + col];
  }

  result[row * matrixSize + col] = sum;
}
```

</details>

## Resources

:::note[Official Documentation]
- [WebGPU Compute Specification](https://gpuweb.github.io/gpuweb/#compute-pipeline)
- [TypeGPU Compute Guide](https://docs.swmansion.com/TypeGPU/fundamentals/compute/)
- [WebGPU Fundamentals: Compute](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
:::
