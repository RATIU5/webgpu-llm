# Compute Pipelines

## Overview

**Compute pipelines** represent one of WebGPU's most powerful capabilities, enabling general-purpose GPU computation (GPGPU) directly in the browser. Unlike render pipelines that focus on graphics and produce visual output to textures or the screen, compute pipelines execute arbitrary parallel algorithms on the GPU without any rendering operations. This distinction makes compute pipelines ideal for data-parallel workloads that benefit from massive parallelism but don't require rasterization or fragment processing.

The significance of GPGPU for web applications cannot be overstated. Modern GPUs contain thousands of processing cores specifically designed to execute the same operation across large datasets simultaneously. This architecture makes them exceptionally well-suited for tasks like machine learning inference, physics simulations, image processing, data analysis, cryptographic operations, and scientific computing. WebGPU's compute capabilities bring these performance characteristics to the web platform, enabling applications that were previously impractical or impossible in browsers.

Compute shaders execute independently of the graphics pipeline, reading input data from storage buffers and textures, performing calculations, and writing results back to GPU memory. They operate on a **workgroup-based execution model** where work is distributed across a three-dimensional grid of thread groups. Each workgroup contains a specified number of work items (individual shader invocations) that can cooperate through shared memory and synchronization primitives. This model maps efficiently to modern GPU hardware while providing flexibility for diverse computational patterns.

Understanding compute pipelines requires shifting from the familiar graphics mindset to a parallel computing perspective. Instead of thinking about vertices, triangles, and pixels, you reason about data arrays, parallel iterations, and work distribution. The GPU becomes a massively parallel coprocessor capable of executing hundreds or thousands of shader instances concurrently, transforming what would be sequential loops on the CPU into parallel operations that complete in a fraction of the time.

## Understanding Parallel Computing

Modern GPU architecture fundamentally differs from CPU design. While CPUs optimize for sequential performance with a few high-performance cores (typically 4-16 on consumer hardware), GPUs prioritize throughput with thousands of smaller, simpler cores. A mid-range GPU might contain 2,000-4,000 cores, while high-end GPUs can exceed 10,000 cores. These cores are optimized for floating-point arithmetic and operate at lower clock speeds than CPU cores, but their sheer quantity enables massive parallelism for suitable workloads.

### GPU Architecture: Thousands of Cores

GPU cores are organized into hierarchical groupings. NVIDIA calls these groupings "streaming multiprocessors" (SMs), AMD uses "compute units" (CUs), and different vendors have varying architectures, but the concept remains consistent: multiple cores share instruction dispatch, caches, and memory controllers. This organization influences how work should be structured for optimal performance.

Each core executes the same instruction stream on different data, following the **SIMT (Single Instruction, Multiple Threads)** execution model. SIMT differs from traditional CPU threading models because all threads within a group execute in lockstep, running the same instruction at the same time on different data elements. When threads within a group diverge (through conditional branches, for example), execution becomes serialized—threads that don't take the branch idle while others execute the branch code, then vice versa.

### SIMT Execution Model

The SIMT model has profound implications for shader programming. Consider a simple conditional:

```wgsl
if (data[id] > threshold) {
  result[id] = expensiveCalculation(data[id]);
} else {
  result[id] = 0.0;
}
```

If half the threads in a workgroup take the `if` branch and half take the `else` branch, the GPU must execute both paths, masking out threads that don't participate in each path. This means the execution time is the sum of both paths, not the faster path. This phenomenon, called **thread divergence**, can significantly impact performance. Well-optimized compute shaders minimize divergence by organizing data so threads in the same workgroup follow similar code paths.

### Why GPUs Excel at Parallel Tasks

GPUs achieve their performance advantage through several architectural features:

**1. Memory Bandwidth**: GPUs have significantly higher memory bandwidth than CPUs (often 500-1000 GB/s versus 50-100 GB/s), enabling them to feed data to thousands of cores simultaneously. This bandwidth is critical for compute workloads that are memory-bound rather than compute-bound.

**2. Latency Hiding**: Unlike CPUs that optimize for low latency on individual operations, GPUs hide memory latency by switching between threads. When one thread stalls waiting for memory, the GPU immediately switches to another thread. With thousands of threads in flight, there's always useful work to do.

**3. Specialized Hardware**: Modern GPUs include specialized functional units for operations like texture sampling, matrix multiplication (tensor cores), and ray tracing. Compute shaders can leverage these units for applicable workloads.

**4. Power Efficiency**: For parallel workloads, GPUs deliver substantially more operations per watt than CPUs, making them attractive for both performance and energy efficiency.

However, GPUs are not universally faster. The benefits only materialize when:
- The workload is highly parallel (thousands or millions of independent operations)
- Data can be efficiently transferred to GPU memory
- The algorithm doesn't require frequent CPU-GPU synchronization
- Memory access patterns align with GPU memory architecture

Research from Chrome's documentation indicates that for matrix multiplication, the GPU becomes advantageous when matrix dimensions exceed 256×256. Below this threshold, the overhead of data transfer and kernel launch exceeds the computational benefits. This crossover point varies by workload and hardware, but it illustrates an important principle: **compute shaders shine for large-scale parallel problems**, not small calculations.

## Workgroups and Work Items

The workgroup model forms the foundation of GPU compute execution. Understanding this model is essential for effective compute shader programming.

### Workgroup: Unit of Parallel Execution

A **workgroup** (also called a thread group or work group in other APIs) is a collection of shader invocations that execute together and can cooperate. Workgroups are the basic unit of work scheduling on the GPU—the scheduler assigns entire workgroups to execution units, not individual invocations.

Workgroups are defined with a three-dimensional size using the `@workgroup_size(x, y, z)` attribute in WGSL:

```wgsl
@compute @workgroup_size(8, 8, 1)
fn computeMain(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  // Shader code here
}
```

This example creates workgroups containing 8×8×1 = 64 invocations. Each invocation runs the `computeMain` function with different built-in values identifying its position in the execution grid.

### Work Items/Threads: Individual Invocations

Each invocation within a workgroup is called a **work item** or thread. These work items execute the same shader code but can access different data based on their indices. The total number of invocations in a workgroup is the product of the three dimensions: `x * y * z`.

WebGPU imposes a limit on the total invocations per workgroup. The minimum guaranteed limit is 256, though most hardware supports more. The specification also defines per-dimension limits (minimum 256 for x and y dimensions, 64 for z). You can query actual limits via:

```javascript
const limits = device.limits;
console.log('Max workgroup size X:', limits.maxComputeWorkgroupSizeX);
console.log('Max workgroup size Y:', limits.maxComputeWorkgroupSizeY);
console.log('Max workgroup size Z:', limits.maxComputeWorkgroupSizeZ);
console.log('Max invocations per workgroup:', limits.maxComputeInvocationsPerWorkgroup);
```

### @workgroup_size(x, y, z) in WGSL

The workgroup size declaration is static—it's part of the shader code and cannot be changed without recompiling the shader. This static declaration enables important optimizations because the driver knows the exact workgroup configuration at compile time.

Choosing workgroup dimensions is both a functional and performance consideration:

**Functional Considerations**: The dimensionality should match your problem domain. Image processing naturally maps to 2D workgroups (x and y for pixel coordinates, z = 1). 3D simulations use 3D workgroups. One-dimensional data uses 1D workgroups (x for element index, y = z = 1).

**Performance Considerations**: Most GPU hardware executes threads in groups of 32 or 64 (called warps on NVIDIA hardware, wavefronts on AMD). For optimal performance, workgroup sizes should be multiples of this native hardware grouping. The commonly recommended workgroup size for general-purpose work is 64, specified as `@workgroup_size(64)` or `@workgroup_size(64, 1, 1)`.

### 3D Grid of Workgroups

When you dispatch compute work, you specify how many workgroups to execute in each dimension. This creates a three-dimensional grid of workgroups, where each workgroup contains its own 3D grid of invocations. For example:

```javascript
// Dispatch 4x3x2 workgroups
pass.dispatchWorkgroups(4, 3, 2);
```

If each workgroup has size `@workgroup_size(8, 8, 1)`, this dispatch creates:
- 4×3×2 = 24 total workgroups
- Each workgroup contains 8×8×1 = 64 invocations
- Total invocations: 24 × 64 = 1,536 shader executions

The GPU may execute multiple workgroups concurrently depending on available hardware resources. However, **the specification provides no guarantees about execution order**. Workgroups might execute sequentially, in parallel, or in any arbitrary order. This indeterminacy is intentional—it gives the GPU scheduler maximum flexibility to optimize for hardware capabilities.

### Built-in Identifiers: local_invocation_id, global_invocation_id, workgroup_id

WGSL provides built-in variables that identify each invocation's position within the execution grid:

**@builtin(local_invocation_id)**: A `vec3<u32>` representing the invocation's position within its workgroup. Values range from (0,0,0) to (workgroup_size.x - 1, workgroup_size.y - 1, workgroup_size.z - 1). All invocations in the same workgroup have different local_invocation_id values.

**@builtin(workgroup_id)**: A `vec3<u32>` identifying which workgroup this invocation belongs to. All invocations within the same workgroup have the same workgroup_id. Values range from (0,0,0) to (dispatch_size.x - 1, dispatch_size.y - 1, dispatch_size.z - 1).

**@builtin(global_invocation_id)**: A `vec3<u32>` representing the invocation's unique position across all workgroups. It's computed as: `workgroup_id * workgroup_size + local_invocation_id`. This is often the most useful identifier for addressing data arrays.

**@builtin(local_invocation_index)**: A `u32` providing a linearized index within the workgroup, calculated as: `local_invocation_id.z * workgroup_size.x * workgroup_size.y + local_invocation_id.y * workgroup_size.x + local_invocation_id.x`. Useful when you need a scalar index for accessing shared memory.

**@builtin(num_workgroups)**: A `vec3<u32>` containing the dispatch dimensions passed to `dispatchWorkgroups()`. This tells the shader how many workgroups were launched.

These built-ins enable each invocation to determine what data to process:

```wgsl
@compute @workgroup_size(64)
fn processArray(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= arrayLength(&inputData)) {
    return; // Guard against out-of-bounds access
  }
  outputData[index] = someFunction(inputData[index]);
}
```

## Dispatching Work

Dispatching is the act of launching compute work on the GPU. It's analogous to draw calls in rendering but for compute shaders.

### dispatchWorkgroups(x, y, z)

The `dispatchWorkgroups()` method on GPUComputePassEncoder specifies how many workgroups to execute in each dimension:

```javascript
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();

passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

passEncoder.end();
device.queue.submit([commandEncoder.finish()]);
```

Each parameter represents the number of workgroups in that dimension. The total number of invocations is:

```
total_invocations = workgroupsX * workgroupsY * workgroupsZ *
                   workgroup_size.x * workgroup_size.y * workgroup_size.z
```

### Calculating Dispatch Dimensions

A common pattern is processing an array or image where you need to ensure every element is processed. The calculation must account for the workgroup size:

**For 1D Data** (e.g., processing 10,000 elements with workgroup size 64):

```javascript
const dataSize = 10000;
const workgroupSize = 64;
const workgroupsNeeded = Math.ceil(dataSize / workgroupSize);
passEncoder.dispatchWorkgroups(workgroupsNeeded, 1, 1);
// Dispatches 157 workgroups (157 * 64 = 10,048 invocations)
```

The shader needs bounds checking because you'll launch slightly more invocations than needed:

```wgsl
@compute @workgroup_size(64)
fn process(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&data)) {
    return; // Don't process beyond array bounds
  }
  // Process data[id.x]
}
```

**For 2D Data** (e.g., processing a 1920×1080 image with 8×8 workgroups):

```javascript
const imageWidth = 1920;
const imageHeight = 1080;
const workgroupSizeX = 8;
const workgroupSizeY = 8;

const workgroupsX = Math.ceil(imageWidth / workgroupSizeX);
const workgroupsY = Math.ceil(imageHeight / workgroupSizeY);

passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
// Dispatches 240×135 workgroups
```

### Covering Data with Workgroups

The key principle is ensuring your dispatch dimensions, when multiplied by workgroup dimensions, equal or exceed your data dimensions. Overshooting is acceptable with proper bounds checking in the shader. Undershooting means some data won't be processed, causing incorrect results.

For non-uniform data sizes or when workgroup size doesn't evenly divide the data size, always round up with `Math.ceil()` and add bounds checks in shaders.

## WebGPU Compute Pipeline

Creating and using compute pipelines in vanilla WebGPU involves several steps: defining the shader, creating the pipeline object, setting up storage buffers for input/output, and encoding compute commands.

### Creating GPUComputePipeline

Compute pipelines are created through `device.createComputePipeline()`:

```javascript
const computePipeline = device.createComputePipeline({
  label: 'data-processing-pipeline',
  layout: pipelineLayout,
  compute: {
    module: shaderModule,
    entryPoint: 'main',
    constants: {
      // Optional: Override pipeline-overridable constants
      blockSize: 256,
    }
  }
});
```

**Asynchronous Creation**: For improved performance during initialization, use the async variant:

```javascript
const computePipeline = await device.createComputePipelineAsync({
  label: 'data-processing-pipeline',
  layout: pipelineLayout,
  compute: {
    module: shaderModule,
    entryPoint: 'main'
  }
});
```

The async version allows the browser to compile the pipeline on background threads without blocking the main thread, which is particularly valuable for complex shaders that take longer to compile.

### GPUComputePipelineDescriptor

The descriptor object configures the compute pipeline:

**label**: An optional debugging label that appears in error messages and profiling tools.

**layout**: Either a GPUPipelineLayout object explicitly defining bind group layouts, or the string `'auto'` to automatically infer the layout from shader bindings. Explicit layouts provide more control but require more code. Auto layout is convenient for simple cases:

```javascript
// Auto layout (inferred from shader)
layout: 'auto'

// Explicit layout (more control)
layout: device.createPipelineLayout({
  bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1]
})
```

**compute**: An object specifying the compute stage configuration:
- **module**: The GPUShaderModule containing compiled WGSL code
- **entryPoint**: The name of the function decorated with `@compute` to use as the entry point
- **constants**: Optional map of pipeline-overridable constants

### Compute Stage Configuration

The compute stage is simpler than render pipeline stages because there's only one stage. The shader module is created from WGSL source:

```javascript
const shaderModule = device.createShaderModule({
  label: 'compute-shader',
  code: `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      let index = id.x;
      if (index < arrayLength(&input)) {
        output[index] = input[index] * 2.0;
      }
    }
  `
});
```

### Storage Buffers

Storage buffers are the primary mechanism for compute shaders to access large amounts of data. Unlike uniform buffers (which have size limitations and are read-only), storage buffers can be large (up to device limits, often gigabytes) and support both reading and writing.

**Creating Storage Buffers**:

```javascript
const inputBuffer = device.createBuffer({
  size: dataArray.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true
});
new Float32Array(inputBuffer.getMappedRange()).set(dataArray);
inputBuffer.unmap();

const outputBuffer = device.createBuffer({
  size: dataArray.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
});
```

### Input and Output Buffers

Storage buffers are declared in WGSL with access mode qualifiers:

**Read-only**: `var<storage, read>` allows the shader to read but not modify the buffer. This is appropriate for input data.

**Read-write**: `var<storage, read_write>` allows both reading and writing. Required for output buffers or buffers that need to be updated in place.

The access mode affects validation and may enable optimizations. If you only read from a buffer, declare it as `read` even though `read_write` would work—it communicates intent and may improve performance.

### Command Encoding

Encoding compute commands follows a pattern similar to render passes but simpler:

```javascript
const commandEncoder = device.createCommandEncoder({
  label: 'compute-commands'
});

const passEncoder = commandEncoder.beginComputePass({
  label: 'compute-pass'
});

passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(
  Math.ceil(elementCount / workgroupSize)
);

passEncoder.end();

device.queue.submit([commandEncoder.finish()]);
```

### beginComputePass()

The `beginComputePass()` method creates a GPUComputePassEncoder. Unlike render passes that require complex descriptors specifying attachments, compute passes are simpler—they just need an optional label:

```javascript
const passEncoder = commandEncoder.beginComputePass({
  label: 'physics-simulation',
  timestampWrites: { // Optional: GPU timestamps for profiling
    querySet: timestampQuerySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1
  }
});
```

### setPipeline(), setBindGroup(), dispatchWorkgroups(), end()

These methods configure and execute the compute pass:

**setPipeline(pipeline)**: Binds a compute pipeline, specifying which shader to run.

**setBindGroup(index, bindGroup)**: Binds resources (buffers, textures, samplers) at the specified bind group index. The index corresponds to `@group(N)` in the shader.

**dispatchWorkgroups(x, y, z)**: Launches the compute work with the specified workgroup counts.

**end()**: Finalizes the compute pass. After calling end(), the pass encoder is invalid and the commands are recorded into the parent command encoder.

**Complete Example**:

```javascript
// Setup
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
  layout: 'auto',
  compute: {
    module: shaderModule,
    entryPoint: 'main'
  }
});

// Create and populate input buffer
const inputData = new Float32Array(1000).map((_, i) => i);
const inputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true
});
new Float32Array(inputBuffer.getMappedRange()).set(inputData);
inputBuffer.unmap();

// Create output buffer
const outputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
});

// Create bind group
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: inputBuffer } },
    { binding: 1, resource: { buffer: outputBuffer } }
  ]
});

// Encode and submit
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(inputData.length / 64));
pass.end();

device.queue.submit([encoder.finish()]);
```

## TypeGPU Compute Pipelines

TypeGPU provides a higher-level abstraction for compute pipelines, reducing boilerplate while maintaining type safety. The API uses builder patterns and automatic resource management.

### root.withCompute(fn) Builder

TypeGPU compute pipelines start with a root object and use the `withCompute()` method:

```typescript
import tgpu from 'typegpu';

const root = await tgpu.init();

const computeFn = tgpu.computeFn([], (/* parameters */) => {
  'use gpu';
  // Shader code written in TGSL
});

const pipeline = root['~unstable']
  .withCompute(computeFn)
  .createPipeline();
```

The `'~unstable'` accessor indicates API features that may change in future versions. TypeGPU is actively developed, and this syntax acknowledges the evolving nature of the API.

### TgpuComputeFn Definition

A `TgpuComputeFn` is created using `tgpu.computeFn()`, which defines a compute shader in TypeScript that will be transpiled to WGSL:

```typescript
import * as d from 'typegpu/data';
import { wgsl } from 'typegpu/experimental';

const inputBuffer = root.createBuffer(d.arrayOf(d.f32, 1000));
const outputBuffer = root.createBuffer(d.arrayOf(d.f32, 1000));

const doubleValues = tgpu.computeFn(
  [inputBuffer, outputBuffer],
  (input, output) => {
    'use gpu';

    wgsl.workgroupSize = [64, 1, 1];
    const id = wgsl.globalInvocationId.x;

    if (id < input.length) {
      output[id] = input[id] * 2.0;
    }
  }
);
```

The `'use gpu'` directive marks this function for GPU execution. TypeGPU's build plugin transpiles the function body to WGSL.

### dispatchWorkgroups() Execution

Executing a TypeGPU compute pipeline uses the same `dispatchWorkgroups()` method, but resources are bound using `.with()`:

```typescript
const pipeline = root['~unstable']
  .withCompute(computeFn)
  .createPipeline();

// Execute
pipeline
  .with(bindGroup)
  .dispatchWorkgroups(Math.ceil(1000 / 64));
```

TypeGPU handles bind group creation and resource binding automatically based on the function parameters.

### Reading Results

After compute execution, you need to read results back to CPU memory. TypeGPU provides methods to facilitate this:

```typescript
// Dispatch work
pipeline
  .with(bindGroup)
  .dispatchWorkgroups(workgroupCount);

// Create staging buffer for readback
const stagingBuffer = device.createBuffer({
  size: outputBuffer.size,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
});

// Copy GPU results to staging buffer
const encoder = device.createCommandEncoder();
encoder.copyBufferToBuffer(
  outputBuffer,
  0,
  stagingBuffer,
  0,
  outputBuffer.size
);
device.queue.submit([encoder.finish()]);

// Map and read
await stagingBuffer.mapAsync(GPUMapMode.READ);
const resultData = new Float32Array(stagingBuffer.getMappedRange());
console.log(resultData);
stagingBuffer.unmap();
```

This pattern is necessary because storage buffers cannot be directly mapped—you must copy to a mappable buffer first.

## Memory Access Patterns

Efficient compute shader performance depends heavily on memory access patterns. GPUs are extremely sensitive to how threads access memory.

### Coalesced Memory Access

**Coalesced access** occurs when adjacent threads access adjacent memory locations. Modern GPUs combine multiple memory requests into single, larger transactions when accesses are coalesced. This dramatically reduces memory latency and increases bandwidth utilization.

**Good (Coalesced)**:
```wgsl
// Each thread reads consecutive elements
output[id.x] = input[id.x];
```

**Bad (Uncoalesced)**:
```wgsl
// Threads read scattered locations
output[id.x] = input[id.x * stride];  // If stride is not 1
```

When threads in the same warp/wavefront access scattered memory locations, the GPU must issue multiple separate memory transactions, drastically reducing throughput.

### Bank Conflicts

Shared memory (workgroup memory in WebGPU) is divided into banks. When multiple threads in a warp access different addresses in the same bank simultaneously, a **bank conflict** occurs, forcing sequential access and degrading performance.

```wgsl
var<workgroup> sharedData: array<f32, 256>;

// Bad: All threads in a warp might hit the same bank
sharedData[local_id.x * 32] = value;

// Good: Consecutive thread IDs access consecutive addresses
sharedData[local_id.x] = value;
```

Bank conflicts are hardware-specific and harder to reason about than coalesced access, but following simple rules (use consecutive indices for consecutive threads) usually avoids problems.

### Shared Memory Optimization

Shared memory (declared with `var<workgroup>`) is fast, on-chip memory shared by all invocations in a workgroup. It's much faster than storage buffers but limited in size (minimum 16 KB guaranteed by WebGPU).

Common pattern: Load data from slow storage buffers into fast shared memory, perform multiple operations using the cached data, then write results back:

```wgsl
var<workgroup> sharedCache: array<f32, 64>;

@compute @workgroup_size(64)
fn process(@builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(global_invocation_id) global_id: vec3<u32>) {
  // Load into shared memory
  sharedCache[local_id.x] = input[global_id.x];

  // Wait for all threads to finish loading
  workgroupBarrier();

  // Perform operations using shared data
  var sum = 0.0;
  for (var i = 0u; i < 64u; i++) {
    sum += sharedCache[i];
  }

  output[global_id.x] = sum;
}
```

## Synchronization

Because invocations in a workgroup execute concurrently, they need synchronization primitives to coordinate access to shared resources.

### workgroupBarrier()

The `workgroupBarrier()` function synchronizes all invocations in a workgroup. When an invocation reaches a barrier, it waits until all other invocations in the workgroup reach the same barrier, then all proceed together.

```wgsl
var<workgroup> data: array<f32, 64>;

@compute @workgroup_size(64)
fn compute(@builtin(local_invocation_id) local_id: vec3<u32>) {
  // Phase 1: Each thread writes
  data[local_id.x] = computeValue();

  // Wait for all writes to complete
  workgroupBarrier();

  // Phase 2: Each thread reads (safely, because all writes are done)
  let sum = data[0] + data[local_id.x];
}
```

Without the barrier, some threads might read `data` before other threads have written, leading to race conditions and undefined behavior.

### storageBarrier()

The `storageBarrier()` function ensures memory operations complete before subsequent operations. It's similar to `workgroupBarrier()` but applies to storage buffer accesses rather than workgroup memory:

```wgsl
@compute @workgroup_size(64)
fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
  buffer[id.x] = someValue;
  storageBarrier();
  let value = buffer[otherIndex];  // Guaranteed to see prior writes
}
```

In practice, `storageBarrier()` is less commonly needed than `workgroupBarrier()` because **WebGPU provides no guarantees about inter-workgroup execution order**. You cannot reliably synchronize between different workgroups—each workgroup should be independent.

### Memory Ordering

WebGPU's memory model is complex, but key principles include:

1. **Workgroup memory** is coherent within a workgroup—threads see each other's writes after synchronization
2. **Storage buffers** have undefined ordering between workgroups
3. **Atomic operations** provide specific ordering guarantees for concurrent access

For most compute shaders, organizing work so each workgroup operates on independent data avoids synchronization issues entirely.

## Common Use Cases

Compute shaders excel at diverse workloads:

### Physics Simulation

N-body simulations, particle systems, fluid dynamics, cloth simulation, and rigid body physics benefit from parallelism. Each particle or body can be updated independently:

```wgsl
@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= particleCount) { return; }

  particles[i].velocity += gravity * deltaTime;
  particles[i].position += particles[i].velocity * deltaTime;
}
```

### Image Processing

Filters (blur, sharpen, edge detection), color grading, tone mapping, and computer vision algorithms process each pixel independently:

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

### Data Parallel Algorithms

Sorting, searching, reductions (sum, max, min), prefix scans, and other algorithms with parallel formulations map well to compute shaders.

### ML Inference

Matrix multiplications, convolutions, activation functions, and other neural network operations can be implemented as compute shaders, enabling browser-based machine learning:

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

## Best Practices

### Workgroup Size of 64

For general-purpose compute work on WebGPU, a workgroup size of 64 is recommended. This size:
- Aligns with common GPU warp/wavefront sizes (32-64 threads)
- Provides enough parallelism to hide latency
- Stays well under the 256 invocation minimum limit
- Works efficiently across diverse GPU architectures

For specialized cases (like image processing), 2D workgroups like `@workgroup_size(8, 8)` (also 64 total) may be more intuitive.

### Memory Access Patterns

- **Coalesce accesses**: Have consecutive threads access consecutive memory locations
- **Minimize divergence**: Keep threads in a workgroup following similar code paths
- **Use shared memory**: Cache frequently accessed data in workgroup memory
- **Align data structures**: Follow WebGPU alignment rules to avoid wasted bandwidth
- **Batch operations**: Process multiple elements per thread if it improves cache utilization

### Additional Best Practices

- **Minimize CPU-GPU transfers**: Keep data on GPU across multiple operations
- **Overlap computation and transfer**: Use multiple command buffers to pipeline work
- **Profile before optimizing**: Use timestamp queries to identify actual bottlenecks
- **Test on target hardware**: Performance characteristics vary significantly between GPUs
- **Validate dispatch dimensions**: Always include bounds checks in shaders

## Common Pitfalls

### Race Conditions

**Problem**: Multiple threads access the same memory location without synchronization.

```wgsl
// Bad: Race condition!
@compute @workgroup_size(64)
fn buggy(@builtin(global_invocation_id) id: vec3<u32>) {
  counter[0] = counter[0] + 1;  // Multiple threads write simultaneously
}
```

**Solution**: Use atomic operations or organize work so threads don't conflict:

```wgsl
// Good: Each thread updates unique location
output[id.x] = process(input[id.x]);

// Or use atomics for shared counters
atomicAdd(&counter[0], 1u);
```

### Exceeding Limits

**Problem**: Requesting workgroup sizes or dispatch dimensions that exceed device limits.

**Solution**: Query limits and validate:

```javascript
const maxWorkgroupSize = device.limits.maxComputeInvocationsPerWorkgroup;
const maxDispatch = device.limits.maxComputeWorkgroupsPerDimension;

if (requestedSize > maxWorkgroupSize) {
  throw new Error(`Workgroup size ${requestedSize} exceeds limit ${maxWorkgroupSize}`);
}
```

### Forgetting Bounds Checks

**Problem**: Dispatching more invocations than data elements without checks leads to out-of-bounds access.

**Solution**: Always guard array accesses:

```wgsl
if (id.x >= arrayLength(&data)) {
  return;
}
```

### Incorrect Synchronization

**Problem**: Assuming workgroups execute in order or can synchronize with each other.

**Solution**: Design algorithms where workgroups are independent. Use multi-pass approaches if dependencies exist.

### Inefficient Memory Patterns

**Problem**: Scattered memory accesses destroy performance.

**Solution**: Profile and restructure data layouts to enable coalesced access.

---

This comprehensive guide covers the essential concepts of compute pipelines in WebGPU. By understanding workgroups, dispatch, memory patterns, and synchronization, you can harness the massive parallelism of modern GPUs for general-purpose computation directly in web applications. Whether implementing physics simulations, processing images, or running machine learning inference, compute pipelines provide the performance foundation for GPU-accelerated web experiences.
