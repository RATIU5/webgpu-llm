---
title: WGSL Address Spaces
sidebar:
  order: 50
---

## Overview

Address spaces are a fundamental concept in the WebGPU Shading Language (WGSL) that define where and how memory is allocated and accessed in GPU programs. Unlike traditional CPU programming where memory is often treated as a unified resource, GPUs have multiple distinct memory regions, each optimized for different access patterns and use cases. Understanding WGSL's memory model and choosing the appropriate address space is crucial for writing efficient, correct shader code.

In WGSL, an address space represents a specific memory location where variables can be instantiated. Each address space has unique characteristics regarding:

- **Lifetime**: How long the memory allocation persists
- **Visibility**: Which shader invocations can access the memory
- **Access patterns**: Whether memory is read-only, write-only, or read-write
- **Performance**: Cache behavior and access speed characteristics
- **Size limits**: Maximum amount of memory available in each space

The GPU's parallel execution model means thousands of shader invocations may run simultaneously. Address spaces provide the memory organization needed to coordinate this massive parallelism efficiently. Some memory is private to each invocation, while other memory is shared across groups of invocations or all invocations in a shader stage.

WGSL defines six primary address spaces: **function**, **private**, **workgroup**, **uniform**, **storage**, and **handle**. Each serves a distinct purpose in the shader's memory architecture, and choosing the wrong address space can lead to compilation errors, performance problems, or incorrect results.

## Understanding Address Spaces

### What Are Address Spaces?

An address space is essentially a named region of memory with specific rules about allocation, access, and lifetime. When you declare a variable in WGSL, it's assigned to an address space either explicitly (through syntax like `var<uniform>`) or implicitly (based on where it's declared).

Think of address spaces as different "memory pools" within the GPU:

- Some pools are tiny but extremely fast (like workgroup shared memory)
- Some are large but slower (like storage buffers)
- Some are read-only but cached efficiently (like uniform buffers)
- Some are completely private to each shader invocation

### Why Address Spaces Exist

GPUs achieve massive parallelism by running thousands of threads simultaneously. This creates unique memory challenges:

1. **Coordination**: When multiple threads access the same memory, you need mechanisms to prevent data races and ensure consistency
2. **Performance**: Different memory types have vastly different access speeds and caching behaviors
3. **Resource constraints**: GPU memory is limited and must be carefully managed
4. **Access patterns**: Some workloads need read-only data (uniform across all threads), while others need per-thread scratch space

Address spaces make these trade-offs explicit. By declaring where data lives, you tell the GPU how to optimize memory access and how to handle coordination between threads.

### Comparison to CPU Memory Models

CPU programming typically presents a unified memory model where all RAM is accessed similarly (though modern CPUs have caches, registers, etc.). In contrast, GPU programming exposes different memory types explicitly:

| CPU Concept             | GPU/WGSL Equivalent     | Key Difference                       |
| ----------------------- | ----------------------- | ------------------------------------ |
| Local variables         | Function address space  | Very limited size (8KB)              |
| Static/global variables | Private address space   | Per-thread, not truly global         |
| Shared memory           | Workgroup address space | Explicitly shared, requires barriers |
| Constant data           | Uniform address space   | Read-only, efficiently cached        |
| Heap allocation         | Storage address space   | Large but requires explicit binding  |

On CPUs, you might casually allocate megabytes of stack space or heap memory. On GPUs, you're working with kilobytes of local memory and must carefully plan how to use larger storage buffers.

## Function Address Space

### Overview

The **function address space** is the default location for variables declared within shader functions. These are local variables with automatic storage duration—they exist only during the function's execution and are allocated independently for each invocation.

### Characteristics

- **Scope**: Local to individual function invocations
- **Lifetime**: Created when the function is entered, destroyed when it exits
- **Visibility**: Completely private to each invocation
- **Syntax**: Variables declared with `let` or `var` inside functions (no explicit address space annotation needed)
- **Size limit**: Combined byte size of all function-scoped variables cannot exceed **8,192 bytes** per function
- **Access speed**: Typically very fast, often allocated in registers or private cache

### When to Use

Use the function address space for:

- Temporary calculations within a function
- Loop counters and iterators
- Intermediate results
- Small working data that doesn't need to persist across function calls

### Examples and Use Cases

```wgsl
@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
    // All these variables are in the function address space
    let scale = 2.0;  // Immutable local variable
    var color = vec3f(0.0);  // Mutable local variable

    // Loop counter is function-scoped
    for (var i = 0u; i < 10u; i++) {
        let angle = f32(i) * 0.628;  // Local to this loop iteration
        color += vec3f(sin(angle), cos(angle), 0.5);
    }

    return vec4f(color * scale, 1.0);
}

fn computeDistance(a: vec3f, b: vec3f) -> f32 {
    let diff = a - b;  // Function address space
    let squaredDist = dot(diff, diff);  // Function address space
    return sqrt(squaredDist);
}
```

### Important Notes

The 8KB limit might seem small, but it's sufficient for most function-local computation. If you need more space, consider:

- Breaking the function into smaller functions (each gets its own 8KB budget)
- Moving large data structures to private or workgroup address spaces
- Using storage buffers for large datasets

## Private Address Space

### Overview

The **private address space** provides per-invocation storage that persists across function calls within a single shader invocation. Unlike function-local variables that disappear when a function returns, private variables maintain their values throughout the shader's execution.

### Characteristics

- **Scope**: Module-scope variables marked as private
- **Lifetime**: Exists for the entire shader invocation
- **Visibility**: Each invocation has its own independent copy
- **Syntax**: `var<private> variableName: Type;` at module scope
- **Size limit**: Combined byte size of all statically accessed private variables cannot exceed **8,192 bytes** per shader
- **Access speed**: Fast, similar to function address space

### Module-Scope Private Variables

Private variables are declared at module scope (outside any function) with the `<private>` annotation:

```wgsl
// Module-scope private variable
var<private> invocationCounter: u32 = 0u;
var<private> previousColor: vec3f;

@fragment
fn fragmentMain() -> @location(0) vec4f {
    invocationCounter += 1u;  // Persists across function calls
    previousColor = vec3f(1.0, 0.0, 0.0);
    return vec4f(previousColor, 1.0);
}
```

### When to Use Private vs Function

| Use Private When:                                        | Use Function When:                  |
| -------------------------------------------------------- | ----------------------------------- |
| Data needs to persist across function calls              | Data is temporary within a function |
| You need module-scope state                              | Local calculations only             |
| Sharing data between helper functions in same invocation | Simple intermediate values          |
| Building up results across multiple function calls       | Loop counters, temporary variables  |

### Examples and Use Cases

```wgsl
// Private state maintained across the shader execution
var<private> randomSeed: u32;
var<private> rayBounceCount: u32 = 0u;

// Initialize private state
fn initRandom(seed: u32) {
    randomSeed = seed;
}

// Uses and modifies private state
fn random() -> f32 {
    randomSeed = randomSeed * 747796405u + 2891336453u;
    var result = ((randomSeed >> ((randomSeed >> 28u) + 4u)) ^ randomSeed) * 277803737u;
    result = (result >> 22u) ^ result;
    return f32(result) / 4294967295.0;
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) id: vec3u) {
    initRandom(id.x + id.y * 1024u);

    // Each invocation has its own randomSeed and rayBounceCount
    for (var i = 0u; i < 10u; i++) {
        let r = random();  // Uses private randomSeed
        rayBounceCount += 1u;
    }
}
```

### Advanced Use Case: Multi-Pass Rendering State

```wgsl
var<private> geometryNormal: vec3f;
var<private> worldPosition: vec3f;
var<private> materialId: u32;

fn geometryPass(vertexData: VertexInput) {
    geometryNormal = calculateNormal(vertexData);
    worldPosition = vertexData.position;
    materialId = vertexData.matId;
}

fn lightingPass() -> vec3f {
    // Access data computed in geometryPass
    return calculateLighting(worldPosition, geometryNormal, materialId);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    geometryPass(input);
    let color = lightingPass();
    return vec4f(color, 1.0);
}
```

## Workgroup Address Space

### Overview

The **workgroup address space** is shared memory accessible by all invocations within a compute shader workgroup. This is one of the most powerful features for parallel computing, enabling efficient cooperation between threads running on the same GPU compute unit.

### Characteristics

- **Scope**: Shared among all invocations in a compute workgroup
- **Lifetime**: Allocated once per workgroup, exists for the workgroup's execution
- **Visibility**: All invocations in the same workgroup can read and write
- **Syntax**: `var<workgroup> variableName: Type;` (module scope)
- **Size limit**: Combined byte size cannot exceed **16,384 bytes** (16 KB) per shader
- **Access speed**: Very fast—typically on-chip SRAM
- **Valid in**: Compute shaders only
- **Synchronization**: Requires explicit barriers (`workgroupBarrier()`)

### Only Valid in Compute Shaders

Workgroup memory only makes sense in compute shaders because they execute in well-defined groups:

```wgsl
@compute @workgroup_size(256)  // 256 threads per workgroup
fn computeMain(@builtin(local_invocation_id) localId: vec3u,
               @builtin(workgroup_id) workgroupId: vec3u) {
    // Can use workgroup variables here
}
```

Vertex and fragment shaders don't have the same structured grouping, so workgroup memory isn't available.

### Synchronization Requirements

Since all invocations in a workgroup share workgroup memory, you must use `workgroupBarrier()` to synchronize access and prevent data races:

```wgsl
var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
    // Each thread writes to its own slot
    sharedData[idx] = f32(idx) * 2.0;

    // CRITICAL: Wait for all threads to finish writing
    workgroupBarrier();

    // Now safe to read from any slot
    let sum = sharedData[idx] + sharedData[(idx + 1u) % 256u];
}
```

**Without the barrier**, you'd have a data race—some threads might read before others finish writing.

### Size Limits and Planning

With only 16KB available, you must be strategic:

```wgsl
// Each f32 is 4 bytes, so 256 floats = 1KB
var<workgroup> cache: array<f32, 256>;

// A vec4f is 16 bytes, so 128 vectors = 2KB
var<workgroup> positions: array<vec4f, 128>;

// Total: 3KB of 16KB budget used
```

### Practical Examples

#### Example 1: Parallel Reduction (Sum)

```wgsl
var<workgroup> sharedSums: array<f32, 256>;

@compute @workgroup_size(256)
fn parallelSum(@builtin(local_invocation_index) idx: u32,
               @builtin(global_invocation_id) globalId: vec3u,
               @group(0) @binding(0) var<storage, read> input: array<f32>,
               @group(0) @binding(1) var<storage, read_write> output: array<f32>) {

    // Each thread loads one value into shared memory
    sharedSums[idx] = input[globalId.x];
    workgroupBarrier();

    // Parallel reduction tree
    var stride = 128u;
    while (stride > 0u) {
        if (idx < stride) {
            sharedSums[idx] += sharedSums[idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes the final sum
    if (idx == 0u) {
        output[workgroupId.x] = sharedSums[0];
    }
}
```

#### Example 2: Image Blur with Shared Memory Cache

```wgsl
var<workgroup> tileCache: array<vec4f, 18 * 18>;  // Slightly larger than workgroup

@compute @workgroup_size(16, 16)
fn blurImage(@builtin(local_invocation_id) localId: vec3u,
             @builtin(global_invocation_id) globalId: vec3u,
             @group(0) @binding(0) var inputTexture: texture_2d<f32>,
             @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>) {

    // Cooperatively load a tile into shared memory (including borders)
    let tileX = localId.x + 1u;
    let tileY = localId.y + 1u;
    tileCache[tileY * 18u + tileX] = textureLoad(inputTexture, globalId.xy, 0);

    // Load borders (some threads load extra pixels)
    if (localId.x == 0u) {
        tileCache[tileY * 18u] = textureLoad(inputTexture, globalId.xy - vec2u(1u, 0u), 0);
    }
    // ... similar for other borders

    workgroupBarrier();

    // Now compute blur using cached data (much faster than texture reads)
    var sum = vec4f(0.0);
    for (var dy = -1i; dy <= 1i; dy++) {
        for (var dx = -1i; dx <= 1i; dx++) {
            let tx = i32(tileX) + dx;
            let ty = i32(tileY) + dy;
            sum += tileCache[ty * 18 + tx];
        }
    }
    textureStore(outputTexture, globalId.xy, sum / 9.0);
}
```

## Uniform Address Space

### Overview

The **uniform address space** provides read-only data that's uniform (identical) across all shader invocations. This is ideal for constants, transformation matrices, camera parameters, and other data that every invocation needs but doesn't modify.

### Characteristics

- **Scope**: Bound to GPU buffer resources
- **Lifetime**: Managed by WebGPU API on the host side
- **Visibility**: All invocations in the shader stage see the same values
- **Syntax**: `@group(N) @binding(M) var<uniform> name: Type;`
- **Access**: Read-only from shader
- **Size**: Typically limited (64KB-256KB depending on hardware)
- **Performance**: Very fast due to efficient caching

### @group and @binding Requirements

Uniform variables must be bound to resources using `@group` and `@binding` attributes:

```wgsl
struct CameraUniforms {
    viewProjection: mat4x4f,
    cameraPosition: vec3f,
    time: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct MaterialUniforms {
    albedo: vec3f,
    roughness: f32,
    metallic: f32,
}

@group(1) @binding(0) var<uniform> material: MaterialUniforms;
```

The `@group` and `@binding` numbers correspond to how you bind buffers on the CPU side with WebGPU API.

### Best Use Cases

Uniform buffers are perfect for:

- **Transformation matrices**: View, projection, model transforms
- **Camera data**: Position, direction, FOV, clip planes
- **Lighting parameters**: Light positions, colors, intensities
- **Material properties**: Colors, roughness, metallic values
- **Time and animation**: Frame counters, elapsed time
- **Configuration**: Shader parameters, debug flags

### Complete Example

```wgsl
struct Transforms {
    model: mat4x4f,
    view: mat4x4f,
    projection: mat4x4f,
    normalMatrix: mat3x3f,
}

struct LightData {
    position: vec3f,
    _pad0: f32,  // Alignment padding
    color: vec3f,
    intensity: f32,
}

@group(0) @binding(0) var<uniform> transforms: Transforms;
@group(0) @binding(1) var<uniform> light: LightData;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldNormal: vec3f,
    @location(1) worldPosition: vec3f,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Use uniform data for transformations
    let worldPos = transforms.model * vec4f(input.position, 1.0);
    output.worldPosition = worldPos.xyz;
    output.position = transforms.projection * transforms.view * worldPos;
    output.worldNormal = transforms.normalMatrix * input.normal;

    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let normal = normalize(input.worldNormal);
    let lightDir = normalize(light.position - input.worldPosition);
    let diffuse = max(dot(normal, lightDir), 0.0);

    // Use uniform light data
    let color = light.color * light.intensity * diffuse;
    return vec4f(color, 1.0);
}
```

### Performance Considerations

Uniform buffers are heavily cached, making them extremely fast for read-only data accessed by all invocations. However:

- Keep uniforms relatively small (a few KB is ideal)
- Update uniforms only when necessary (not every frame if data doesn't change)
- Group related data in the same uniform buffer for better cache utilization

## Storage Address Space

### Overview

The **storage address space** provides access to large read-write GPU buffers. This is your main mechanism for working with substantial datasets in compute shaders and for writing results back to GPU memory.

### Characteristics

- **Scope**: Bound to GPU buffer resources
- **Lifetime**: Managed by WebGPU API
- **Visibility**: Shared across all invocations (with synchronization requirements)
- **Syntax**: `@group(N) @binding(M) var<storage, access_mode> name: Type;`
- **Size**: Large (hundreds of MB to several GB depending on hardware)
- **Access modes**: `read`, `read_write` (write-only not explicitly supported)

### Access Modes: read, write, read_write

Storage buffers support different access modes:

```wgsl
// Read-only storage buffer
@group(0) @binding(0) var<storage, read> inputData: array<f32>;

// Read-write storage buffer (default if not specified is 'read')
@group(0) @binding(1) var<storage, read_write> outputData: array<f32>;

// Equivalent to read_write (when access mode is omitted, 'read' is assumed)
@group(0) @binding(2) var<storage> debugData: array<u32>;
```

**Note**: By default, if you omit the access mode, it's `read`. You must explicitly specify `read_write` for writable buffers.

### Best Use Cases

Storage buffers are ideal for:

- **Large datasets**: Vertex buffers, particle systems, simulation data
- **Compute shader output**: Results from parallel computations
- **Indirect draw arguments**: GPU-driven rendering
- **Structured data**: Arrays of structures, graph data, spatial data structures
- **Accumulation buffers**: Building up results across multiple shader invocations

### Practical Examples

#### Example 1: Particle Simulation

```wgsl
struct Particle {
    position: vec3f,
    velocity: vec3f,
    mass: f32,
    _pad: f32,  // Alignment
}

struct SimParams {
    deltaTime: f32,
    damping: f32,
    gravity: vec3f,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    var particle = particles[index];

    // Apply forces
    particle.velocity += params.gravity * params.deltaTime;
    particle.velocity *= params.damping;

    // Update position
    particle.position += particle.velocity * params.deltaTime;

    // Write back to storage
    particles[index] = particle;
}
```

#### Example 2: Histogram Computation with Atomics

```wgsl
@group(0) @binding(0) var<storage, read> imageData: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn computeHistogram(@builtin(global_invocation_id) id: vec3u) {
    let pixelIndex = id.x;
    if (pixelIndex >= arrayLength(&imageData)) {
        return;
    }

    let pixelValue = imageData[pixelIndex];
    let binIndex = pixelValue % 256u;

    // Atomic add prevents race conditions
    atomicAdd(&histogram[binIndex], 1u);
}
```

#### Example 3: Prefix Sum (Scan)

```wgsl
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn prefixSum(@builtin(local_invocation_index) localIdx: u32,
             @builtin(global_invocation_id) globalId: vec3u) {

    let idx = globalId.x;

    // Load into shared memory
    temp[2u * localIdx] = select(0u, input[2u * idx], 2u * idx < arrayLength(&input));
    temp[2u * localIdx + 1u] = select(0u, input[2u * idx + 1u], 2u * idx + 1u < arrayLength(&input));
    workgroupBarrier();

    // Up-sweep phase
    var offset = 1u;
    for (var d = 256u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (localIdx < d) {
            let ai = offset * (2u * localIdx + 1u) - 1u;
            let bi = offset * (2u * localIdx + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        offset = offset << 1u;
    }

    // Clear the last element
    if (localIdx == 0u) {
        temp[511] = 0u;
    }

    // Down-sweep phase
    for (var d = 1u; d < 512u; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if (localIdx < d) {
            let ai = offset * (2u * localIdx + 1u) - 1u;
            let bi = offset * (2u * localIdx + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Write results
    if (2u * idx < arrayLength(&output)) {
        output[2u * idx] = temp[2u * localIdx];
    }
    if (2u * idx + 1u < arrayLength(&output)) {
        output[2u * idx + 1u] = temp[2u * localIdx + 1u];
    }
}
```

### Synchronization Requirements

When multiple invocations write to the same storage buffer:

- Use **atomic operations** for simple increments/adds (e.g., counters, histograms)
- Use **storageBarrier()** in compute shaders to ensure memory ordering
- Design algorithms to minimize write conflicts (each thread writes to different locations)

## Pointer Types

### Overview

WGSL supports pointer types that enable forming references to values stored in memory. Pointers allow you to pass variables by reference, create aliases, and work with complex data structures more efficiently.

### Syntax

The pointer type syntax is:

```wgsl
ptr<address_space, T, access_mode>
```

Where:

- `address_space`: The address space (function, private, workgroup, storage, uniform)
- `T`: The type being pointed to
- `access_mode`: Optional access mode (for storage: read or read_write)

### Valid Pointer Usage Patterns

#### Passing by Reference

```wgsl
fn modifyValue(ptr: ptr<function, f32>) {
    *ptr = *ptr * 2.0;
}

fn example() {
    var value = 5.0;
    modifyValue(&value);
    // value is now 10.0
}
```

#### Working with Storage Buffers

```wgsl
@group(0) @binding(0) var<storage, read_write> data: array<vec4f>;

fn processElement(ptr: ptr<storage, vec4f, read_write>) {
    (*ptr).x += 1.0;
    (*ptr).y *= 2.0;
}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) id: vec3u) {
    processElement(&data[id.x]);
}
```

#### Array Element Pointers

```wgsl
var<private> myArray: array<f32, 10>;

fn sumArray() -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 10u; i++) {
        let elementPtr = &myArray[i];
        sum += *elementPtr;
    }
    return sum;
}
```

### Restrictions on Pointers

WGSL pointers have important restrictions compared to traditional C/C++ pointers:

1. **No arbitrary pointer arithmetic**: You cannot add/subtract offsets to pointers
2. **Must originate from declared variables**: Cannot create pointers to arbitrary memory addresses
3. **Cannot be stored**: Pointers cannot be stored in variables that outlive the pointer's scope
4. **Address space matching**: Cannot convert pointers between incompatible address spaces
5. **No null pointers**: All pointers must be valid references

#### Valid Pointer Formation

```wgsl
// Valid: Taking address of a variable
var x = 5.0;
let ptr = &x;

// Valid: Taking address of array element
var arr: array<f32, 4>;
let elemPtr = &arr[2];

// Valid: Taking address of struct member
struct Data { value: f32, }
var data: Data;
let memberPtr = &data.value;
```

#### Invalid Pointer Operations

```wgsl
// Invalid: Cannot do pointer arithmetic
// let nextPtr = ptr + 1;  // Compilation error

// Invalid: Cannot store pointers in structs
// struct Container { ptr: ptr<function, f32>, }  // Error

// Invalid: Cannot cast between address spaces
// let storagePtr: ptr<storage, f32> = /* ... */;
// let privatePtr: ptr<private, f32> = storagePtr;  // Error
```

### Dereferencing

Use the `*` operator to dereference a pointer:

```wgsl
fn example() {
    var value = 10.0;
    let ptr = &value;

    // Read through pointer
    let x = *ptr;  // x = 10.0

    // Write through pointer
    *ptr = 20.0;  // value is now 20.0
}
```

## Performance Implications

### Cache Behavior Differences

Different address spaces have vastly different performance characteristics:

| Address Space | Typical Location      | Access Speed                  | Cache Behavior                      |
| ------------- | --------------------- | ----------------------------- | ----------------------------------- |
| Function      | Registers / L1 cache  | Fastest (~1 cycle)            | Not applicable (too small)          |
| Private       | L1 cache              | Very fast (~4-10 cycles)      | Minimal, per-invocation             |
| Workgroup     | On-chip SRAM          | Fast (~20-40 cycles)          | Shared among workgroup              |
| Uniform       | VRAM + constant cache | Fast (cached: ~40 cycles)     | Highly efficient for uniform access |
| Storage       | VRAM                  | Slow (uncached: ~400+ cycles) | Coalesced reads help                |

### Access Speed Considerations

**Function and Private**: These are the fastest because they're typically allocated in registers or L1 cache close to the compute units. Use them for hot paths and frequently accessed data.

**Workgroup**: Shared memory is fast but limited. It's ideal when multiple threads need to cooperate on a small dataset. The latency is low, but you must synchronize access carefully.

**Uniform**: Read-only uniform data benefits from specialized constant caches. All threads reading the same value is extremely efficient (broadcast). However, if threads read different locations ("divergent" access), performance degrades.

**Storage**: Large but slowest. Storage buffer access goes to main VRAM. Performance depends heavily on:

- **Coalesced access**: Adjacent threads accessing adjacent memory locations
- **Access patterns**: Sequential reads are much faster than random access
- **Bank conflicts**: Can occur with certain access patterns

### Choosing the Right Address Space

Follow this decision process:

```
Is the data constant for all invocations?
├─ Yes → Use UNIFORM address space
└─ No → Continue

Is the data large (> 8KB)?
├─ Yes → Use STORAGE address space
└─ No → Continue

Does the data need to be shared between invocations in a workgroup?
├─ Yes → Use WORKGROUP address space
└─ No → Continue

Does the data need to persist across function calls?
├─ Yes → Use PRIVATE address space
└─ No → Use FUNCTION address space
```

### Optimization Tips

1. **Minimize storage buffer access**: Cache frequently accessed values in function/private variables

   ```wgsl
   // Bad: Read from storage multiple times
   let result = data[id.x] * 2.0 + data[id.x] * 3.0;

   // Good: Cache in local variable
   let value = data[id.x];
   let result = value * 2.0 + value * 3.0;
   ```

2. **Use workgroup memory for data reuse**: If multiple threads need the same data, load it once into workgroup memory

   ```wgsl
   // Load once into shared memory
   var<workgroup> cached: array<f32, 256>;
   cached[localIdx] = inputBuffer[globalIdx];
   workgroupBarrier();

   // All threads can now access quickly
   let value = cached[someIndex];
   ```

3. **Coalesce storage buffer access**: Structure your data so adjacent threads access adjacent memory

   ```wgsl
   // Good: Coalesced access (adjacent threads, adjacent memory)
   let value = data[globalId.x];

   // Bad: Strided access (poor cache utilization)
   let value = data[globalId.x * 1000u];
   ```

4. **Keep workgroup size appropriate**: Larger workgroups can share more data, but too large reduces occupancy

## Best Practices

### Decision Flowchart for Choosing Address Spaces

```
START: I need to store data in my shader

│
├─ Is it constant across all invocations?
│  └─ YES → UNIFORM
│      Examples: Camera matrices, light data, material properties
│      Size: Keep small (< 64KB)
│      Access: Read-only
│
├─ Is it larger than 16KB?
│  └─ YES → STORAGE
│      Examples: Vertex buffers, particle arrays, large datasets
│      Size: Can be very large (GB)
│      Access: Read or read-write
│      Note: Use atomics for concurrent writes
│
├─ Compute shader AND shared between threads in workgroup?
│  └─ YES → WORKGROUP
│      Examples: Tile caches, reduction trees, cooperative algorithms
│      Size: Limited (< 16KB)
│      Access: Requires barriers
│      Note: Only in compute shaders
│
├─ Needs to persist across function calls?
│  └─ YES → PRIVATE
│      Examples: Random state, accumulated results, per-thread state
│      Size: Limited (< 8KB per shader)
│      Access: Fast, per-invocation
│
└─ Temporary calculation?
   └─ YES → FUNCTION
       Examples: Loop counters, intermediate values, local calculations
       Size: Limited (< 8KB per function)
       Access: Fastest, truly local
```

### Practical Guidelines

**For Vertex Shaders:**

- Input attributes are read from storage implicitly
- Use uniform for transformation matrices
- Use private for per-vertex state if needed across helper functions
- Use function for temporary calculations

**For Fragment Shaders:**

- Use uniform for material properties, lighting data
- Use storage for large lookup tables (if needed)
- Use private for accumulated values (rare)
- Use function for color calculations, texture coordinate manipulation

**For Compute Shaders:**

- Use storage for input/output data (main data pipeline)
- Use uniform for algorithm parameters
- Use workgroup for cooperative algorithms (reductions, scans, tile caching)
- Use private for per-thread state (random seeds, counters)
- Use function for temporary calculations

### Memory Budget Planning

Track your usage carefully:

```wgsl
// Example memory budget calculation

// Function space: ~200 bytes per function
fn myFunction() {
    var tempArray: array<vec4f, 10>;  // 10 * 16 = 160 bytes
    var counter: u32;  // 4 bytes
    var result: vec3f;  // 12 bytes
    // Total: 176 bytes (within 8KB limit)
}

// Private space: 500 bytes
var<private> randomState: u32;  // 4 bytes
var<private> hitRecord: array<vec3f, 32>;  // 32 * 12 = 384 bytes
var<private> rayCount: u32;  // 4 bytes
// Total: 392 bytes (within 8KB limit)

// Workgroup space: 8KB
var<workgroup> tileCache: array<vec4f, 512>;  // 512 * 16 = 8192 bytes
// Total: 8KB (within 16KB limit)
```

## Common Pitfalls

### 1. Address Space Mismatches

**Problem**: Trying to assign or pass values between incompatible address spaces

```wgsl
// Error: Cannot assign storage pointer to function pointer
@group(0) @binding(0) var<storage> data: array<f32>;

fn processData(ptr: ptr<function, f32>) {  // Expects function pointer
    *ptr = 1.0;
}

@compute @workgroup_size(64)
fn computeMain() {
    processData(&data[0]);  // ERROR: address space mismatch
}
```

**Solution**: Match address spaces or pass by value

```wgsl
// Correct: Pass by value
fn processData(value: f32) -> f32 {
    return value * 2.0;
}

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) id: vec3u) {
    let result = processData(data[id.x]);
    data[id.x] = result;
}
```

### 2. Invalid Pointer Usage

**Problem**: Storing pointers beyond their valid scope

```wgsl
// Error: Cannot store pointer in a struct
struct Container {
    ptr: ptr<function, f32>,  // INVALID
}
```

**Solution**: Only use pointers as function parameters or immediate references

```wgsl
// Correct: Use pointers only in function parameters
fn updateValue(ptr: ptr<function, f32>, newValue: f32) {
    *ptr = newValue;
}
```

### 3. Forgetting Synchronization

**Problem**: Reading workgroup memory without proper barriers

```wgsl
var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
    shared[idx] = f32(idx);

    // BUG: No barrier! Reading while others are still writing
    let value = shared[(idx + 1u) % 256u];  // RACE CONDITION
}
```

**Solution**: Always use workgroupBarrier() before reading shared data

```wgsl
var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32) {
    shared[idx] = f32(idx);

    workgroupBarrier();  // Wait for all writes to complete

    let value = shared[(idx + 1u) % 256u];  // Safe
}
```

### 4. Exceeding Size Limits

**Problem**: Declaring too much function or private memory

```wgsl
fn processData() {
    // Error: Exceeds 8KB limit
    var hugeArray: array<mat4x4f, 200>;  // 200 * 64 = 12,800 bytes
}
```

**Solution**: Use storage buffers for large data

```wgsl
@group(0) @binding(0) var<storage, read_write> largeData: array<mat4x4f>;

@compute @workgroup_size(64)
fn processData(@builtin(global_invocation_id) id: vec3u) {
    let matrix = largeData[id.x];
    // Process matrix...
}
```

### 5. Non-Coalesced Storage Access

**Problem**: Poor memory access patterns causing slow performance

```wgsl
// Bad: Strided access pattern
@compute @workgroup_size(64)
fn badAccess(@builtin(global_invocation_id) id: vec3u) {
    // Adjacent threads access memory 1000 elements apart
    let value = data[id.x * 1000u];
}
```

**Solution**: Structure data for coalesced access

```wgsl
// Good: Sequential access pattern
@compute @workgroup_size(64)
fn goodAccess(@builtin(global_invocation_id) id: vec3u) {
    // Adjacent threads access adjacent memory
    let value = data[id.x];
}
```

### 6. Forgetting Read-Write Access Mode

**Problem**: Trying to write to a read-only storage buffer

```wgsl
@group(0) @binding(0) var<storage> data: array<f32>;  // Default is 'read'

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) id: vec3u) {
    data[id.x] = 1.0;  // ERROR: Cannot write to read-only storage
}
```

**Solution**: Explicitly specify read_write access mode

```wgsl
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) id: vec3u) {
    data[id.x] = 1.0;  // Correct
}
```

---

## Conclusion

Understanding WGSL address spaces is fundamental to writing efficient, correct GPU programs. Each address space serves a specific purpose in the GPU's memory hierarchy:

- **Function**: Fast, temporary, function-local variables
- **Private**: Per-invocation state that persists across function calls
- **Workgroup**: Shared memory for cooperation within compute workgroups
- **Uniform**: Read-only constants shared across all invocations
- **Storage**: Large read-write buffers for main data processing

By choosing the appropriate address space for each use case, considering size limits, understanding synchronization requirements, and following best practices, you can write shaders that are both performant and correct. Always remember to profile your shaders and measure the impact of different memory access patterns—the GPU's parallel architecture rewards careful attention to memory organization.
