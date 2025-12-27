---
title: WGSL Address Spaces
sidebar:
  order: 50
---

## Overview

WGSL defines five address spaces representing distinct memory regions with different performance, visibility, and access characteristics. Each address space maps to specific hardware memory types and determines how GPU cores read and write data.

Understanding address spaces is essential for writing efficient GPU code—choosing the right address space affects performance by orders of magnitude.

## function Address Space

The `function` address space stores local variables that exist only during function execution, analogous to stack memory in CPU programming.

```wgsl
fn compute_normal(pos: vec3f) -> vec3f {
    var temp_vec: vec3f = pos;       // Implicitly function address space
    var result: vec3f;

    temp_vec = normalize(temp_vec);
    result = temp_vec * 2.0 - 1.0;

    return result;
}
```

Characteristics:
- **Automatic lifetime**: Allocated on function entry, deallocated on return
- **Private to invocation**: Each shader thread has its own copy
- **Fast access**: Stored in registers or L1 cache
- **Default address space**: `var` without explicit address space uses `function`
- **Size limit**: 8KB per function

Large arrays in function space may spill to slower global memory. Keep function-local arrays small; use storage buffers for large datasets.

## private Address Space

The `private` address space stores module-scope variables that persist for the entire shader execution, with each invocation getting its own copy.

```wgsl
var<private> invocation_counter: u32 = 0u;
var<private> accumulated_color: vec4f;

@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
    invocation_counter += 1u;
    accumulated_color += texture_sample(uv);
    return accumulated_color;
}
```

Characteristics:
- **Module scope**: Declared outside functions
- **Per-invocation isolation**: Each invocation has independent copies
- **Mutable**: Can be read and written throughout shader execution
- **Persistence**: Values persist across function calls within the same invocation
- **Size limit**: 8KB per shader

Use `private` for accumulators, caching expensive computations, and state that persists within a single invocation.

## workgroup Address Space

The `workgroup` address space provides shared memory visible to all invocations within a compute shader workgroup. This is critical for parallel algorithms requiring thread cooperation.

```wgsl
var<workgroup> shared_data: array<f32, 256>;
var<workgroup> group_sum: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local_idx: u32) {
    // Each thread loads one element
    shared_data[local_idx] = input_buffer[local_idx];

    // Barrier ensures all loads complete before any thread proceeds
    workgroupBarrier();

    // Now all threads can safely read any element
    let neighbor = shared_data[(local_idx + 1u) % 256u];
}
```

Characteristics:
- **Compute shaders only**: Not available in vertex or fragment shaders
- **Workgroup scope**: Visible to all invocations in the same workgroup
- **Fast access**: Backed by on-chip shared memory (scratchpad)
- **Synchronization required**: Use `workgroupBarrier()` to ensure visibility
- **Size limit**: 16KB typically

### Synchronization

`workgroupBarrier()` ensures all workgroup invocations reach the barrier before any proceed:

```wgsl
var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local_id: vec3u) {
    let idx = local_id.x + local_id.y * 16u;

    // Phase 1: All threads load
    tile[idx] = compute_value(local_id);
    workgroupBarrier();

    // Phase 2: All threads process
    let sum = tile[idx] + tile[(idx + 1u) % 256u];
    workgroupBarrier();

    // Phase 3: Write results
    output[idx] = sum;
}
```

Missing or misplaced barriers cause race conditions. Always barrier before reading data written by other threads.

### Atomics in Workgroup Memory

```wgsl
var<workgroup> counter: atomic<u32>;

@compute @workgroup_size(64)
fn count_positives(
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3u
) {
    if (local_idx == 0u) {
        atomicStore(&counter, 0u);
    }
    workgroupBarrier();

    let global_idx = wg_id.x * 64u + local_idx;
    if (data[global_idx] > 0.0) {
        atomicAdd(&counter, 1u);
    }
    workgroupBarrier();

    if (local_idx == 0u) {
        result[wg_id.x] = atomicLoad(&counter);
    }
}
```

## uniform Address Space

The `uniform` address space stores read-only data shared across all invocations. The GPU broadcasts uniform data efficiently, making it ideal for transformation matrices and material properties.

```wgsl
struct CameraUniforms {
    view_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    camera_position: vec3f,
    near_plane: f32,
    far_plane: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@vertex
fn vertex_main(@location(0) position: vec3f) -> @builtin(position) vec4f {
    let world_pos = vec4f(position, 1.0);
    let view_pos = camera.view_matrix * world_pos;
    return camera.projection_matrix * view_pos;
}
```

Characteristics:
- **Read-only**: Cannot be written in shaders
- **Uniform access**: All invocations see the same data
- **Broadcast optimization**: GPU broadcasts values efficiently
- **Size limit**: 16-64KB per binding typically

### When to Use Uniform vs Storage

| Aspect | uniform | storage |
|--------|---------|---------|
| Access | Read-only | Read-only or read-write |
| Size limit | ~64KB typical | Much larger (GB) |
| Update frequency | Per-draw updates | Large datasets |
| Access pattern | Same value all threads | Different elements per thread |

## storage Address Space

The `storage` address space provides access to large buffers with optional write capability. Storage buffers can hold gigabytes of data and support per-element random access.

```wgsl
struct Particle {
    position: vec3f,
    velocity: vec3f,
    lifetime: f32,
}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;

@compute @workgroup_size(64)
fn update_particles(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;
    var p = particles_in[idx];

    p.position += p.velocity * delta_time;
    p.lifetime -= delta_time;

    particles_out[idx] = p;
}
```

### Access Modes

- `read`: Read-only access, enables optimizations
- `read_write`: Full read and write access

If access mode is omitted, `read` is assumed. Specify `read_write` explicitly for writable buffers.

### Runtime-Sized Arrays

Storage buffers support runtime-sized arrays as the last struct member:

```wgsl
struct ParticleBuffer {
    count: u32,
    _padding: u32,
    _padding2: u32,
    _padding3: u32,
    particles: array<Particle>  // Size determined at runtime
}

@group(0) @binding(0) var<storage, read> data: ParticleBuffer;

@compute @workgroup_size(64)
fn process(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= data.count) { return; }
    let p = data.particles[id.x];
}
```

### Atomics in Storage

Storage buffers support atomic operations for cross-workgroup synchronization:

```wgsl
@group(0) @binding(0) var<storage, read_write> global_counter: atomic<u32>;

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3u) {
    if (condition) {
        atomicAdd(&global_counter, 1u);
    }
}
```

## Address Space Comparison

| Address Space | Scope | Visibility | Access | Speed | Size |
|--------------|-------|------------|--------|-------|------|
| `function` | Function | Invocation | Read/Write | Fastest | 8KB |
| `private` | Module | Invocation | Read/Write | Very Fast | 8KB |
| `workgroup` | Workgroup | Workgroup | Read/Write | Fast | 16KB |
| `uniform` | Buffer | All invocations | Read | Fast | 16-64KB |
| `storage` | Buffer | All invocations | Read or R/W | Variable | Large |

## Memory Access Patterns

### Coalesced Access

GPUs read memory in bursts. Sequential access patterns are much faster than random access:

```wgsl
@compute @workgroup_size(64)
fn coalesced_access(@builtin(global_invocation_id) id: vec3u) {
    // Good: consecutive threads access consecutive elements
    let value = input[id.x];
}

@compute @workgroup_size(64)
fn strided_access(@builtin(global_invocation_id) id: vec3u) {
    // Bad: strided access wastes bandwidth
    let value = input[id.x * 8u];
}
```

### Using Workgroup Memory for Irregular Access

```wgsl
var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn blur_kernel(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u
) {
    // Coalesced load into shared memory
    let local_idx = local_id.x + local_id.y * 16u;
    tile[local_idx] = input[global_id.x + global_id.y * width];
    workgroupBarrier();

    // Random access within fast workgroup memory
    var sum: f32 = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor_idx = (local_id.x + dx) + (local_id.y + dy) * 16u;
            sum += tile[neighbor_idx];
        }
    }

    output[global_id.x + global_id.y * width] = sum / 9.0;
}
```

## Choosing an Address Space

Use this decision process:

1. **Constant for all invocations?** → `uniform`
2. **Large data (> 16KB)?** → `storage`
3. **Compute shader, shared between workgroup threads?** → `workgroup`
4. **Persists across function calls?** → `private`
5. **Temporary calculation?** → `function`

### Optimization Guidelines

Cache storage reads in local variables:

```wgsl
// Suboptimal: multiple reads
let result = data[id.x] * 2.0 + data[id.x] * 3.0;

// Better: cache in local variable
let value = data[id.x];
let result = value * 2.0 + value * 3.0;
```

Use workgroup memory for data reuse:

```wgsl
var<workgroup> cached: array<f32, 256>;
cached[local_idx] = input_buffer[global_idx];
workgroupBarrier();
let value = cached[some_index];  // Fast access
```

## TypeGPU Address Space Mapping

TypeGPU abstracts address spaces through buffer usage declarations:

```typescript
import tgpu from "typegpu";
import { arrayOf, f32, struct, vec3f } from "typegpu/data";

// Uniform buffer (uniform address space)
const CameraSchema = struct({
  viewMatrix: mat4x4f,
  projMatrix: mat4x4f,
});
const camera = root.createBuffer(CameraSchema, data).$usage("uniform");

// Storage buffer read-only (storage, read)
const particles = root.createBuffer(arrayOf(ParticleSchema, 1000)).$usage("storage");

// Storage buffer read-write (storage, read_write)
const output = root.createBuffer(arrayOf(f32, 1000)).$usage("storage");
```

TypeGPU infers the appropriate WGSL address space from the usage pattern.

---

Address spaces are fundamental to GPU performance. Use `function` and `private` for per-invocation data, `workgroup` for inter-thread cooperation in compute shaders, `uniform` for shared constants, and `storage` for large datasets. Choose based on data size, access patterns, and sharing requirements.
