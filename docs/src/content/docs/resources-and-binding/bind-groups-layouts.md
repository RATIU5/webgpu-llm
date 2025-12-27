---
title: Bind Groups and Layouts
sidebar:
  order: 10
---

## Overview

Bind groups and layouts form the cornerstone of WebGPU's resource binding model, providing a structured way to connect GPU resources to shader programs.

:::note[Two-Level Hierarchy]
- **Bind group layouts** define the schema—what types of resources a shader expects
- **Bind groups** contain the actual resource instances matching that schema

This separation enables flexible resource management, efficient validation, and swapping resource sets while maintaining pipeline configuration.
:::

## Key Concepts

### Bind Group

A collection of GPU resources (buffers, textures, samplers) bundled as a single unit:

```javascript title="Conceptual bind group"
// Instead of binding resources individually:
bindResource(0, uniformBuffer);
bindResource(1, storageBuffer);
bindResource(2, texture);

// Bind them together:
passEncoder.setBindGroup(0, myBindGroup);
```

### Bind Group Layout

Defines the structure bind groups must follow:
- Number of bindings
- Resource type at each binding (buffer, texture, sampler)
- Shader stages that can access each resource
- Buffer types, texture formats, access modes

### WGSL Binding Indices

Resources are referenced using two attributes:

```wgsl title="WGSL resource declarations"
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec3f>;
@group(1) @binding(0) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(1) var textureSampler: sampler;
```

:::tip[Organization Strategy]
Group 0: Per-frame data (rarely changes)
Group 1: Per-material data (changes between materials)
Group 2: Per-object data (changes every draw call)
:::

### Shader Stage Visibility

| Stage | Flag | Use Case |
|-------|------|----------|
| Vertex | `GPUShaderStage.VERTEX` | Transformations, vertex data |
| Fragment | `GPUShaderStage.FRAGMENT` | Textures, lighting |
| Compute | `GPUShaderStage.COMPUTE` | General computation |

Combine with bitwise OR: `GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT`

## Creating Bind Group Layouts

```javascript title="Compute bind group layout" {5-6,11-12,17-18}
const bindGroupLayout = device.createBindGroupLayout({
  label: "Compute Bind Group Layout",
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "uniform",
        minBindingSize: 16,
      },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage",
      },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage", // Read-write
      },
    },
  ],
});
```

### Entry Types

<details>
<summary>**Buffer Bindings**</summary>

```javascript title="Buffer entry definition"
{
  binding: 0,
  visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
  buffer: {
    type: "uniform" | "storage" | "read-only-storage",
    hasDynamicOffset: false,
    minBindingSize: 0
  }
}
```

| Type | WGSL | Description |
|------|------|-------------|
| `uniform` | `var<uniform>` | Read-only, optimized for small data |
| `storage` | `var<storage, read_write>` | Read-write access |
| `read-only-storage` | `var<storage, read>` | Read-only storage |

</details>

<details>
<summary>**Sampler Bindings**</summary>

```javascript title="Sampler entry definition"
{
  binding: 1,
  visibility: GPUShaderStage.FRAGMENT,
  sampler: {
    type: "filtering" | "non-filtering" | "comparison"
  }
}
```

| Type | Use Case |
|------|----------|
| `filtering` | Standard texture sampling with interpolation |
| `non-filtering` | Exact texel access |
| `comparison` | Shadow mapping depth comparison |

</details>

<details>
<summary>**Texture Bindings**</summary>

```javascript title="Texture entry definition"
{
  binding: 2,
  visibility: GPUShaderStage.FRAGMENT,
  texture: {
    sampleType: "float" | "unfilterable-float" | "depth" | "sint" | "uint",
    viewDimension: "2d" | "3d" | "cube" | "2d-array",
    multisampled: false
  }
}
```

</details>

<details>
<summary>**Storage Texture Bindings**</summary>

```javascript title="Storage texture entry definition"
{
  binding: 3,
  visibility: GPUShaderStage.COMPUTE,
  storageTexture: {
    access: "write-only" | "read-only" | "read-write",
    format: "rgba8unorm",
    viewDimension: "2d"
  }
}
```

:::caution
Storage textures require explicit format specification.
:::

</details>

## Creating Bind Groups

```javascript title="Create bind group from layout" {2,5-9}
const bindGroup = device.createBindGroup({
  label: "Compute Bind Group",
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: {
        buffer: uniformBuffer,
        offset: 0,
        size: 16,
      },
    },
    {
      binding: 1,
      resource: { buffer: inputBuffer },
    },
    {
      binding: 2,
      resource: { buffer: outputBuffer },
    },
  ],
});
```

For textures and samplers:

```javascript title="Texture and sampler bind group"
const renderBindGroup = device.createBindGroup({
  label: "Render Bind Group",
  layout: renderBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: sampler, // GPUSampler object
    },
    {
      binding: 1,
      resource: texture.createView(), // GPUTextureView
    },
  ],
});
```

:::danger[Validation Requirements]
Entries must exactly match the layout:
- Same binding numbers
- Correct resource types
- All bindings provided
:::

## Pipeline Layouts

Combines multiple bind group layouts for a complete pipeline interface:

```javascript title="Create pipeline layout"
const pipelineLayout = device.createPipelineLayout({
  label: "Main Pipeline Layout",
  bindGroupLayouts: [
    bindGroupLayout0, // @group(0)
    bindGroupLayout1, // @group(1)
    bindGroupLayout2, // @group(2)
  ],
});

const computePipeline = device.createComputePipeline({
  label: "Compute Pipeline",
  layout: pipelineLayout,
  compute: {
    module: shaderModule,
    entryPoint: "main",
  },
});
```

### Automatic Layout

Use `layout: "auto"` to infer layout from shader:

```javascript title="Auto layout pipeline"
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: shaderModule,
    entryPoint: "main",
  },
});

// Retrieve inferred layout for bind group creation
const bindGroupLayout = pipeline.getBindGroupLayout(0);
```

:::tip
Explicit layouts enable sharing between pipelines and pre-compilation before resources are ready.
:::

## TypeGPU Bind Groups

TypeGPU provides type-safe abstractions, eliminating manual index management.

### Type-safe Layouts

```typescript title="TypeGPU bind group layout"
import { tgpu, d } from "typegpu";

const UniformsSchema = d.struct({
  viewProjection: d.mat4x4f,
  time: d.f32,
  lightDirection: d.vec3f,
});

const sceneLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: UniformsSchema },
  instanceData: { storage: d.arrayOf(d.mat4x4f) },
});
```

:::note[TypeGPU Benefits]
- **Named properties** instead of numeric indices
- **Automatic index assignment** in declaration order
- **Type inference** with autocomplete and compile-time errors
:::

### Resource Types in TypeGPU

```typescript title="TypeGPU resource type syntax"
const layout = tgpu.bindGroupLayout({
  // Uniform buffer (read-only)
  uniforms: { uniform: d.struct({ value: d.f32 }) },

  // Storage buffer (read-only by default)
  readData: { storage: d.arrayOf(d.f32) },

  // Mutable storage buffer
  writeData: { storage: d.f32, access: "mutable" },

  // Filtering sampler
  linearSampler: { sampler: "filtering" },

  // Texture
  diffuseTexture: { texture: d.tex2d<"f32"> },

  // Storage texture
  outputTexture: { storageTexture: d.storageTex2d<"rgba8unorm"> },
});
```

### Creating TypeGPU Bind Groups

```typescript title="Type-safe bind group creation"
const uniformBuffer = root.createBuffer(UniformsSchema, {
  viewProjection: mat4.identity(),
  time: 0,
  lightDirection: vec3(0, -1, 0),
});

const storageBuffer = root.createBuffer(
  d.arrayOf(d.f32),
  new Float32Array(1000)
);

const bindGroup = root.createBindGroup(sceneLayout, {
  uniforms: uniformBuffer,
  instanceData: storageBuffer,
});
```

## Dynamic Binding

Dynamic offsets allow using different buffer sections without creating multiple bind groups.

### Enable Dynamic Offsets

```javascript title="Layout with dynamic offset" {7-8}
const layout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: "uniform",
        hasDynamicOffset: true,
        minBindingSize: 256,
      },
    },
  ],
});
```

### Use Dynamic Offsets

```javascript title="Render loop with dynamic offsets" {5}
const passEncoder = commandEncoder.beginRenderPass(/* ... */);
passEncoder.setPipeline(pipeline);

for (let i = 0; i < objectCount; i++) {
  const offset = i * 256; // Must be 256-byte aligned
  passEncoder.setBindGroup(0, bindGroup, [offset]);
  passEncoder.draw(vertexCount, 1, 0, 0);
}

passEncoder.end();
```

:::caution[Alignment Requirements]
- **Uniform buffers**: 256-byte alignment
- **Storage buffers**: 32-byte alignment

```javascript
// Calculate aligned offset
const alignedOffset = Math.ceil(dataSize / 256) * 256;
```
:::

## Resource Visibility

### Visibility Patterns

| Pattern | Visibility | Use Case |
|---------|------------|----------|
| Per-frame data | `VERTEX \| FRAGMENT` | Camera, time, global settings |
| Transformations | `VERTEX` | Model matrices, bone data |
| Materials | `FRAGMENT` | Textures, lighting parameters |
| Compute-only | `COMPUTE` | Storage buffers, dispatch data |

:::tip[Principle of Least Privilege]
Only grant access to stages that need the resource. This helps drivers optimize and catches shader errors early.

```javascript
{
  binding: 0,
  visibility: GPUShaderStage.VERTEX, // Not VERTEX | FRAGMENT
  buffer: { type: "uniform" }
}
```
:::

## Efficient Binding

### Group by Update Frequency

```javascript title="Organized pipeline layout"
// Group 0: Per-frame (updated once per frame)
const frameLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform" } }, // Camera, time
  ],
});

// Group 1: Per-material (updated when material changes)
const materialLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT,
      sampler: { type: "filtering" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType: "float" } },
  ],
});

// Group 2: Per-object (updated every draw call)
const objectLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX,
      buffer: { type: "uniform", hasDynamicOffset: true } },
  ],
});
```

### Efficient Render Loop

```javascript title="Minimizing bind group switches" {2,5,9}
const frameBindGroup = /* created once per frame */;
passEncoder.setBindGroup(0, frameBindGroup);

for (const material of materials) {
  passEncoder.setBindGroup(1, material.bindGroup);

  for (const object of objectsWithMaterial(material)) {
    // Use dynamic offset instead of new bind group
    passEncoder.setBindGroup(2, objectBindGroup, [object.uniformOffset]);
    passEncoder.draw(/* ... */);
  }
}
```

### Share Layouts Between Pipelines

```javascript title="Reusable bind group layout"
const sharedLayout = device.createBindGroupLayout({ /* ... */ });

const pipeline1 = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [sharedLayout],
  }),
  // ...
});

const pipeline2 = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [sharedLayout], // Same layout
  }),
  // ...
});

// One bind group works with both pipelines
const bindGroup = device.createBindGroup({
  layout: sharedLayout,
  entries: [/* ... */]
});
```

## Complete Example

```javascript title="Full compute pipeline with bind groups" {18-35,43-56,58-65}
// WGSL compute shader
const shaderCode = `
struct Params {
  multiplier: f32,
  offset: f32
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index < arrayLength(&input)) {
    output[index] = input[index] * params.multiplier + params.offset;
  }
}
`;

const shaderModule = device.createShaderModule({ code: shaderCode });

// Create bind group layout
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } },
  ],
});

// Create pipeline
const pipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  }),
  compute: {
    module: shaderModule,
    entryPoint: "main",
  },
});

// Create buffers
const paramsBuffer = device.createBuffer({
  size: 8,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const inputData = new Float32Array(1024);
for (let i = 0; i < inputData.length; i++) inputData[i] = i;

const inputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const outputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// Write data
device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([2.0, 1.0]));
device.queue.writeBuffer(inputBuffer, 0, inputData);

// Create bind group
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    { binding: 0, resource: { buffer: paramsBuffer } },
    { binding: 1, resource: { buffer: inputBuffer } },
    { binding: 2, resource: { buffer: outputBuffer } },
  ],
});

// Execute
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(Math.ceil(inputData.length / 64));
passEncoder.end();

device.queue.submit([commandEncoder.finish()]);
```

## Resources

:::note[Official Documentation]
- [WebGPU Bind Groups Specification](https://gpuweb.github.io/gpuweb/#bind-groups)
- [TypeGPU Bind Groups Guide](https://docs.swmansion.com/TypeGPU/fundamentals/bind-groups/)
- [WebGPU Fundamentals: Uniforms](https://webgpufundamentals.org/webgpu/lessons/webgpu-uniforms.html)
:::
