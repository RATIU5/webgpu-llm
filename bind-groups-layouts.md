# Bind Groups and Layouts

## Overview

Bind groups and layouts form the cornerstone of WebGPU's resource binding model, providing a structured way to connect GPU resources to shader programs. When writing GPU-accelerated applications, shaders need access to various resources such as buffers containing uniform data, storage buffers for computation results, textures for image processing, and samplers for texture filtering. The bind group system organizes these resources into logical collections that can be efficiently bound to the GPU pipeline.

The binding model uses a two-level hierarchy: **bind group layouts** define the schema or template describing what types of resources will be used, while **bind groups** contain the actual resource instances that match this schema. This separation allows for flexible resource management, efficient validation, and the ability to swap different resource sets while maintaining the same pipeline configuration.

Think of bind group layouts as interfaces or contracts that specify "what" resources a shader expects, and bind groups as the implementations that provide the actual resources. This abstraction enables powerful patterns like sharing layouts across multiple pipelines, pre-compiling pipelines before resources are fully loaded, and organizing resources by their update frequency for optimal performance.

## Key Concepts

### Bind Group

A **bind group** is a collection of GPU resources (buffers, textures, samplers) bundled together as a single unit. Rather than binding resources individually, bind groups allow you to bind multiple related resources in one operation. This not only improves API ergonomics but also enables GPU drivers to optimize resource binding operations.

Each bind group is an instance of `GPUBindGroup` and contains actual GPU resources at specific binding slots. For example, a bind group for a lighting shader might contain a uniform buffer with light properties, a storage buffer with shadow data, and a texture containing a shadow map.

### Bind Group Layout

A **bind group layout** defines the schema or structure that bind groups must follow. It specifies:

- The number of bindings in the group
- The type of resource at each binding (buffer, texture, sampler, etc.)
- The shader stages that can access each resource
- Additional properties like buffer types, texture formats, and access modes

The layout is represented by `GPUBindGroupLayout` and acts as a template for creating compatible bind groups. Multiple bind groups can share the same layout, allowing you to swap between different resource sets efficiently.

### Binding Indices in WGSL

In WGSL shader code, resources are referenced using two attributes:

- `@group(n)` - Specifies which bind group (0-3) contains the resource
- `@binding(m)` - Specifies the index within that bind group

For example:
```wgsl
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec3f>;
@group(1) @binding(0) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(1) var textureSampler: sampler;
```

This hierarchical addressing allows organizing resources logically. Group 0 might contain per-frame data that rarely changes, while group 1 contains per-material data that varies between draw calls.

### Resource Visibility Across Shader Stages

Each binding in a layout declares which shader stages can access it. The available stages are:

- **VERTEX** - Accessible in vertex shaders
- **FRAGMENT** - Accessible in fragment shaders
- **COMPUTE** - Accessible in compute shaders

You can combine multiple stages using bitwise OR operations (`GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT`). Properly restricting visibility can help drivers optimize resource access and catch errors where resources are used in inappropriate stages.

## WebGPU Binding Model

### WGSL Bindings

WGSL uses decorators to connect shader variables to bind group resources. The syntax is straightforward but powerful:

```wgsl
// Uniform buffer binding
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Read-only storage buffer
@group(0) @binding(1) var<storage, read> inputData: array<f32>;

// Read-write storage buffer
@group(0) @binding(2) var<storage, read_write> outputData: array<f32>;

// Texture and sampler bindings
@group(1) @binding(0) var baseColorTexture: texture_2d<f32>;
@group(1) @binding(1) var linearSampler: sampler;

// Storage texture binding
@group(2) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
```

The `var<storage, read>` syntax specifies a storage buffer with read-only access, while `var<storage, read_write>` allows both reading and writing. Uniform buffers use `var<uniform>` and are always read-only but optimized for small, frequently accessed data.

### Creating Bind Group Layouts

Bind group layouts are created using `device.createBindGroupLayout()` with a `GPUBindGroupLayoutDescriptor`. Each entry in the layout specifies a binding's configuration:

```javascript
const bindGroupLayout = device.createBindGroupLayout({
  label: "Compute Bind Group Layout",
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "uniform",
        minBindingSize: 16  // Optional: minimum buffer size in bytes
      }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage",
        minBindingSize: 0
      }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "storage"  // Read-write storage buffer
      }
    }
  ]
});
```

#### Entry Definitions

Each entry type has specific properties:

**Buffer Bindings:**
```javascript
{
  binding: 0,
  visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
  buffer: {
    type: "uniform" | "storage" | "read-only-storage",
    hasDynamicOffset: false,  // Enable dynamic offsets
    minBindingSize: 0  // Minimum size required (0 = no minimum)
  }
}
```

**Sampler Bindings:**
```javascript
{
  binding: 1,
  visibility: GPUShaderStage.FRAGMENT,
  sampler: {
    type: "filtering" | "non-filtering" | "comparison"
  }
}
```

**Texture Bindings:**
```javascript
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

**Storage Texture Bindings:**
```javascript
{
  binding: 3,
  visibility: GPUShaderStage.COMPUTE,
  storageTexture: {
    access: "write-only" | "read-only" | "read-write",
    format: "rgba8unorm",  // Must specify exact format
    viewDimension: "2d"
  }
}
```

### Creating Bind Groups

Once you have a layout, create bind groups that conform to it using `device.createBindGroup()`:

```javascript
const bindGroup = device.createBindGroup({
  label: "Compute Bind Group",
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: {
        buffer: uniformBuffer,
        offset: 0,
        size: 16  // Size in bytes, can be omitted to use entire buffer
      }
    },
    {
      binding: 1,
      resource: {
        buffer: inputBuffer
      }
    },
    {
      binding: 2,
      resource: {
        buffer: outputBuffer
      }
    }
  ]
});
```

For textures and samplers:

```javascript
const renderBindGroup = device.createBindGroup({
  label: "Render Bind Group",
  layout: renderBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: sampler  // GPUSampler object
    },
    {
      binding: 1,
      resource: texture.createView()  // GPUTextureView object
    }
  ]
});
```

The entries in the bind group must exactly match the bindings defined in the layout. Any mismatch in binding numbers, resource types, or missing bindings will cause a validation error.

### Pipeline Layouts

A **pipeline layout** combines multiple bind group layouts and defines the complete resource binding interface for a pipeline:

```javascript
const pipelineLayout = device.createPipelineLayout({
  label: "Main Pipeline Layout",
  bindGroupLayouts: [
    bindGroupLayout0,  // @group(0) in shaders
    bindGroupLayout1,  // @group(1) in shaders
    bindGroupLayout2   // @group(2) in shaders
  ]
});
```

You then use this layout when creating render or compute pipelines:

```javascript
const computePipeline = device.createComputePipeline({
  label: "Compute Pipeline",
  layout: pipelineLayout,
  compute: {
    module: shaderModule,
    entryPoint: "main"
  }
});
```

Alternatively, you can use `layout: "auto"` to have WebGPU automatically create a pipeline layout based on the shader bindings:

```javascript
const computePipeline = device.createComputePipeline({
  label: "Compute Pipeline",
  layout: "auto",  // Infer layout from shader
  compute: {
    module: shaderModule,
    entryPoint: "main"
  }
});
```

When using automatic layouts, retrieve the bind group layout to create compatible bind groups:

```javascript
const bindGroupLayout = computePipeline.getBindGroupLayout(0);  // Get group 0
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [/* ... */]
});
```

## TypeGPU Bind Groups

TypeGPU provides a type-safe, ergonomic abstraction over WebGPU's bind group system, eliminating manual index management and providing compile-time type checking.

### Type-safe Layouts

Instead of numeric binding indices, TypeGPU uses named properties:

```typescript
import { tgpu, d } from "typegpu";

// Define a data structure for uniforms
const UniformsSchema = d.struct({
  viewProjection: d.mat4x4f,
  time: d.f32,
  lightDirection: d.vec3f
});

// Create a type-safe bind group layout
const sceneLayout = tgpu.bindGroupLayout({
  uniforms: { uniform: UniformsSchema },
  instanceData: { storage: d.arrayOf(d.mat4x4f) }
});
```

Key features of TypeGPU layouts:

**Named Properties**: Instead of `binding: 0`, `binding: 1`, you use meaningful names like `uniforms` and `instanceData`. TypeGPU automatically assigns sequential binding indices.

**Automatic Index Assignment**: Properties are mapped to binding indices in the order they appear. You don't need to track which binding index corresponds to which resource.

**Type Inference**: TypeScript knows the exact type of each binding, providing autocomplete and catching type errors at compile time.

**Resource Types**: TypeGPU supports all WebGPU resource types with clear syntax:

```typescript
const layout = tgpu.bindGroupLayout({
  // Uniform buffer (read-only)
  uniforms: { uniform: d.struct({ value: d.f32 }) },

  // Storage buffer (read-only by default)
  readData: { storage: d.arrayOf(d.f32) },

  // Mutable storage buffer
  writeData: { storage: d.f32, access: 'mutable' },

  // Filtering sampler
  linearSampler: { sampler: 'filtering' },

  // Texture
  diffuseTexture: { texture: d.tex2d<'f32'> },

  // Storage texture
  outputTexture: { storageTexture: d.storageTex2d<'rgba8unorm'> }
});
```

### Creating Bind Groups

Creating bind groups with TypeGPU is straightforward and type-safe:

```typescript
// Create buffers
const uniformBuffer = root.createBuffer(UniformsSchema, {
  viewProjection: mat4.identity(),
  time: 0,
  lightDirection: vec3(0, -1, 0)
});

const storageBuffer = root.createBuffer(d.arrayOf(d.f32),
  new Float32Array(1000));

// Populate the bind group
const bindGroup = root.createBindGroup(sceneLayout, {
  uniforms: uniformBuffer,
  instanceData: storageBuffer
});
```

The `populate` method ensures that:
- All required bindings are provided
- Each resource matches the expected type
- You can't accidentally swap resources between bindings

TypeGPU also supports runtime-sized arrays using functions:

```typescript
const layout = tgpu.bindGroupLayout({
  // Size determined at bind group creation
  dynamicData: { storage: (n: number) => d.arrayOf(d.f32, n) }
});

const bindGroup = root.createBindGroup(layout, {
  dynamicData: root.createBuffer(d.arrayOf(d.f32, 500), data)
});
```

### Using Bind Groups in Pipelines

TypeGPU integrates bind groups seamlessly with pipeline creation:

```typescript
import { wgsl } from "typegpu";

// Create a compute shader using the bind group layout
const computeFn = tgpu.computeFn({
  workgroupSize: [256, 1, 1],
  bindings: [sceneLayout]  // Declare bind groups
}, (props) => wgsl`
  @compute @workgroup_size(256, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3u) {
    // Access bindings using property names
    let time = ${props.bindings[0].uniforms}.time;
    let matrix = ${props.bindings[0].instanceData}[gid.x];
    // ... compute logic
  }
`);

// Execute with bind group
computeFn.execute({
  bindings: [bindGroup],
  workgroups: [4, 1, 1]
});
```

For multiple bind groups:

```typescript
const layout0 = tgpu.bindGroupLayout({
  sceneData: { uniform: SceneDataSchema }
});

const layout1 = tgpu.bindGroupLayout({
  materialData: { uniform: MaterialDataSchema },
  baseColorTexture: { texture: d.tex2d<'f32'> }
});

const bindGroup0 = root.createBindGroup(layout0, { /* ... */ });
const bindGroup1 = root.createBindGroup(layout1, { /* ... */ });

const pipeline = tgpu.renderPipeline({
  vertex: vertexFn,
  fragment: fragmentFn,
  bindGroupLayouts: [layout0, layout1]
});

// Use both bind groups
pipeline.execute({
  bindGroups: [bindGroup0, bindGroup1],
  // ... other render parameters
});
```

## Resource Visibility

Resource visibility controls which shader stages can access each binding. Proper visibility configuration is crucial for both correctness and performance.

### Shader Stage Flags

WebGPU defines three shader stages:

- **GPUShaderStage.VERTEX**: Resources accessible in vertex shaders
- **GPUShaderStage.FRAGMENT**: Resources accessible in fragment shaders
- **GPUShaderStage.COMPUTE**: Resources accessible in compute shaders

Combine flags using bitwise OR:

```javascript
const visibilityInVertexAndFragment =
  GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
```

### Visibility Best Practices

**Principle of Least Privilege**: Only grant access to stages that actually need the resource. This helps GPU drivers optimize resource access patterns and can catch shader errors early.

```javascript
const layout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      // Camera matrix only needed in vertex shader
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: "uniform" }
    },
    {
      binding: 1,
      // Material properties only needed in fragment shader
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform" }
    },
    {
      binding: 2,
      // Shared data accessible in both stages
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" }
    }
  ]
});
```

### Common Visibility Patterns

**Per-Frame Data**: Often used across all stages
```javascript
visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
```

**Transformation Data**: Typically vertex-only
```javascript
visibility: GPUShaderStage.VERTEX
```

**Lighting and Materials**: Usually fragment-only
```javascript
visibility: GPUShaderStage.FRAGMENT
```

**Compute Resources**: Exclusively compute stage
```javascript
visibility: GPUShaderStage.COMPUTE
```

## Dynamic Binding

Dynamic offsets allow using different sections of a buffer without creating multiple bind groups, significantly reducing bind group management overhead.

### Enabling Dynamic Offsets

Mark a buffer binding as dynamic in the layout:

```javascript
const layout = device.createBindGroupLayout({
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.VERTEX,
    buffer: {
      type: "uniform",
      hasDynamicOffset: true,  // Enable dynamic offsets
      minBindingSize: 256  // Each offset must be 256-byte aligned
    }
  }]
});
```

### Using Dynamic Offsets

Provide offsets when setting bind groups during rendering:

```javascript
const passEncoder = commandEncoder.beginRenderPass(/* ... */);
passEncoder.setPipeline(pipeline);

// Draw multiple objects using different buffer offsets
for (let i = 0; i < objectCount; i++) {
  const offset = i * 256;  // Must be aligned to 256 bytes
  passEncoder.setBindGroup(0, bindGroup, [offset]);  // Dynamic offset array
  passEncoder.draw(vertexCount, 1, 0, 0);
}

passEncoder.end();
```

### Dynamic Offset Requirements

**Alignment**: Dynamic offsets must be aligned to 256 bytes for uniform buffers and 32 bytes for storage buffers. This is a hardware requirement that ensures efficient memory access.

**Multiple Offsets**: If a bind group has multiple dynamic bindings, provide offsets in binding order:

```javascript
// Layout has dynamic buffers at bindings 0 and 2
passEncoder.setBindGroup(0, bindGroup, [offset0, offset2]);
```

**Use Cases**:
- Drawing multiple objects with per-object uniforms in a single buffer
- Batching particles or instances with individual parameters
- Implementing circular buffers for streaming data

## Best Practices

### Layout Organization

**Group by Update Frequency**: Organize bind groups based on how often resources change:

- **Group 0**: Per-frame data (camera, time, global settings) - Updated once per frame
- **Group 1**: Per-material data (textures, colors) - Updated when materials change
- **Group 2**: Per-object data (transform matrices) - Updated for each draw call

This minimizes bind group switches and maximizes GPU efficiency.

```javascript
// Good: Organized by frequency
const frameLayout = createLayoutForFrameData();      // Group 0
const materialLayout = createLayoutForMaterialData(); // Group 1
const objectLayout = createLayoutForObjectData();     // Group 2

const pipelineLayout = device.createPipelineLayout({
  bindGroupLayouts: [frameLayout, materialLayout, objectLayout]
});
```

### Binding Frequency Optimization

Minimize the number of `setBindGroup` calls by:

1. **Reusing bind groups**: Cache and reuse bind groups when possible
2. **Sorting draw calls**: Sort by material/bind group to reduce switches
3. **Using dynamic offsets**: Instead of creating many bind groups with small variations

```javascript
// Efficient rendering loop
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

### Resource Sharing

Share bind group layouts between pipelines when possible:

```javascript
// Good: Reuse layouts
const sharedLayout = device.createBindGroupLayout(/* ... */);

const pipeline1 = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [sharedLayout]
  }),
  // ...
});

const pipeline2 = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [sharedLayout]  // Same layout
  }),
  // ...
});

// Now bind groups work with both pipelines
const bindGroup = device.createBindGroup({ layout: sharedLayout, /* ... */ });
```

### Buffer Size Considerations

When using dynamic offsets or sharing buffers:

```javascript
// Ensure proper alignment for dynamic offsets
const uniformBufferSize = Math.ceil(structSize / 256) * 256;  // 256-byte aligned

// Create buffer large enough for all instances
const buffer = device.createBuffer({
  size: uniformBufferSize * instanceCount,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
```

## Common Pitfalls

### Mismatched Layouts

**Problem**: Bind group doesn't match the pipeline's expected layout.

```javascript
// Layout expects storage buffer at binding 0
const layout = device.createBindGroupLayout({
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: "storage" }  // Storage buffer
  }]
});

// Wrong: Providing uniform buffer instead
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [{
    binding: 0,
    resource: { buffer: uniformBuffer }  // This is a uniform buffer!
  }]
});
```

**Solution**: Ensure the buffer's usage flags and type match the layout:

```javascript
const storageBuffer = device.createBuffer({
  size: 1024,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [{
    binding: 0,
    resource: { buffer: storageBuffer }
  }]
});
```

### Missing Bindings

**Problem**: Layout defines a binding but it's not provided in the bind group.

```javascript
const layout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
  ]
});

// Wrong: Missing binding 1
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [
    { binding: 0, resource: { buffer: uniformBuffer } }
    // binding 1 is missing!
  ]
});
```

**Solution**: Provide all bindings defined in the layout:

```javascript
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: storageBuffer } }
  ]
});
```

### Wrong Resource Types

**Problem**: Providing a sampler where a texture is expected, or vice versa.

```javascript
const layout = device.createBindGroupLayout({
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.FRAGMENT,
    texture: { sampleType: "float" }  // Expects texture
  }]
});

// Wrong: Providing sampler instead of texture
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [{
    binding: 0,
    resource: sampler  // Should be texture.createView()
  }]
});
```

**Solution**: Match resource types exactly:

```javascript
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [{
    binding: 0,
    resource: texture.createView()  // Correct: texture view
  }]
});
```

### Incorrect WGSL Binding Annotations

**Problem**: WGSL `@group` and `@binding` don't match JavaScript bind group indices.

```wgsl
// Shader expects data at group 0, binding 1
@group(0) @binding(1) var<storage, read> data: array<f32>;
```

```javascript
// Wrong: Bound at binding 0 instead of 1
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [{
    binding: 0,  // Should be 1
    resource: { buffer: storageBuffer }
  }]
});
```

**Solution**: Keep WGSL and JavaScript binding indices synchronized:

```javascript
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [{
    binding: 1,  // Matches @binding(1) in shader
    resource: { buffer: storageBuffer }
  }]
});
```

### Forgetting Visibility Flags

**Problem**: Resource not accessible in the shader stage where it's used.

```javascript
const layout = device.createBindGroupLayout({
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.VERTEX,  // Only vertex stage
    buffer: { type: "uniform" }
  }]
});
```

```wgsl
// Fragment shader tries to access it - ERROR!
@fragment
fn fragmentMain() -> @location(0) vec4f {
  let value = uniforms.someValue;  // uniforms not visible here
  return vec4f(value);
}
```

**Solution**: Include all necessary shader stages in visibility:

```javascript
visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
```

### Dynamic Offset Alignment Issues

**Problem**: Dynamic offsets not aligned to required boundaries.

```javascript
// Wrong: Offset not aligned to 256 bytes
passEncoder.setBindGroup(0, bindGroup, [100]);  // Error!
```

**Solution**: Always align offsets appropriately:

```javascript
const offset = Math.floor(index * uniformSize / 256) * 256;
passEncoder.setBindGroup(0, bindGroup, [offset]);  // Aligned to 256
```

## Complete Example

Here's a complete example demonstrating bind groups in a compute pipeline:

```javascript
// Create GPU device
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

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
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }
  ]
});

// Create pipeline
const pipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  }),
  compute: {
    module: shaderModule,
    entryPoint: "main"
  }
});

// Create buffers
const paramsBuffer = device.createBuffer({
  size: 8,  // 2 f32s
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const inputData = new Float32Array(1024);
for (let i = 0; i < inputData.length; i++) inputData[i] = i;

const inputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});

const outputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
});

// Write data to buffers
device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([2.0, 1.0]));
device.queue.writeBuffer(inputBuffer, 0, inputData);

// Create bind group
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    { binding: 0, resource: { buffer: paramsBuffer } },
    { binding: 1, resource: { buffer: inputBuffer } },
    { binding: 2, resource: { buffer: outputBuffer } }
  ]
});

// Encode and submit commands
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(Math.ceil(inputData.length / 64));
passEncoder.end();

device.queue.submit([commandEncoder.finish()]);
```

This comprehensive guide covers the essential concepts and practical usage of bind groups and layouts in WebGPU, providing you with the knowledge to effectively manage GPU resources in your applications.
