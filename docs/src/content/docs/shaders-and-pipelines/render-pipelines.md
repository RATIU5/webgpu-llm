---
title: Render Pipelines
sidebar:
  order: 10
---

## Overview

Render pipelines define how raw vertex data transforms into pixels on screen. They configure a sequence of programmable and fixed-function stages that process geometry through vertex transformation, rasterization, fragment shading, and output merging.

:::note[Explicit Configuration]
Unlike older graphics APIs, WebGPU requires explicit pipeline configuration, giving fine-grained control over every aspect of rendering while enabling powerful optimizations.
:::

## Pipeline Stages

### Vertex Stage

The first programmable stage—processes each vertex independently:

- Transform positions using model-view-projection matrices
- Calculate per-vertex lighting (Gouraud shading)
- Generate texture coordinates
- Pass attributes to the fragment shader

```wgsl title="Vertex shader output"
@builtin(position) position: vec4f  // Required: clip space position
```

:::tip[Clip Space Coordinates]
Vertex shaders output positions in clip space (4D homogeneous coordinates). After perspective divide, coordinates in [-1, 1] for x/y and [0, 1] for z are visible.
:::

### Primitive Assembly

Groups vertices into geometric primitives:

| Topology | Description |
|----------|-------------|
| `triangle-list` | Every 3 vertices form a triangle (most common) |
| `triangle-strip` | Shared edges between adjacent triangles |
| `line-list` | Pairs of vertices form lines |
| `line-strip` | Connected line segments |
| `point-list` | Individual points |

### Rasterization

Converts primitives to fragments (potential pixels):
1. Clips primitives to view frustum
2. Performs perspective division
3. Maps to viewport coordinates
4. Determines pixel coverage
5. Interpolates vertex attributes

### Fragment Stage

Second programmable stage—determines pixel colors:

- Samples textures and applies filtering
- Calculates lighting (Phong, PBR)
- Implements effects (bump mapping, reflections)
- Performs alpha testing

:::tip[Early Depth Testing]
Modern GPUs can skip fragment shader execution for fragments that fail depth testing, saving significant computation. Avoid writing to depth or using `discard` to maintain this optimization.
:::

### Output Merging

Combines fragment output with existing framebuffer:

**Depth Testing**:

| Compare | Description |
|---------|-------------|
| `less` | Standard 3D rendering |
| `less-equal` | Same geometry multiple times |
| `greater` | Reverse depth buffers |
| `always` | Disable depth testing |

**Blending**:

| Mode | Formula | Use Case |
|------|---------|----------|
| Opaque | Replace | Solid objects |
| Alpha | `src × α + dst × (1-α)` | Transparency |
| Additive | `src + dst` | Particles, lights |

## Creating Render Pipelines

### Synchronous vs Async

```javascript title="Synchronous creation"
const pipeline = device.createRenderPipeline(descriptor);
```

```javascript title="Async creation (recommended)"
const pipeline = await device.createRenderPipelineAsync(descriptor);
```

:::tip[Use Async in Production]
Async creation compiles pipelines on background threads, avoiding frame drops during rendering. Create pipelines during loading screens.
:::

### Complete Pipeline Descriptor

```javascript title="Full render pipeline" {3-4,6-15,16-27}
const pipeline = device.createRenderPipeline({
  label: "Main Render Pipeline",
  layout: "auto",
  vertex: {
    module: shaderModule,
    entryPoint: "vertexMain",
    buffers: [{
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
      ],
    }],
  },
  fragment: {
    module: shaderModule,
    entryPoint: "fragmentMain",
    targets: [{
      format: navigator.gpu.getPreferredCanvasFormat(),
      blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      },
    }],
  },
  primitive: {
    topology: "triangle-list",
    cullMode: "back",
    frontFace: "ccw",
  },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",
  },
});
```

## Shader Modules

```javascript title="Create shader module"
const shaderModule = device.createShaderModule({
  label: "Triangle Shaders",
  code: `
    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
    };

    @vertex
    fn vertexMain(@location(0) pos: vec2f, @location(1) color: vec4f) -> VertexOutput {
      var output: VertexOutput;
      output.position = vec4f(pos, 0.0, 1.0);
      output.color = color;
      return output;
    }

    @fragment
    fn fragmentMain(@location(0) color: vec4f) -> @location(0) vec4f {
      return color;
    }
  `,
});
```

:::caution[Location Matching]
Fragment shader inputs must match vertex shader outputs by `@location` number and type:

```wgsl
// Vertex output
@location(0) color: vec3f,
@location(1) texCoord: vec2f,

// Fragment input - must match
fn fragmentMain(@location(0) color: vec3f, @location(1) texCoord: vec2f)
```
:::

## Vertex State

### Buffer Layouts

```javascript title="Vertex buffer layout" {2-3}
buffers: [{
  arrayStride: 24,  // Bytes per vertex
  stepMode: "vertex",
  attributes: [
    { shaderLocation: 0, offset: 0, format: "float32x3" },   // Position
    { shaderLocation: 1, offset: 12, format: "float32x3" },  // Normal
  ],
}]
```

### Attribute Formats

<details>
<summary>**Float Formats**</summary>

| Format | Size | Description |
|--------|------|-------------|
| `float32` | 4 bytes | Single float |
| `float32x2` | 8 bytes | 2D vector |
| `float32x3` | 12 bytes | 3D vector |
| `float32x4` | 16 bytes | 4D vector |
| `float16x2/x4` | 4/8 bytes | Half precision |

</details>

<details>
<summary>**Normalized Formats**</summary>

| Format | Range | Use Case |
|--------|-------|----------|
| `unorm8x4` | 0-255 → 0.0-1.0 | Vertex colors |
| `snorm8x4` | -128-127 → -1.0-1.0 | Normals |
| `unorm16x2` | Higher precision | Texture coords |

</details>

### Step Modes

| Mode | Advances | Use Case |
|------|----------|----------|
| `vertex` | Per vertex | Standard attributes |
| `instance` | Per instance | Instanced rendering |

```javascript title="Instanced rendering layout"
buffers: [
  { arrayStride: 12, stepMode: "vertex", attributes: [/* per-vertex */] },
  { arrayStride: 16, stepMode: "instance", attributes: [/* per-instance */] },
]
```

## Fragment State

### Render Targets

```javascript title="Multiple render targets"
targets: [
  { format: "rgba16float" },  // Position
  { format: "rgba16float" },  // Normal
  { format: "rgba8unorm" },   // Albedo
]
```

### Blend Configurations

<details>
<summary>**Standard Alpha Blending**</summary>

```javascript
blend: {
  color: {
    srcFactor: "src-alpha",
    dstFactor: "one-minus-src-alpha",
    operation: "add",
  },
  alpha: {
    srcFactor: "one",
    dstFactor: "one-minus-src-alpha",
    operation: "add",
  },
}
```

</details>

<details>
<summary>**Additive Blending**</summary>

```javascript
blend: {
  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
}
```

</details>

<details>
<summary>**Premultiplied Alpha**</summary>

```javascript
blend: {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
}
```

</details>

## Primitive State

```javascript title="Primitive configuration"
primitive: {
  topology: "triangle-list",
  frontFace: "ccw",        // Counter-clockwise = front
  cullMode: "back",        // Cull back faces
  stripIndexFormat: undefined,
}
```

:::tip[Enable Culling]
Back-face culling reduces fragment work by ~50% for closed meshes:

```javascript
cullMode: "back"
```

Disable only for transparent or double-sided geometry.
:::

## Depth/Stencil State

```javascript title="Depth testing configuration"
depthStencil: {
  format: "depth24plus",
  depthWriteEnabled: true,
  depthCompare: "less",
}
```

| Format | Description |
|--------|-------------|
| `depth16unorm` | 16-bit depth |
| `depth24plus` | At least 24-bit (recommended) |
| `depth32float` | Maximum precision |
| `depth24plus-stencil8` | Depth + stencil |

:::danger[Create Depth Texture]
Depth testing requires a depth texture attached to the render pass:

```javascript
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```
:::

## TypeGPU Render Pipelines

### Builder Pattern

```typescript title="TypeGPU pipeline builder"
const pipeline = root
  .withVertex(vertexShader, { position, normal, texCoord })
  .withFragment(fragmentShader, { color: { format: "bgra8unorm" } })
  .withDepthStencil({ format: "depth24plus", depthCompare: "less", depthWriteEnabled: true })
  .withPrimitive({ topology: "triangle-list", cullMode: "back" })
  .createPipeline();
```

### Vertex Layout

```typescript title="TypeGPU vertex layout"
const vertexLayout = tgpu.vertexLayout({
  position: { format: "float32x3", offset: 0 },
  normal: { format: "float32x3", offset: 12 },
  texCoord: { format: "float32x2", offset: 24 },
}, {
  arrayStride: 32,
  stepMode: "vertex",
});
```

### Drawing

```typescript title="TypeGPU draw call"
pipeline
  .with(vertexLayout, vertexBuffer)
  .with(uniformBindGroup)
  .withColorAttachment({
    color: {
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
      storeOp: "store",
    },
  })
  .withDepthStencilAttachment({
    view: depthTextureView,
    depthLoadOp: "clear",
    depthClearValue: 1.0,
    depthStoreOp: "store",
  })
  .draw(vertexCount);
```

## Complete Example

```javascript title="Full WebGPU render pipeline" {36-55}
const shaderModule = device.createShaderModule({
  code: `
    struct Uniforms { modelViewProjection: mat4x4f };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    struct VertexInput { @location(0) position: vec3f, @location(1) color: vec3f };
    struct VertexOutput { @builtin(position) position: vec4f, @location(0) color: vec3f };

    @vertex
    fn vertexMain(input: VertexInput) -> VertexOutput {
      var output: VertexOutput;
      output.position = uniforms.modelViewProjection * vec4f(input.position, 1.0);
      output.color = input.color;
      return output;
    }

    @fragment
    fn fragmentMain(@location(0) color: vec3f) -> @location(0) vec4f {
      return vec4f(color, 1.0);
    }
  `,
});

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: shaderModule,
    entryPoint: "vertexMain",
    buffers: [{
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
      ],
    }],
  },
  fragment: {
    module: shaderModule,
    entryPoint: "fragmentMain",
    targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
  },
  primitive: { topology: "triangle-list", cullMode: "back" },
  depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
});

// Render
const commandEncoder = device.createCommandEncoder();
const renderPass = commandEncoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    loadOp: "clear",
    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
    storeOp: "store",
  }],
  depthStencilAttachment: {
    view: depthTextureView,
    depthLoadOp: "clear",
    depthClearValue: 1.0,
    depthStoreOp: "store",
  },
});

renderPass.setPipeline(pipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.setBindGroup(0, bindGroup);
renderPass.draw(3);
renderPass.end();

device.queue.submit([commandEncoder.finish()]);
```

## Performance Guidelines

:::tip[Create Pipelines During Initialization]
Pipeline creation can take several milliseconds. Create all pipelines during startup:

```javascript
const pipelines = await Promise.all([
  device.createRenderPipelineAsync(desc1),
  device.createRenderPipelineAsync(desc2),
]);
```
:::

:::tip[Draw Order]
For opaque geometry, sort objects front-to-back to maximize early depth rejection. Draw transparent objects back-to-front after opaques with depth writes disabled.
:::

:::tip[Batch Draw Calls]
Minimize draw calls through:
- Instanced rendering for repeated geometry
- Combining static geometry into larger buffers
- Texture atlases instead of individual textures
:::

:::caution[Transparent Objects]
Transparent rendering requires correct blend state and drawing order:

```javascript
depthStencil: {
  depthWriteEnabled: false,  // Don't write depth
  depthCompare: "less",       // Still test depth
}
```
:::

## Resources

:::note[Official Documentation]
- [WebGPU Render Pipeline Specification](https://gpuweb.github.io/gpuweb/#render-pipeline)
- [TypeGPU Rendering Guide](https://docs.swmansion.com/TypeGPU/fundamentals/rendering/)
- [WebGPU Fundamentals: Pipelines](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html)
:::
