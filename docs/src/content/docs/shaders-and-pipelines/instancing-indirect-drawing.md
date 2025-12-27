---
title: Instancing and Indirect Drawing
sidebar:
  order: 25
---

## Overview

Instancing draws multiple copies of geometry with a single draw call. Indirect drawing reads draw parameters from GPU buffers, enabling GPU-driven rendering without CPU readback.

## Instancing

### Basic Instanced Drawing

```javascript title="Draw 1000 instances" {5}
// Draw 36 vertices, 1000 times
pass.draw(36, 1000);

// Or with indices
pass.drawIndexed(36, 1000);
```

### Accessing Instance Data

```wgsl title="Using instance_index builtin"
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@group(0) @binding(0) var<storage, read> transforms: array<mat4x4f>;
@group(0) @binding(1) var<storage, read> colors: array<vec4f>;

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @builtin(instance_index) instanceIdx: u32
) -> VertexOutput {
  var output: VertexOutput;
  output.position = transforms[instanceIdx] * vec4f(position, 1.0);
  output.color = colors[instanceIdx];
  return output;
}
```

### Instance Step Mode

Alternative: store per-instance data in vertex buffers with `stepMode: "instance"`:

```javascript title="Per-instance vertex attributes" {7-12}
const pipeline = device.createRenderPipeline({
  vertex: {
    buffers: [
      // Per-vertex data
      { arrayStride: 12, stepMode: "vertex", attributes: [...] },
      // Per-instance data (advances once per instance)
      {
        arrayStride: 64,
        stepMode: "instance",
        attributes: [
          { shaderLocation: 1, offset: 0, format: "float32x4" },  // row 0
          { shaderLocation: 2, offset: 16, format: "float32x4" }, // row 1
          { shaderLocation: 3, offset: 32, format: "float32x4" }, // row 2
          { shaderLocation: 4, offset: 48, format: "float32x4" }, // row 3
        ],
      },
    ],
  },
  // ...
});
```

## Indirect Drawing

### Indirect Buffer Layout

Draw parameters come from GPU buffers instead of JavaScript:

| Method | Buffer Contents (u32) | Size |
|--------|----------------------|------|
| `drawIndirect` | vertexCount, instanceCount, firstVertex, firstInstance | 16 bytes |
| `drawIndexedIndirect` | indexCount, instanceCount, firstIndex, baseVertex, firstInstance | 20 bytes |

```javascript title="Create indirect buffer" {2-8}
const indirectBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
});

// Write draw parameters
const params = new Uint32Array([36, 1000, 0, 0]); // 36 verts, 1000 instances
device.queue.writeBuffer(indirectBuffer, 0, params);
```

### Indirect Draw Calls

```javascript title="Execute indirect draw"
pass.drawIndirect(indirectBuffer, 0);

// Or indexed
pass.drawIndexedIndirect(indirectBuffer, 0);
```

:::caution[indirect-first-instance Feature]
Non-zero `firstInstance` requires the `indirect-first-instance` feature:

```javascript
const device = await adapter.requestDevice({
  requiredFeatures: ["indirect-first-instance"],
});
```

Without this feature, non-zero firstInstance values are treated as no-ops.
:::

### Multiple Indirect Draws

:::tip[Batch Indirect Buffers]
Combine multiple draw calls into one buffer to reduce validation overhead:

```javascript title="Batched indirect draws" {2-7}
// One buffer for all draws (16 bytes each)
const batchedBuffer = device.createBuffer({
  size: drawCount * 16,
  usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
});

// Execute multiple draws from same buffer
for (let i = 0; i < drawCount; i++) {
  pass.drawIndirect(batchedBuffer, i * 16);
}
```

Separate buffers per draw can cause 50%+ overhead from validation.
:::

## GPU-Driven Rendering

Combine compute shaders with indirect drawing for GPU culling:

```wgsl title="GPU culling compute shader"
struct DrawIndirect {
  vertexCount: u32,
  instanceCount: u32,
  firstVertex: u32,
  firstInstance: u32,
}

@group(0) @binding(0) var<storage, read> boundingSpheres: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> drawCommands: array<DrawIndirect>;
@group(0) @binding(2) var<storage, read_write> visibleInstances: array<u32>;
@group(0) @binding(3) var<storage, read_write> visibleCount: atomic<u32>;

@compute @workgroup_size(64)
fn cullInstances(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  let sphere = boundingSpheres[idx];

  if (isVisible(sphere)) {
    let slot = atomicAdd(&visibleCount, 1u);
    visibleInstances[slot] = idx;
  }
}
```

### Culling Pipeline

1. **Reset**: Clear visible count to 0
2. **Cull**: Compute shader tests each instance, writes visible indices
3. **Update**: Copy visible count to indirect buffer's instanceCount
4. **Draw**: Execute indirect draw with GPU-determined instance count

```javascript title="GPU culling workflow"
// Compute pass: cull instances
const computePass = encoder.beginComputePass();
computePass.setPipeline(cullPipeline);
computePass.setBindGroup(0, cullBindGroup);
computePass.dispatchWorkgroups(Math.ceil(instanceCount / 64));
computePass.end();

// Copy visible count to indirect buffer
encoder.copyBufferToBuffer(
  visibleCountBuffer, 0,
  indirectBuffer, 4,  // offset to instanceCount field
  4
);

// Render pass: draw visible instances
const renderPass = encoder.beginRenderPass(descriptor);
renderPass.setPipeline(renderPipeline);
renderPass.drawIndirect(indirectBuffer, 0);
renderPass.end();
```

## TypeGPU Instancing

```typescript title="TypeGPU instanced rendering"
import tgpu from "typegpu";
import * as d from "typegpu/data";

const transforms = root.createBuffer(d.arrayOf(d.mat4x4f, 1000))
  .$usage("storage");

const vertexFn = tgpu.vertexFn({
  position: d.vec3f,
}, d.vec4f).does`(input) -> vec4f {
  let idx = ${tgpu.builtin.instanceIndex};
  return ${transforms}[idx] * vec4f(input.position, 1.0);
}`;
```

## Performance Comparison

| Technique | CPU Overhead | GPU Control | Use Case |
|-----------|--------------|-------------|----------|
| Direct draw | Highest | None | Simple scenes |
| Instancing | Medium | None | Many identical objects |
| Indirect draw | Low | Partial | Dynamic instance counts |
| GPU culling + indirect | Lowest | Full | Large scenes, culling |

:::note[When to Use Each]
- **Direct**: < 100 objects, different meshes
- **Instancing**: 100-10,000 identical objects (trees, particles)
- **Indirect**: Dynamic counts, GPU-generated geometry
- **GPU culling**: 10,000+ objects, complex visibility tests
:::
