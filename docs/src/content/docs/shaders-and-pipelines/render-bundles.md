---
title: Render Bundles
sidebar:
  order: 30
---

## Overview

Render bundles pre-record sequences of render commands for efficient replay. They reduce CPU overhead by validating commands once during encoding rather than every frame.

:::note[When Bundles Help]
Render bundles benefit **CPU-bound** applications where JavaScript overhead limits frame rate. They do not improve **GPU-bound** applications where fill rate or vertex processing is the bottleneck.
:::

## Creating Render Bundles

### Bundle Encoder

```javascript title="Create bundle encoder" {1-7}
const bundleEncoder = device.createRenderBundleEncoder({
  colorFormats: ["bgra8unorm"],
  depthStencilFormat: "depth24plus",
  sampleCount: 1,
});

// Record commands (same API as render pass)
bundleEncoder.setPipeline(pipeline);
bundleEncoder.setBindGroup(0, bindGroup);
bundleEncoder.setVertexBuffer(0, vertexBuffer);
bundleEncoder.draw(36, 100);

// Finalize into reusable bundle
const bundle = bundleEncoder.finish();
```

### Executing Bundles

```javascript title="Execute in render pass" {8}
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    loadOp: "clear",
    storeOp: "store",
  }],
  depthStencilAttachment: { /* ... */ },
});

pass.executeBundles([bundle]);
pass.end();
```

## Configuration Requirements

Bundle encoder settings must match the render pass:

| Property | Description |
|----------|-------------|
| `colorFormats` | Array of color attachment formats |
| `depthStencilFormat` | Depth/stencil format (if used) |
| `sampleCount` | MSAA sample count (1 or 4) |
| `depthReadOnly` | If true, bundle won't write depth |
| `stencilReadOnly` | If true, bundle won't write stencil |

:::danger[Format Mismatch]
Executing a bundle in an incompatible render pass causes a validation error. Always match formats exactly.
:::

## Limitations

Bundles cannot:
- Begin/end occlusion queries
- Set scissor rect
- Set viewport
- Set blend constant
- Set stencil reference

These values are inherited from the render pass that executes the bundle.

## Use Cases

### VR Rendering

Record scene once, replay for each eye with different view matrix:

```javascript title="VR stereo rendering" {8-9,15-16}
// Record scene commands once
const sceneBundle = createSceneBundle();

// Left eye
updateViewMatrix(leftEyeView);
device.queue.writeBuffer(viewUniformBuffer, 0, leftEyeView);
const leftPass = encoder.beginRenderPass(leftEyeDescriptor);
leftPass.executeBundles([sceneBundle]);
leftPass.end();

// Right eye
updateViewMatrix(rightEyeView);
device.queue.writeBuffer(viewUniformBuffer, 0, rightEyeView);
const rightPass = encoder.beginRenderPass(rightEyeDescriptor);
rightPass.executeBundles([sceneBundle]);
rightPass.end();
```

### Static Scene Elements

Separate static and dynamic content:

```javascript title="Static + dynamic rendering"
// Pre-recorded static geometry (terrain, buildings)
const staticBundle = createStaticBundle();

function render() {
  const pass = encoder.beginRenderPass(descriptor);

  // Execute pre-recorded static content
  pass.executeBundles([staticBundle]);

  // Draw dynamic content directly
  pass.setPipeline(characterPipeline);
  pass.setBindGroup(0, characterBindGroup);
  pass.draw(characterVertexCount);

  pass.end();
}
```

### Multiple Bundles

```javascript title="Composing bundles"
const terrainBundle = createTerrainBundle();
const buildingsBundle = createBuildingsBundle();
const vegetationBundle = createVegetationBundle();

// Execute all in single call
pass.executeBundles([terrainBundle, buildingsBundle, vegetationBundle]);
```

## Dynamic Content with Bundles

### Buffer Updates

Bundle commands are fixed, but buffer contents can change:

```javascript title="Dynamic uniforms with static bundle"
// Bundle uses same bind group every frame
const bundle = createBundle(uniformBindGroup);

function render(time) {
  // Update buffer content (not the bundle)
  uniformData[0] = time;
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Execute unchanged bundle
  pass.executeBundles([bundle]);
}
```

### Indirect Drawing in Bundles

Combine bundles with indirect draws for dynamic instance counts:

```javascript title="Bundle with indirect draw" {4-5}
const bundleEncoder = device.createRenderBundleEncoder(config);
bundleEncoder.setPipeline(pipeline);
bundleEncoder.setBindGroup(0, bindGroup);
// Instance count read from buffer at execution time
bundleEncoder.drawIndirect(indirectBuffer, 0);
const bundle = bundleEncoder.finish();

// Compute shader can update indirectBuffer each frame
// Bundle stays unchanged, but drawn geometry varies
```

:::tip[GPU Culling + Bundles]
Use indirect draws in bundles for GPU-driven visibility. The bundle records the draw command; the compute shader determines what actually gets drawn.
:::

## Performance

### Benchmark Results

Testing on M1 Mac with 40,000 objects:

| Method | Frame Time |
|--------|------------|
| Direct draw calls | ~10ms |
| Render bundle | ~2-5ms |

**2-5x speedup** for CPU-bound scenarios.

### When to Use

| Scenario | Use Bundle? |
|----------|-------------|
| Static scenes | Yes |
| VR (same scene, two views) | Yes |
| Many draw calls (1000+) | Yes |
| Frequently changing pipelines | No |
| GPU-bound rendering | No benefit |
| Dynamic scissor/viewport | No |

### Profiling

```javascript title="Measure bundle benefit"
const start = performance.now();
for (let i = 0; i < 1000; i++) {
  // Simulate draw calls
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(36);
}
const directTime = performance.now() - start;

const bundleStart = performance.now();
pass.executeBundles([bundleWith1000Draws]);
const bundleTime = performance.now() - bundleStart;

console.log(`Direct: ${directTime}ms, Bundle: ${bundleTime}ms`);
```

## TypeGPU Integration

```typescript title="TypeGPU with bundles"
import tgpu from "typegpu";

// Create pipeline through TypeGPU
const pipeline = root["~unstable"]
  .withVertex(vertexFn, vertexLayout)
  .withFragment(fragmentFn, { format: "bgra8unorm" })
  .withPrimitive({ topology: "triangle-list" })
  .createPipeline();

// Get raw WebGPU pipeline for bundle
const rawPipeline = pipeline.unwrap();

// Record bundle with raw API
const bundleEncoder = device.createRenderBundleEncoder({
  colorFormats: ["bgra8unorm"],
});
bundleEncoder.setPipeline(rawPipeline);
// ... record commands
const bundle = bundleEncoder.finish();
```

:::note[Hybrid Approach]
Use TypeGPU for type-safe pipeline creation, then extract raw WebGPU objects for bundle recording when maximum performance is needed.
:::
