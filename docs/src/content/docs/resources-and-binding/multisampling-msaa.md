---
title: Multisampling (MSAA)
sidebar:
  order: 40
---

## Overview

Multisample Anti-Aliasing (MSAA) reduces jagged edges by sampling each pixel multiple times. WebGPU v1 supports 4x MSAA, where each pixel stores 4 samples that are resolved to a single color.

:::note[How MSAA Works]
1. GPU rasterizes triangles to a 4-sample texture
2. Fragment shader runs once per pixel, result written to covered samples
3. Samples are averaged (resolved) to produce final single-sample output
:::

## Sample Counts

| Count | Description |
|-------|-------------|
| 1 | No multisampling (default) |
| 4 | 4x MSAA (only other option in WebGPU v1) |

## Setup Requirements

MSAA requires matching configuration across:
1. Multisampled color texture
2. Multisampled depth texture (if used)
3. Render pipeline
4. Render pass with resolve target

## Creating Multisampled Textures

### Color Texture

```javascript title="4x MSAA color texture" {4}
const msaaTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: navigator.gpu.getPreferredCanvasFormat(),
  sampleCount: 4,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

### Depth Texture

```javascript title="4x MSAA depth texture" {4}
const msaaDepthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  sampleCount: 4,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

:::caution[Matching Sample Counts]
Color and depth textures must have the same `sampleCount`. Mismatched counts cause validation errors.
:::

## Pipeline Configuration

```javascript title="MSAA pipeline" {4-6}
const pipeline = device.createRenderPipeline({
  layout: "auto",
  // ...vertex and fragment config...
  multisample: {
    count: 4,
  },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",
  },
});
```

## Render Pass Configuration

### With Resolve Target

```javascript title="MSAA render pass with resolve" {3-4,8}
const renderPass = encoder.beginRenderPass({
  colorAttachments: [{
    view: msaaTexture.createView(),           // Multisampled texture
    resolveTarget: context.getCurrentTexture().createView(), // Single-sample output
    loadOp: "clear",
    storeOp: "store",
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
  }],
  depthStencilAttachment: {
    view: msaaDepthTexture.createView(),
    depthLoadOp: "clear",
    depthStoreOp: "store",
    depthClearValue: 1.0,
  },
});
```

### Resolution Process

The `resolveTarget` receives the averaged result automatically when the render pass ends. No manual resolve step needed.

:::tip[When to Resolve]
For multi-pass rendering, only set `resolveTarget` on the final pass to avoid unnecessary resolve operations:

```javascript
// Pass 1: No resolve
{ view: msaaTexture.createView(), storeOp: "store" }

// Pass 2: Resolve to canvas
{ view: msaaTexture.createView(), resolveTarget: canvasView, storeOp: "store" }
```
:::

## Complete Example

```javascript title="Full MSAA setup"
const format = navigator.gpu.getPreferredCanvasFormat();

// 1. Create multisampled textures
const msaaColor = device.createTexture({
  size: [canvas.width, canvas.height],
  format: format,
  sampleCount: 4,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const msaaDepth = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  sampleCount: 4,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

// 2. Configure pipeline
const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: shaderModule, entryPoint: "vertexMain" },
  fragment: {
    module: shaderModule,
    entryPoint: "fragmentMain",
    targets: [{ format }],
  },
  multisample: { count: 4 },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less",
  },
});

// 3. Render with MSAA
function render() {
  const encoder = device.createCommandEncoder();

  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: msaaColor.createView(),
      resolveTarget: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
    }],
    depthStencilAttachment: {
      view: msaaDepth.createView(),
      depthLoadOp: "clear",
      depthStoreOp: "store",
      depthClearValue: 1.0,
    },
  });

  pass.setPipeline(pipeline);
  pass.draw(vertexCount);
  pass.end();

  device.queue.submit([encoder.finish()]);
}
```

## Reading Multisampled Textures in Shaders

Multisampled textures cannot be sampled normally. Use `textureLoad` with sample index:

```wgsl title="Reading multisampled texture"
@group(0) @binding(0) var msaaTex: texture_multisampled_2d<f32>;

@fragment
fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let coord = vec2i(pos.xy);

  // Read each sample manually
  var color = vec4f(0.0);
  color += textureLoad(msaaTex, coord, 0);
  color += textureLoad(msaaTex, coord, 1);
  color += textureLoad(msaaTex, coord, 2);
  color += textureLoad(msaaTex, coord, 3);

  return color / 4.0;  // Average
}
```

### Multisampled Depth

```wgsl title="Reading multisampled depth"
@group(0) @binding(0) var msaaDepth: texture_depth_multisampled_2d;

@fragment
fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let coord = vec2i(pos.xy);
  let depth = textureLoad(msaaDepth, coord, 0);  // Sample 0
  return vec4f(depth, depth, depth, 1.0);
}
```

:::caution[No Depth Resolve Target]
Unlike color attachments, depth attachments don't support `resolveTarget`. To access resolved depth, manually resolve in a compute or fragment shader.
:::

## Resize Handling

Recreate multisampled textures when canvas size changes:

```javascript title="Handle resize"
function onResize() {
  msaaColor.destroy();
  msaaDepth.destroy();

  msaaColor = device.createTexture({
    size: [canvas.width, canvas.height],
    format: format,
    sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  msaaDepth = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}
```

## Performance Considerations

| Aspect | Impact |
|--------|--------|
| Memory | 4x for color, 4x for depth |
| Bandwidth | Higher due to sample storage |
| Fill rate | Fragment shader runs once per pixel (not per sample) |
| Resolve | Automatic, minimal overhead |

:::tip[When to Use MSAA]
- **Use**: 3D scenes with geometric edges, low-to-medium resolution
- **Skip**: UI rendering, pixel art, very high resolution displays
- **Alternative**: Post-process AA (FXAA, TAA) for shader aliasing
:::

## Render Bundles with MSAA

```javascript title="MSAA render bundle" {3}
const bundleEncoder = device.createRenderBundleEncoder({
  colorFormats: [format],
  sampleCount: 4,
  depthStencilFormat: "depth24plus",
});

// Record commands...
const bundle = bundleEncoder.finish();
```

## Format Support

Not all formats support multisampling. Common supported formats:

| Format | Multisampling |
|--------|---------------|
| `bgra8unorm` | Yes |
| `rgba8unorm` | Yes |
| `rgba16float` | Yes |
| `depth24plus` | Yes |
| `depth32float` | Yes |
| `bgra8unorm-srgb` | No (some platforms) |

:::note[Check Support]
Query adapter limits for format-specific sample count support if needed. 4x MSAA is guaranteed for common formats.
:::
