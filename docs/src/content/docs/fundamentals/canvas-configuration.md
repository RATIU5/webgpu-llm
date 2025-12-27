---
title: Canvas Configuration and Context
sidebar:
  order: 20
---

## Overview

The HTML canvas element serves as the rendering surface for WebGPU applications. Before rendering can occur, you must configure the canvas context to establish the connection between your WebGPU device and the display surface.

:::note[Requirements]
WebGPU requires a secure context (HTTPS or localhost) and returns either `'bgra8unorm'` or `'rgba8unorm'` as the preferred format depending on platform.
:::

## Obtaining the Context

```javascript title="Getting WebGPU context"
const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");

if (!context) {
  console.error("WebGPU not supported");
  return;
}
```

The `GPUCanvasContext` interface provides:

| Method | Description |
|--------|-------------|
| `configure(config)` | Sets up context with GPU device and rendering parameters |
| `getCurrentTexture()` | Returns the next `GPUTexture` for the current frame |
| `unconfigure()` | Removes configuration and destroys textures |
| `getConfiguration()` | Returns the current configuration |

## Preferred Canvas Format

Different platforms prefer different texture formats. Use `navigator.gpu.getPreferredCanvasFormat()` for optimal performance:

```javascript title="Get preferred format"
const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
// Returns 'bgra8unorm' or 'rgba8unorm'
```

| Platform | Preferred Format |
|----------|-----------------|
| macOS/Metal | `bgra8unorm` (IOSurface requirement) |
| Windows/D3D12 | `rgba8unorm` |
| Linux/Vulkan | Varies by driver |

:::caution
Using the non-preferred format causes unnecessary format conversions and memory copies.
:::

## Context Configuration

Configure the context once during initialization:

```javascript title="Complete context configuration"
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();

context.configure({
  device: device,
  format: format,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
  alphaMode: "opaque",
  colorSpace: "srgb",
});
```

### Configuration Options

<details>
<summary>**device** (required)</summary>

The `GPUDevice` that produces content for this canvas.

</details>

<details>
<summary>**format** (required)</summary>

Use `navigator.gpu.getPreferredCanvasFormat()`. Both `bgra8unorm` and `rgba8unorm` use 8 bits per channel.

</details>

<details>
<summary>**usage**</summary>

Defines operations on canvas textures. Default is `GPUTextureUsage.RENDER_ATTACHMENT`. Add `COPY_SRC` for reading back canvas contents:

```javascript
usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
```

</details>

<details>
<summary>**alphaMode**</summary>

How alpha affects compositing with the page:
- `'opaque'` (default): Alpha ignored, pixels fully opaque
- `'premultiplied'`: Alpha determines transparency, colors must be premultiplied

</details>

<details>
<summary>**colorSpace**</summary>

Color space interpretation. `'srgb'` is standard for web content.

</details>

## Canvas Sizing

The canvas has two sizes:
1. **Display size** — CSS dimensions (how large it appears)
2. **Resolution** — `width`/`height` attributes (actual pixel dimensions)

For sharp rendering, match resolution to device pixels:

```javascript title="Match device pixel ratio"
const dpr = window.devicePixelRatio || 1;
canvas.width = Math.min(
  canvas.clientWidth * dpr,
  device.limits.maxTextureDimension2D
);
canvas.height = Math.min(
  canvas.clientHeight * dpr,
  device.limits.maxTextureDimension2D
);
```

:::note
The default resolution is 300×150 if not specified.
:::

### High-DPI Support with ResizeObserver

Use `ResizeObserver` with `devicePixelContentBoxSize` for pixel-perfect rendering:

```javascript title="ResizeObserver for high-DPI" {4-6}
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    let width, height;

    if (entry.devicePixelContentBoxSize) {
      // Most accurate - actual device pixels
      width = entry.devicePixelContentBoxSize[0].inlineSize;
      height = entry.devicePixelContentBoxSize[0].blockSize;
    } else {
      // Fallback - CSS pixels × devicePixelRatio
      const dpr = window.devicePixelRatio;
      width = entry.contentBoxSize[0].inlineSize * dpr;
      height = entry.contentBoxSize[0].blockSize * dpr;
    }

    entry.target.width = Math.max(
      1,
      Math.min(Math.floor(width), device.limits.maxTextureDimension2D)
    );
    entry.target.height = Math.max(
      1,
      Math.min(Math.floor(height), device.limits.maxTextureDimension2D)
    );
  }
});

observer.observe(canvas);
```

:::caution
Only update canvas dimensions when they actually change—setting `canvas.width` or `canvas.height` triggers expensive operations.
:::

## Getting the Current Texture

Call `getCurrentTexture()` once per frame within your render loop:

```javascript title="Basic render loop"
function render() {
  const texture = context.getCurrentTexture();
  const view = texture.createView();

  const encoder = device.createCommandEncoder();
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: view,
      clearValue: { r: 0.0, g: 0.0, b: 0.5, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });

  // Rendering commands...

  renderPass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(render);
}
```

:::danger[Don't Cache Textures]
The texture is valid only for the current frame and automatically presented after queue submission. Get a fresh texture each frame—do not cache it.
:::

## Alpha Compositing

### Premultiplied Alpha

In premultiplied alpha, RGB values are multiplied by alpha before storage:

| Type | Red at 50% |
|------|------------|
| Standard | `(1.0, 0.0, 0.0, 0.5)` |
| Premultiplied | `(0.5, 0.0, 0.0, 0.5)` (RGB × A) |

Configure for transparency:

```javascript title="Enable premultiplied alpha"
context.configure({
  device,
  format,
  alphaMode: "premultiplied",
});
```

Fragment shader must output premultiplied colors:

```wgsl title="Premultiply in shader"
@fragment
fn fragmentMain() -> @location(0) vec4f {
  var color = vec4f(1.0, 0.0, 0.0, 0.5);
  // Premultiply
  color = vec4f(color.rgb * color.a, color.a);
  return color;  // (0.5, 0.0, 0.0, 0.5)
}
```

Or use blend state:

```javascript title="Blend state for premultiplied"
blend: {
  color: {
    srcFactor: "one",
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

## Multiple Canvases

Render to multiple canvases with a single device:

```javascript title="Multi-canvas rendering"
const context1 = canvas1.getContext("webgpu");
const context2 = canvas2.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();

context1.configure({ device, format });
context2.configure({ device, format });

function render() {
  const encoder = device.createCommandEncoder();

  const pass1 = encoder.beginRenderPass({
    colorAttachments: [{
      view: context1.getCurrentTexture().createView(),
      clearValue: { r: 1, g: 0, b: 0, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass1.end();

  const pass2 = encoder.beginRenderPass({
    colorAttachments: [{
      view: context2.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 1, b: 0, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass2.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(render);
}
```

## Storage Texture Usage

To write to canvas from compute shaders, check for `bgra8unorm-storage`:

```javascript title="Enable storage texture usage"
const adapter = await navigator.gpu.requestAdapter();
const hasStorage = adapter.features.has("bgra8unorm-storage");

const device = await adapter.requestDevice({
  requiredFeatures: hasStorage ? ["bgra8unorm-storage"] : [],
});

if (hasStorage || format === "rgba8unorm") {
  context.configure({
    device,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
  });
}
```

## Complete Example

```javascript title="Full canvas setup" {14-18,21-37}
async function initWebGPU() {
  if (!navigator.gpu) {
    console.error("WebGPU not supported");
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("No WebGPU adapter found");
    return null;
  }

  const device = await adapter.requestDevice();
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      let width, height;
      if (entry.devicePixelContentBoxSize) {
        width = entry.devicePixelContentBoxSize[0].inlineSize;
        height = entry.devicePixelContentBoxSize[0].blockSize;
      } else {
        const dpr = window.devicePixelRatio;
        width = entry.contentBoxSize[0].inlineSize * dpr;
        height = entry.contentBoxSize[0].blockSize * dpr;
      }

      canvas.width = Math.max(1, Math.min(Math.floor(width), device.limits.maxTextureDimension2D));
      canvas.height = Math.max(1, Math.min(Math.floor(height), device.limits.maxTextureDimension2D));
    }
  });
  observer.observe(canvas);

  return { device, context, format };
}

function render(device, context) {
  const encoder = device.createCommandEncoder();
  const view = context.getCurrentTexture().createView();

  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view,
      clearValue: { r: 0.0, g: 0.0, b: 0.4, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.end();

  device.queue.submit([encoder.finish()]);
}

async function main() {
  const webgpu = await initWebGPU();
  if (!webgpu) return;

  function frame() {
    render(webgpu.device, webgpu.context);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
```
