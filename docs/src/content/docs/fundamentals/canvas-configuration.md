---
title: Canvas Configuration and Context
sidebar:
  order: 20
---

## Overview

The HTML canvas element serves as the rendering surface for WebGPU applications, bridging the gap between GPU-accelerated computations and visual display in the browser. Before any rendering can occur, you must properly configure the canvas context to establish the connection between your WebGPU device and the display surface. This configuration process involves obtaining a WebGPU context, selecting appropriate texture formats, configuring alpha blending modes, and handling canvas sizing for various display scenarios including high-DPI screens.

Unlike traditional 2D canvas contexts, WebGPU's canvas configuration requires careful attention to GPU capabilities, platform-specific format preferences, and color management to ensure optimal performance and visual quality across different devices and operating systems.

## Key Concepts

### HTMLCanvasElement.getContext('webgpu')

The first step in setting up WebGPU rendering is obtaining a WebGPU context from your canvas element. This is accomplished by calling `getContext()` with the `'webgpu'` string identifier:

```javascript
const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
```

This method returns a `GPUCanvasContext` object if WebGPU is supported, or `null` if it's not available. Always check for null to handle browsers without WebGPU support:

```javascript
if (!context) {
  console.error("WebGPU not supported");
  return;
}
```

It's important to note that WebGPU requires a secure context (HTTPS or localhost) and may not be available in all browsers. The context object is unique to each canvas and manages the lifecycle of textures presented to that canvas.

### GPUCanvasContext Interface

The `GPUCanvasContext` interface provides methods for configuring and managing the canvas rendering context. Key methods include:

- **`configure(configuration)`** - Sets up the context with a GPU device and rendering parameters. This clears the canvas to transparent black and establishes how rendered content will be presented.

- **`getCurrentTexture()`** - Returns the next `GPUTexture` that will be composited to the document. This texture serves as the render target for the current frame. You must call this method within your render loop to obtain the texture for each frame.

- **`unconfigure()`** - Removes the current configuration and destroys any textures created while the context was configured. This is useful for cleanup or when switching devices.

- **`getConfiguration()`** - Returns the current configuration object applied to the context, allowing you to inspect the active settings.

The interface also exposes a read-only `canvas` property that returns a reference to the associated HTMLCanvasElement.

### GPU.getPreferredCanvasFormat()

Different platforms and graphics APIs have preferences for specific texture formats. The `GPU.getPreferredCanvasFormat()` method returns the optimal canvas texture format for the user's system:

```javascript
const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
// Returns either 'bgra8unorm' or 'rgba8unorm'
```

This method is essential for maximizing performance, as using the preferred format avoids unnecessary format conversions and memory copies during presentation. The returned format will be either `'bgra8unorm'` or `'rgba8unorm'`, both of which use 8 bits per channel with unsigned normalized values.

**Always use this method** rather than hardcoding a format, as the preference varies by platform:

- macOS/Metal typically prefers `'bgra8unorm'` due to IOSurface requirements
- Windows/D3D12 and Linux/Vulkan often prefer `'rgba8unorm'`

### Color Space and Tone Mapping

WebGPU provides control over color space interpretation and tone mapping for HDR content:

- **Color Space**: The `colorSpace` parameter controls how color values are interpreted during presentation. Options include `'srgb'` (the default) for standard sRGB color space. Proper color space management ensures colors appear consistently across different displays.

- **Tone Mapping**: The `toneMapping` configuration manages dynamic range mapping when presenting content to the display. This becomes particularly important when working with high dynamic range (HDR) content that exceeds the display's capability.

## Canvas Setup

### HTML Canvas Element Configuration

Begin with a properly configured HTML canvas element in your markup:

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>WebGPU Application</title>
    <style>
      body {
        margin: 0;
        overflow: hidden;
      }
      canvas {
        display: block;
        width: 100vw;
        height: 100vh;
      }
    </style>
  </head>
  <body>
    <canvas id="webgpu-canvas"></canvas>
  </body>
</html>
```

### Canvas Sizing

The canvas element has two distinct sizes:

1. **Display size** - Controlled by CSS, determines how large the canvas appears on the page
2. **Resolution** - Set via the `width` and `height` attributes, determines the actual pixel dimensions

For pixel-perfect rendering, these should match the actual device pixels:

```javascript
// Basic sizing (not DPI-aware)
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;
```

The default canvas resolution is 300×150 pixels if not specified. Always constrain dimensions to the device's maximum supported texture size:

```javascript
canvas.width = Math.max(
  1,
  Math.min(desiredWidth, device.limits.maxTextureDimension2D),
);
canvas.height = Math.max(
  1,
  Math.min(desiredHeight, device.limits.maxTextureDimension2D),
);
```

### CSS Considerations

When styling your canvas, consider these important factors:

```css
canvas {
  /* Prevent unwanted default sizing */
  display: block;

  /* Remove inline-block spacing */
  vertical-align: top;

  /* Prevent image smoothing when CSS size differs from resolution */
  image-rendering: pixelated; /* Or 'crisp-edges' */

  /* For transparent canvases */
  background: transparent;
}
```

The `display: block` declaration prevents the small gap that appears below inline elements, while `image-rendering` controls how the browser scales the canvas when CSS dimensions don't match the resolution.

## Context Configuration

The `configure()` method establishes all rendering parameters for the canvas. Here's a complete configuration example:

```javascript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

context.configure({
  device: device,
  format: presentationFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
  alphaMode: "premultiplied",
  viewFormats: [],
  colorSpace: "srgb",
});
```

### Configuration Options Explained

#### device

The `GPUDevice` that will produce content for this canvas. This establishes the connection between WebGPU operations and canvas output:

```javascript
device: device; // Required parameter
```

You must configure the context with a valid device before rendering. The device must remain valid for the duration of rendering operations.

#### format

Specifies the texture format for the canvas surface. Always use `navigator.gpu.getPreferredCanvasFormat()`:

```javascript
format: navigator.gpu.getPreferredCanvasFormat(); // 'bgra8unorm' or 'rgba8unorm'
```

The format determines how pixel data is encoded in the canvas texture. Both supported formats use 8 bits per color channel (red, green, blue, alpha) with values normalized to the range [0, 1].

#### usage

Defines the intended operations on canvas textures. The default and most common value is `GPUTextureUsage.RENDER_ATTACHMENT`:

```javascript
usage: GPUTextureUsage.RENDER_ATTACHMENT; // Default value
```

This flag indicates the texture will be used as a render target. You can combine this with other usage flags if needed:

```javascript
// For reading back canvas contents or using as texture input
usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
```

For storage texture usage (writing directly via compute shaders), you may need the `bgra8unorm-storage` feature on certain platforms.

#### alphaMode

Determines how alpha channel values affect the canvas when composited with the page. Two modes are available:

**'opaque'** (default):

```javascript
alphaMode: "opaque";
```

The alpha channel is ignored, and all pixels are treated as fully opaque. The canvas cannot be transparent, and any alpha values in the rendered content are cleared to 1.0 during presentation.

**'premultiplied'**:

```javascript
alphaMode: "premultiplied";
```

Alpha values determine pixel transparency, and color values must be premultiplied by alpha. See the "Alpha Compositing" section below for detailed explanation.

#### viewFormats

An array of additional texture formats that can be used when creating texture views from the canvas texture:

```javascript
viewFormats: []; // Empty array by default
```

This is typically empty unless you need to create views with different formats than the configured format. For example, to create an sRGB view of a linear texture:

```javascript
viewFormats: ["rgba8unorm-srgb"];
```

#### colorSpace

Controls color space interpretation during presentation:

```javascript
colorSpace: "srgb"; // Default and most common value
```

The `'srgb'` color space is standard for most web content and ensures colors are interpreted with proper gamma correction. This affects how numerical color values are converted to physical light output.

## Texture Format Selection

### Understanding bgra8unorm vs rgba8unorm

Both `bgra8unorm` and `rgba8unorm` are 8-bit-per-channel unsigned normalized formats that store red, green, blue, and alpha values:

- **rgba8unorm**: Red, Green, Blue, Alpha order in memory
- **bgra8unorm**: Blue, Green, Red, Alpha order in memory

The WebGPU specification guarantees both formats will work on any device, but performance characteristics differ by platform.

### Platform-Specific Format Preferences

**macOS/Metal**:

- Strongly prefers `bgra8unorm`
- Canvas textures are backed by IOSurfaces, which only support BGRA format
- Using `rgba8unorm` requires emulation with intermediate textures and blits, causing significant performance overhead
- The `bgra8unorm` format can universally be used as a storage texture on Metal

**Windows/D3D12**:

- Typically prefers `rgba8unorm`
- BGRA storage textures are never supported on D3D12
- Using `bgra8unorm` without storage works fine, but limits flexibility

**Linux/Vulkan**:

- Support varies by driver
- May prefer either format depending on hardware
- BGRA storage support is inconsistent across devices

### Optimal Format Selection

Always use the preferred format to avoid performance penalties:

```javascript
// Correct approach - platform-optimized
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

// Wrong approach - hardcoded format
context.configure({ device, format: "rgba8unorm" }); // ❌ May be slow on macOS
```

### Storage Texture Considerations

If you need to write to the canvas texture from a compute shader, check for the `bgra8unorm-storage` feature:

```javascript
const adapter = await navigator.gpu.requestAdapter();
const hasStorageFeature = adapter.features.has("bgra8unorm-storage");

const device = await adapter.requestDevice({
  requiredFeatures: hasStorageFeature ? ["bgra8unorm-storage"] : [],
});

// Now you can use STORAGE_BINDING with the preferred format
if (hasStorageFeature || format === "rgba8unorm") {
  context.configure({
    device,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
  });
}
```

## Alpha Compositing

### Understanding Premultiplied Alpha

In premultiplied alpha representation, RGB color values are multiplied by the alpha value before storage. This is the mathematically correct way to represent transparent colors for compositing operations.

**Standard (non-premultiplied) representation**:

- Color: RGB(1.0, 0.0, 0.0) at 50% transparency
- Stored as: (1.0, 0.0, 0.0, 0.5)

**Premultiplied representation**:

- Same color premultiplied by alpha
- Stored as: (0.5, 0.0, 0.0, 0.5)
- RGB values are multiplied by A: (R×A, G×A, B×A, A)

### When to Use Each Mode

**Use 'opaque' when**:

- Your canvas content is always fully opaque
- You don't need transparency with the page background
- Maximum simplicity is desired (default behavior)
- The canvas covers its entire area with solid content

**Use 'premultiplied' when**:

- You need transparency effects with the page background
- Creating overlay UI elements
- Implementing transparent windows or panels
- Compositing multiple layers with alpha blending

### Implementing Premultiplied Alpha

To use premultiplied alpha, configure the context and ensure your shader outputs premultiplied colors:

```javascript
// Configuration
context.configure({
  device,
  format: presentationFormat,
  alphaMode: "premultiplied",
});

// In your render pass, clear to transparent
const renderPass = encoder.beginRenderPass({
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 }, // Transparent black
      loadOp: "clear",
      storeOp: "store",
    },
  ],
});
```

Your fragment shader must output premultiplied colors:

```wgsl
@fragment
fn fragmentMain() -> @location(0) vec4f {
  var color = vec4f(1.0, 0.0, 0.0, 0.5);  // Red at 50% alpha

  // Premultiply the RGB by alpha
  color.r *= color.a;
  color.g *= color.a;
  color.b *= color.a;

  return color;  // Returns (0.5, 0.0, 0.0, 0.5)
}
```

Alternatively, configure your blend state to handle premultiplication:

```javascript
const pipelineDescriptor = {
  // ... other configuration
  fragment: {
    targets: [
      {
        format: presentationFormat,
        blend: {
          color: {
            srcFactor: "one", // Already premultiplied
            dstFactor: "one-minus-src-alpha",
            operation: "add",
          },
          alpha: {
            srcFactor: "one",
            dstFactor: "one-minus-src-alpha",
            operation: "add",
          },
        },
      },
    ],
  },
};
```

### Blending with Page Background

When using premultiplied alpha mode, the canvas becomes part of the page's compositing tree. The page background shows through transparent areas:

```html
<style>
  body {
    background: linear-gradient(45deg, #f0f0f0 25%, #e0e0e0 25%);
  }
  canvas {
    display: block;
    /* Canvas will blend with this gradient background */
  }
</style>
```

This enables creative effects like UI overlays, transparent windows, and layered compositions without requiring framebuffer copies.

## Canvas Resizing

### Handling Window Resize

Canvas resizing is critical for responsive applications and must account for CSS dimensions, device pixel ratio, and GPU limits. The recommended approach uses `ResizeObserver`:

```javascript
function resizeCanvas(canvas, device) {
  const width = canvas.clientWidth * window.devicePixelRatio;
  const height = canvas.clientHeight * window.devicePixelRatio;

  const needResize = canvas.width !== width || canvas.height !== height;

  if (needResize) {
    canvas.width = Math.max(
      1,
      Math.min(width, device.limits.maxTextureDimension2D),
    );
    canvas.height = Math.max(
      1,
      Math.min(height, device.limits.maxTextureDimension2D),
    );
  }

  return needResize;
}

const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const canvas = entry.target;

    // Use devicePixelContentBoxSize for pixel-perfect rendering
    if (entry.devicePixelContentBoxSize) {
      const width = entry.devicePixelContentBoxSize[0].inlineSize;
      const height = entry.devicePixelContentBoxSize[0].blockSize;

      canvas.width = Math.max(
        1,
        Math.min(width, device.limits.maxTextureDimension2D),
      );
      canvas.height = Math.max(
        1,
        Math.min(height, device.limits.maxTextureDimension2D),
      );
    } else {
      // Fallback for browsers without devicePixelContentBoxSize (Safari)
      resizeCanvas(canvas, device);
    }
  }

  // Render with new dimensions
  render();
});

observer.observe(canvas);
```

### High-DPI Display Support

Modern displays often have pixel densities higher than 1:1 (Retina displays, 4K monitors, etc.). The `window.devicePixelRatio` property indicates the ratio between physical pixels and CSS pixels:

```javascript
// High-DPI aware sizing
function updateCanvasSize(canvas, device) {
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = canvas.clientWidth;
  const displayHeight = canvas.clientHeight;

  // Calculate actual pixel dimensions
  const width = Math.floor(displayWidth * dpr);
  const height = Math.floor(displayHeight * dpr);

  // Update canvas resolution
  canvas.width = Math.min(width, device.limits.maxTextureDimension2D);
  canvas.height = Math.min(height, device.limits.maxTextureDimension2D);
}
```

### devicePixelContentBoxSize for Pixel Perfection

The `devicePixelContentBoxSize` property on `ResizeObserverEntry` provides the most accurate pixel dimensions, accounting for zoom, device pixel ratio, and CSS transforms:

```javascript
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    let width, height;

    if (entry.devicePixelContentBoxSize) {
      // Most accurate - measures in actual device pixels
      width = entry.devicePixelContentBoxSize[0].inlineSize;
      height = entry.devicePixelContentBoxSize[0].blockSize;
    } else if (entry.contentBoxSize) {
      // Fallback - CSS pixels, must multiply by devicePixelRatio
      const dpr = window.devicePixelRatio;
      width = entry.contentBoxSize[0].inlineSize * dpr;
      height = entry.contentBoxSize[0].blockSize * dpr;
    } else {
      // Old browsers - use clientWidth/clientHeight
      width = entry.target.clientWidth * window.devicePixelRatio;
      height = entry.target.clientHeight * window.devicePixelRatio;
    }

    // Apply dimensions
    entry.target.width = Math.max(1, Math.floor(width));
    entry.target.height = Math.max(1, Math.floor(height));
  }

  requestAnimationFrame(render);
});
```

**Important**: Only update canvas dimensions when they actually change, as setting `canvas.width` or `canvas.height` triggers expensive operations in some browsers, including immediate drawing buffer replacement.

### ResizeObserver Best Practices

1. **Check for actual size changes** before updating dimensions
2. **Defer rendering** - don't call render directly in the observer callback
3. **Update dependent resources** - recreate depth textures and framebuffer attachments
4. **Debounce if needed** - for very frequent resize events, consider debouncing

```javascript
let resizeTimeout;
const observer = new ResizeObserver((entries) => {
  clearTimeout(resizeTimeout);

  resizeTimeout = setTimeout(() => {
    for (const entry of entries) {
      updateCanvasDimensions(entry);
    }

    // Recreate size-dependent resources
    recreateDepthTexture();

    // Render with new size
    requestAnimationFrame(render);
  }, 100);
});
```

## Getting the Current Texture

The `getCurrentTexture()` method returns the texture that will be composited to the display for the current frame. Call this method once per frame within your render function:

```javascript
function render() {
  // Get the current canvas texture
  const canvasTexture = context.getCurrentTexture();
  const textureView = canvasTexture.createView();

  const encoder = device.createCommandEncoder();
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.5, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  // Rendering commands...

  renderPass.end();
  device.queue.submit([encoder.finish()]);

  // Texture is automatically presented after submission
  requestAnimationFrame(render);
}
```

### Texture Lifecycle

The texture returned by `getCurrentTexture()`:

- Is valid only for the current frame
- Is automatically presented to the display after queue submission
- Must not be used after the frame completes
- Should not be cached or reused across frames

Always obtain a fresh texture for each frame:

```javascript
// ❌ Wrong - don't cache the texture
const texture = context.getCurrentTexture();
function render() {
  // Using stale texture causes errors
}

// ✅ Correct - get fresh texture each frame
function render() {
  const texture = context.getCurrentTexture();
  // Use texture for this frame only
}
```

### Frame Timing

The canvas texture and display presentation are synchronized with the browser's animation frame timing. Using `requestAnimationFrame` ensures your rendering stays synchronized:

```javascript
function frame(timestamp) {
  // Timestamp in milliseconds since page load
  updateAnimation(timestamp);

  const commandEncoder = device.createCommandEncoder();
  const textureView = context.getCurrentTexture().createView();

  // Render frame...

  device.queue.submit([commandEncoder.finish()]);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
```

## Multiple Canvas Support

WebGPU allows rendering to multiple canvases using a single device, enabling complex multi-view applications:

```javascript
// Set up multiple canvases
const canvas1 = document.getElementById("canvas1");
const canvas2 = document.getElementById("canvas2");

const context1 = canvas1.getContext("webgpu");
const context2 = canvas2.getContext("webgpu");

// Configure both with the same device
const device = await adapter.requestDevice();
const format = navigator.gpu.getPreferredCanvasFormat();

context1.configure({ device, format });
context2.configure({ device, format });

// Render to both canvases
function render() {
  const encoder = device.createCommandEncoder();

  // Render to first canvas
  const pass1 = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context1.getCurrentTexture().createView(),
        clearValue: { r: 1, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  // ... render commands for canvas1 ...
  pass1.end();

  // Render to second canvas
  const pass2 = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context2.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 1, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  // ... render commands for canvas2 ...
  pass2.end();

  // Submit all rendering
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(render);
}
```

This technique is useful for:

- Multi-monitor applications
- Picture-in-picture views
- Split-screen rendering
- Debug visualization alongside main view

## Best Practices and Common Pitfalls

### Best Practices

1. **Always use `getPreferredCanvasFormat()`** - Never hardcode texture formats. Platform preferences significantly impact performance.

2. **Configure once, render many** - Call `configure()` during initialization, not every frame. Reconfiguring is expensive.

3. **Handle resize efficiently** - Only update canvas dimensions when they actually change. Use `ResizeObserver` with `devicePixelContentBoxSize`.

4. **Match canvas resolution to display pixels** - Multiply CSS dimensions by `devicePixelRatio` for sharp rendering on high-DPI screens.

5. **Respect GPU limits** - Always clamp dimensions to `device.limits.maxTextureDimension2D`.

6. **Get texture per frame** - Call `getCurrentTexture()` once per frame within your render loop, not outside it.

7. **Use premultiplied alpha correctly** - When using `alphaMode: 'premultiplied'`, ensure your shaders output premultiplied colors (RGB multiplied by A).

8. **Update dependent resources on resize** - Recreate depth textures, shadow maps, and other size-dependent resources when canvas dimensions change.

9. **Use appropriate alpha mode** - Default to `'opaque'` for simpler code; only use `'premultiplied'` when transparency is needed.

10. **Consider storage texture features** - If writing to canvas from compute shaders, request the `bgra8unorm-storage` feature if available.

### Common Pitfalls

1. **Hardcoding texture format**

   ```javascript
   // ❌ Wrong - may be slow on some platforms
   context.configure({ device, format: "rgba8unorm" });

   // ✅ Correct - uses optimal format
   context.configure({
     device,
     format: navigator.gpu.getPreferredCanvasFormat(),
   });
   ```

2. **Forgetting devicePixelRatio**

   ```javascript
   // ❌ Wrong - blurry on high-DPI displays
   canvas.width = canvas.clientWidth;

   // ✅ Correct - sharp on all displays
   canvas.width = canvas.clientWidth * window.devicePixelRatio;
   ```

3. **Caching canvas texture**

   ```javascript
   // ❌ Wrong - texture is stale after first frame
   const texture = context.getCurrentTexture();
   function render() {
     const view = texture.createView(); // Error!
   }

   // ✅ Correct - fresh texture each frame
   function render() {
     const texture = context.getCurrentTexture();
     const view = texture.createView();
   }
   ```

4. **Not constraining dimensions**

   ```javascript
   // ❌ Wrong - may exceed GPU limits
   canvas.width = window.innerWidth * 4;

   // ✅ Correct - respects limits
   canvas.width = Math.min(
     window.innerWidth * devicePixelRatio,
     device.limits.maxTextureDimension2D,
   );
   ```

5. **Resizing in every frame**

   ```javascript
   // ❌ Wrong - expensive operation every frame
   function render() {
     canvas.width = canvas.clientWidth * devicePixelRatio;
     // ...
   }

   // ✅ Correct - only resize when needed
   const observer = new ResizeObserver(() => {
     canvas.width = canvas.clientWidth * devicePixelRatio;
   });
   ```

6. **Wrong alpha compositing**

   ```javascript
   // ❌ Wrong - non-premultiplied colors with premultiplied mode
   context.configure({ device, format, alphaMode: "premultiplied" });
   // Fragment shader outputs (1.0, 0.0, 0.0, 0.5) - not premultiplied

   // ✅ Correct - premultiplied colors
   // Fragment shader outputs (0.5, 0.0, 0.0, 0.5) - RGB * A
   ```

7. **Ignoring canvas aspect ratio**

   ```javascript
   // ❌ Wrong - scene gets stretched
   const projection = mat4.perspective(fov, 1.0, near, far);

   // ✅ Correct - maintains aspect ratio
   const aspect = canvas.width / canvas.height;
   const projection = mat4.perspective(fov, aspect, near, far);
   ```

8. **Reconfiguring every frame**

   ```javascript
   // ❌ Wrong - expensive reconfiguration
   function render() {
     context.configure({ device, format: preferredFormat });
     // ...
   }

   // ✅ Correct - configure once
   context.configure({ device, format: preferredFormat });
   function render() {
     // Just render
   }
   ```

9. **Using canvas before configuration**

   ```javascript
   // ❌ Wrong - context not configured yet
   const texture = context.getCurrentTexture(); // Error!
   context.configure({ device, format });

   // ✅ Correct - configure first
   context.configure({ device, format });
   const texture = context.getCurrentTexture();
   ```

10. **Not handling configuration failure**

    ```javascript
    // ❌ Wrong - no error handling
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    // ✅ Correct - handle missing support
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.error("WebGPU not supported");
      return;
    }
    const device = await adapter.requestDevice();
    ```

## Cross-References

For more information on related WebGPU topics, see:

- **[webgpu-core-concepts.md](./webgpu-core-concepts.md)** - Fundamental WebGPU concepts including devices, adapters, and the overall API structure
- **[render-pipelines.md](./render-pipelines.md)** - Details on creating render pipelines that output to canvas textures, including blend states for alpha compositing

## Complete Example

Here's a complete, production-ready example incorporating all best practices:

```javascript
async function initWebGPU() {
  // Check for WebGPU support
  if (!navigator.gpu) {
    console.error("WebGPU not supported");
    return null;
  }

  // Request adapter and device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("No WebGPU adapter found");
    return null;
  }

  const device = await adapter.requestDevice();

  // Set up canvas
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");

  // Configure context with optimal settings
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "opaque",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Handle resize with high-DPI support
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

      canvas.width = Math.max(
        1,
        Math.min(Math.floor(width), device.limits.maxTextureDimension2D),
      );
      canvas.height = Math.max(
        1,
        Math.min(Math.floor(height), device.limits.maxTextureDimension2D),
      );
    }
  });
  observer.observe(canvas);

  return { device, context, presentationFormat };
}

// Render function
function render(device, context) {
  const commandEncoder = device.createCommandEncoder();
  const textureView = context.getCurrentTexture().createView();

  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.4, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  // Your rendering commands here...

  renderPass.end();
  device.queue.submit([commandEncoder.finish()]);
}

// Application entry point
async function main() {
  const webgpu = await initWebGPU();
  if (!webgpu) return;

  const { device, context, presentationFormat } = webgpu;

  // Create your render pipeline and resources...

  // Start render loop
  function frame() {
    render(device, context);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
```

---

## Sources

This documentation was compiled from the following authoritative sources:

- [WebGPU Specification - W3C](https://www.w3.org/TR/webgpu/)
- [GPUCanvasContext - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/GPUCanvasContext)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [WebGPU Fundamentals - Basics](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html)
- [WebGPU Transparency and Blending](https://webgpufundamentals.org/webgpu/lessons/webgpu-transparency.html)
- [WebGPU Resizing the Canvas](https://webgpufundamentals.org/webgpu/lessons/webgpu-resizing-the-canvas.html)
- [WebGPU Unleashed: Canvas Resizing](https://shi-yan.github.io/webgpuunleashed/Control/canvas_resizing.html)
- [Pixel-perfect rendering with devicePixelContentBox](https://web.dev/device-pixel-content-box/)
- [GPUs prefer premultiplication - Real-Time Rendering](https://www.realtimerendering.com/blog/gpus-prefer-premultiplication/)
- [WebGPU Canvas Alpha Configuration Discussion](https://github.com/gpuweb/gpuweb/issues/1425)
- [WebGPU Texture Format Issues](https://github.com/gpuweb/gpuweb/issues/2535)
