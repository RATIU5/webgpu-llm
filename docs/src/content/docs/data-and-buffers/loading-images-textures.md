---
title: Loading Images and Textures
sidebar:
  order: 30
---

## Overview

WebGPU provides multiple methods for loading image data into textures. The approach depends on your source: files, canvas elements, video frames, or compressed formats.

:::note[Core Pattern]
Most image loading follows: **fetch → decode → upload**. Use `createImageBitmap()` for async decoding, then `copyExternalImageToTexture()` for GPU upload.
:::

## Loading from URLs

### Basic Image Loading

```javascript title="Load image from URL" {5-6,13-16}
async function loadTexture(device, url) {
  // 1. Fetch and decode
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);

  // 2. Create texture
  const texture = device.createTexture({
    size: [imageBitmap.width, imageBitmap.height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  // 3. Upload to GPU
  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture },
    [imageBitmap.width, imageBitmap.height]
  );

  return texture;
}
```

### createImageBitmap Options

```javascript title="Control decode behavior"
const imageBitmap = await createImageBitmap(blob, {
  // Prevent color space conversion (preserve raw values)
  colorSpaceConversion: "none",

  // Resize during decode (more efficient than GPU resize)
  resizeWidth: 512,
  resizeHeight: 512,
  resizeQuality: "high", // "pixelated" | "low" | "medium" | "high"

  // Pre-multiply alpha (matches WebGPU default)
  premultiplyAlpha: "premultiply", // "none" | "premultiply" | "default"
});
```

:::tip[Image Formats]
Prefer **WebP** for best size/quality ratio. **PNG** for lossless. **JPG** for photos. Avoid GIF and BMP for textures.
:::

## copyExternalImageToTexture

### Parameters

```javascript title="Full parameter options"
device.queue.copyExternalImageToTexture(
  // Source
  {
    source: imageBitmap,  // ImageBitmap, HTMLCanvasElement, HTMLVideoElement, etc.
    origin: [0, 0],       // Optional: source region offset
    flipY: false,         // Optional: flip vertically
  },
  // Destination
  {
    texture: gpuTexture,
    origin: [0, 0, 0],          // Optional: write location [x, y, layer]
    mipLevel: 0,                // Optional: target mip level
    colorSpace: "srgb",         // Optional: "srgb" | "display-p3"
    premultipliedAlpha: false,  // Optional: premultiply RGB by alpha
  },
  // Copy size
  [width, height]  // Or { width, height, depthOrArrayLayers }
);
```

### Valid Source Types

| Source | Notes |
|--------|-------|
| `ImageBitmap` | Preferred—async decode |
| `HTMLCanvasElement` | Direct copy, no decode needed |
| `OffscreenCanvas` | Worker-compatible |
| `HTMLVideoElement` | Current frame (use `importExternalTexture` for video) |
| `VideoFrame` | WebCodecs integration |
| `ImageData` | From canvas `getImageData()` |

:::caution[Avoid HTMLImageElement]
Passing `HTMLImageElement` directly can cause synchronous decoding, blocking the main thread. Always use `createImageBitmap()` first.
:::

### Handling Y-Axis Orientation

WebGPU textures have origin at top-left. Some image formats store bottom-to-top:

```javascript title="Flip image on upload"
device.queue.copyExternalImageToTexture(
  { source: imageBitmap, flipY: true },
  { texture },
  [imageBitmap.width, imageBitmap.height]
);
```

:::tip[When to Flip]
- **OpenGL assets**: Often stored bottom-up
- **3D model textures**: Check format conventions
- **Screenshots**: Usually top-down (no flip needed)
:::

## Required Texture Usage Flags

```javascript title="Texture for sampling"
const texture = device.createTexture({
  size: [width, height],
  format: "rgba8unorm",
  usage:
    GPUTextureUsage.TEXTURE_BINDING |  // Use in shaders
    GPUTextureUsage.COPY_DST,          // Required for copyExternalImageToTexture
});
```

If generating mipmaps via render pass, add `RENDER_ATTACHMENT`:

```javascript title="Texture with mipmap support"
const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

const texture = device.createTexture({
  size: [width, height],
  format: "rgba8unorm",
  mipLevelCount,
  usage:
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_DST |
    GPUTextureUsage.RENDER_ATTACHMENT,  // For mipmap generation
});
```

## Canvas Textures

Canvas elements can be copied directly without `createImageBitmap()`:

```javascript title="Copy from canvas"
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");

// Draw something
ctx.fillStyle = "red";
ctx.fillRect(0, 0, 100, 100);

// Copy to texture
const texture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

device.queue.copyExternalImageToTexture(
  { source: canvas },
  { texture },
  [canvas.width, canvas.height]
);
```

## Video Textures

### External Textures (Recommended)

For video playback, `importExternalTexture()` provides a zero-copy fast path:

```javascript title="Video texture" {7,12-13}
const video = document.createElement("video");
video.src = "video.mp4";
video.loop = true;
await video.play();

function frame() {
  // Create external texture each frame
  const externalTexture = device.importExternalTexture({ source: video });

  // Use in render pass...

  // Must use immediately—destroyed at end of task
  requestAnimationFrame(frame);
}
```

:::danger[External Texture Lifetime]
External textures are automatically destroyed when JavaScript returns to the browser event loop. You must create and use them within the same callback:

```javascript
// ✗ Wrong: texture destroyed before use
const tex = device.importExternalTexture({ source: video });
await someAsyncOperation();  // Crossing await boundary
renderWithTexture(tex);       // Error: texture expired

// ✓ Correct: create and use in same frame
requestAnimationFrame(() => {
  const tex = device.importExternalTexture({ source: video });
  renderWithTexture(tex);
});
```
:::

### WGSL for External Textures

```wgsl title="Sampling external texture"
@group(0) @binding(0) var externalTex: texture_external;
@group(0) @binding(1) var samp: sampler;

@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureSampleBaseClampToEdge(externalTex, samp, uv);
}
```

### Video via copyExternalImageToTexture

For persistent video textures (slower but longer-lived):

```javascript title="Copy video frame to regular texture"
function frame() {
  if (video.readyState >= video.HAVE_CURRENT_DATA) {
    device.queue.copyExternalImageToTexture(
      { source: video },
      { texture: videoTexture },
      [video.videoWidth, video.videoHeight]
    );
  }
  requestAnimationFrame(frame);
}
```

## Compressed Textures

Compressed textures reduce GPU memory and bandwidth. WebGPU supports three formats, but availability depends on hardware.

### Format Support

| Format | Desktop | Mobile | Feature Name |
|--------|---------|--------|--------------|
| BC (DXT/S3TC) | ✓ Windows/Mac | ✗ | `texture-compression-bc` |
| ETC2 | ✗ | ✓ iOS/Android | `texture-compression-etc2` |
| ASTC | ✗ | ✓ Newer mobile | `texture-compression-astc` |

:::note[Guaranteed Coverage]
All WebGPU implementations support either BC **or** both ETC2 and ASTC. You're guaranteed at least one compressed format.
:::

### Checking Feature Support

```javascript title="Query compression support"
const adapter = await navigator.gpu.requestAdapter();

const hasBC = adapter.features.has("texture-compression-bc");
const hasETC2 = adapter.features.has("texture-compression-etc2");
const hasASTC = adapter.features.has("texture-compression-astc");

// Request device with needed features
const device = await adapter.requestDevice({
  requiredFeatures: hasBC ? ["texture-compression-bc"] : [],
});
```

### Loading Compressed Data

Compressed textures use `writeTexture()` instead of `copyExternalImageToTexture()`:

```javascript title="Load BC/DXT compressed texture"
// Assume compressedData is Uint8Array from .dds or .ktx2 file
const texture = device.createTexture({
  size: [width, height],
  format: "bc1-rgba-unorm",  // DXT1
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

device.queue.writeTexture(
  { texture },
  compressedData,
  {
    bytesPerRow: Math.ceil(width / 4) * 8,  // BC1: 8 bytes per 4x4 block
    rowsPerImage: Math.ceil(height / 4),
  },
  [width, height]
);
```

### Common Compressed Formats

<details>
<summary>**BC Formats (Desktop)**</summary>

| Format | Bits/Pixel | Alpha | Use Case |
|--------|------------|-------|----------|
| `bc1-rgba-unorm` | 4 | 1-bit | Opaque or cutout |
| `bc3-rgba-unorm` | 8 | Full | Full alpha |
| `bc4-r-unorm` | 4 | N/A | Grayscale, height maps |
| `bc5-rg-unorm` | 8 | N/A | Normal maps |
| `bc7-rgba-unorm` | 8 | Full | High-quality RGBA |

</details>

<details>
<summary>**ETC2 Formats (Mobile)**</summary>

| Format | Bits/Pixel | Alpha | Use Case |
|--------|------------|-------|----------|
| `etc2-rgb8unorm` | 4 | None | Opaque |
| `etc2-rgba8unorm` | 8 | Full | Full alpha |
| `etc2-rgb8a1unorm` | 4 | 1-bit | Cutout alpha |

</details>

<details>
<summary>**ASTC Formats (Modern Mobile)**</summary>

| Format | Block Size | Bits/Pixel |
|--------|------------|------------|
| `astc-4x4-unorm` | 4×4 | 8 |
| `astc-6x6-unorm` | 6×6 | 3.56 |
| `astc-8x8-unorm` | 8×8 | 2 |

Larger blocks = smaller files, lower quality.

</details>

### Universal Texture Containers

**KTX2** files can contain multiple formats, transcoded at runtime:

```javascript title="KTX2 loading pattern"
// Using a KTX2 loader library
import { KTX2Loader } from "ktx2-loader";

const loader = new KTX2Loader(device);
const texture = await loader.load("texture.ktx2");
// Automatically transcodes to BC, ETC2, or ASTC based on device
```

:::tip[Best Practice]
Ship KTX2 with UASTC or ETC1S encoding. Transcode to device-native format at load time for universal compatibility.
:::

## TypeGPU Image Loading

```typescript title="TypeGPU texture from image"
import tgpu from "typegpu";

const root = await tgpu.init();

// Load image
const response = await fetch("texture.png");
const blob = await response.blob();
const imageBitmap = await createImageBitmap(blob);

// Create texture with TypeGPU
const texture = root.createTexture({
  size: [imageBitmap.width, imageBitmap.height],
  format: "rgba8unorm",
}).$usage("sampled");

// Upload using raw device
const device = root.unwrap().device;
device.queue.copyExternalImageToTexture(
  { source: imageBitmap },
  { texture: root.unwrap(texture) },
  [imageBitmap.width, imageBitmap.height]
);
```

## Complete Example

```javascript title="Texture loader utility"
class TextureLoader {
  constructor(device) {
    this.device = device;
  }

  async fromURL(url, options = {}) {
    const { flipY = false, generateMipmaps = false } = options;

    const response = await fetch(url);
    const blob = await response.blob();
    const imageBitmap = await createImageBitmap(blob, {
      colorSpaceConversion: "none",
    });

    const { width, height } = imageBitmap;
    const mipLevelCount = generateMipmaps
      ? Math.floor(Math.log2(Math.max(width, height))) + 1
      : 1;

    const texture = this.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      mipLevelCount,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        (generateMipmaps ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
    });

    this.device.queue.copyExternalImageToTexture(
      { source: imageBitmap, flipY },
      { texture },
      [width, height]
    );

    if (generateMipmaps) {
      this.generateMipmaps(texture, mipLevelCount);
    }

    return texture;
  }

  generateMipmaps(texture, levelCount) {
    // See textures-samplers.md for mipmap generation
  }
}
```

## Resources

:::note[References]
- [WebGPU Fundamentals: Importing Textures](https://webgpufundamentals.org/webgpu/lessons/webgpu-importing-textures.html)
- [Toji's Image Texture Best Practices](https://toji.dev/webgpu-best-practices/img-textures.html)
- [MDN: copyExternalImageToTexture](https://developer.mozilla.org/en-US/docs/Web/API/GPUQueue/copyExternalImageToTexture)
- [WebGPU Spec: Compressed Textures](https://www.w3.org/TR/webgpu/#texture-compression)
:::
