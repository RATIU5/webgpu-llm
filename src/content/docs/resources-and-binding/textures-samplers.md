---
title: Textures and Samplers
sidebar:
  order: 20
---

Textures are fundamental to modern GPU programming, serving as containers for image data that can be efficiently accessed and filtered by specialized hardware. Unlike buffers that store linear arrays of data, textures are optimized for spatial access patterns and provide built-in interpolation, filtering, and addressing modes through samplers. This makes them ideal for storing images, lookup tables, and any data that benefits from multi-dimensional organization and hardware-accelerated sampling.

In WebGPU, textures and samplers work together to enable efficient image processing. Textures hold the actual pixel data in GPU memory, while samplers define how that data is read—controlling filtering quality, coordinate wrapping behavior, and mipmap selection. Understanding this separation of concerns is key to writing efficient, high-quality graphics code.

This comprehensive guide covers everything from basic texture creation to advanced sampling techniques, including practical examples for both native WebGPU and the TypeGPU library.

## Understanding Textures

### What Textures Are and Why They're Special

Textures are multi-dimensional arrays of data stored in GPU memory, designed specifically for efficient spatial access patterns. While buffers provide simple linear storage, textures offer several distinct advantages:

**Spatial Organization**: Textures organize data in 1D, 2D, or 3D grids, matching the natural structure of images and volumetric data. This spatial organization enables GPU caches to exploit locality—when you read one pixel, nearby pixels are often cached automatically.

**Hardware Filtering**: The GPU's texture units can automatically interpolate between adjacent texels (texture pixels) using specialized hardware. This filtering happens essentially for free, whereas implementing equivalent interpolation in a shader reading from a buffer would be significantly slower.

**Specialized Memory Layout**: Texture memory uses swizzled or tiled layouts optimized for 2D locality. When you read a 2x2 block of pixels, all four values are likely in the same cache line, unlike linear buffer layouts where adjacent rows might be far apart in memory.

**Compressed Formats**: Many texture formats support hardware-accelerated compression (BC, ETC2, ASTC), reducing memory bandwidth and storage requirements without impacting sampling performance.

### Hardware Acceleration via Samplers

Samplers are configuration objects that tell the GPU's texture units how to read texture data. This hardware acceleration is one of the most compelling reasons to use textures instead of buffers for image data:

**Bilinear and Trilinear Filtering**: Modern GPUs can interpolate between 4 (bilinear) or 8 (trilinear with mipmaps) texture samples in a single instruction, producing smooth gradients even when textures are magnified or minified.

**Anisotropic Filtering**: For surfaces viewed at oblique angles, anisotropic filtering samples multiple points along the direction of greatest distortion, dramatically improving visual quality for textured surfaces receding into the distance.

**Automatic Mipmap Selection**: The GPU automatically calculates texture coordinate derivatives to select the appropriate mipmap level, preventing flickering and aliasing when objects are far from the camera.

**Address Mode Hardware**: Wrapping, clamping, and mirroring coordinates happens in dedicated hardware before memory access, adding virtually no performance cost.

### Texture vs Buffer for Image Data

While both textures and buffers can store image data, each has distinct use cases:

**Use Textures When**:

- Sampling with interpolation (any non-point-sampled images)
- Reading the same data from multiple shader invocations (textures cache better)
- Working with compressed image formats
- Needing automatic mipmap generation or selection
- Implementing lookup tables with interpolation (color grading, noise functions)

**Use Buffers When**:

- Random write access is required (textures require `STORAGE_BINDING` usage)
- Data doesn't have spatial locality
- You need atomic operations on the data
- Implementing algorithms that don't benefit from texture caching

For most image rendering scenarios, textures provide superior performance and convenience.

## Texture Types

WebGPU supports multiple texture types, each optimized for different use cases.

### 1D, 2D, 3D Textures

**1D Textures**: Single-row arrays of texels, useful for lookup tables and gradients:

```javascript
const gradientTexture = device.createTexture({
  size: [256, 1, 1],
  dimension: "1d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

1D textures are excellent for color ramps, transfer functions, and other one-dimensional lookup tables where you want hardware interpolation.

**2D Textures**: Standard images with width and height, by far the most common texture type:

```javascript
const diffuseTexture = device.createTexture({
  size: [1024, 1024, 1],
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

2D textures are used for photographs, UI elements, sprite sheets, normal maps, and nearly all traditional texture mapping.

**3D Textures**: Volumetric data with width, height, and depth:

```javascript
const volumeTexture = device.createTexture({
  size: [128, 128, 128],
  dimension: "3d",
  format: "r32float",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

3D textures excel at volumetric effects like fog, clouds, medical imaging data, and 3D noise functions. Unlike 2D texture arrays, 3D textures support interpolation between depth slices.

### Cube Maps

Cube maps consist of six 2D textures arranged as the faces of a cube, used primarily for environment mapping and skyboxes:

```javascript
const skyboxTexture = device.createTexture({
  size: [512, 512, 6], // Width, height, 6 faces
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// Create a cube view
const cubeView = skyboxTexture.createView({
  dimension: "cube",
});
```

In shaders, cube maps are sampled using 3D direction vectors rather than 2D coordinates, making them perfect for reflections and skyboxes.

### Texture Arrays

Texture arrays store multiple independent 2D images in a single texture object:

```javascript
const textureArray = device.createTexture({
  size: [256, 256, 10], // 10 layers of 256x256 images
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

const arrayView = textureArray.createView({
  dimension: "2d-array",
});
```

Arrays are ideal for sprite animations, terrain splatmaps, or any scenario where you need to select between different textures dynamically without rebinding resources.

### Multisampled Textures

Multisampled textures store multiple samples per pixel for antialiasing:

```javascript
const msaaTexture = device.createTexture({
  size: [800, 600, 1],
  dimension: "2d",
  format: "rgba8unorm",
  sampleCount: 4, // 4x MSAA
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

Multisampled textures can only be used as render attachments and must be resolved to regular textures before being sampled in shaders.

## Creating Textures

### GPUDevice.createTexture()

The `createTexture()` method creates a new texture resource:

```javascript
const texture = device.createTexture({
  label: "Main Color Texture",
  size: { width: 512, height: 512, depthOrArrayLayers: 1 },
  dimension: "2d",
  format: "rgba8unorm",
  usage:
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_DST |
    GPUTextureUsage.RENDER_ATTACHMENT,
  mipLevelCount: 1,
  sampleCount: 1,
});
```

### GPUTextureDescriptor Options

**label**: Optional debugging string that appears in error messages and GPU profilers. Always use descriptive labels in development.

**size**: Texture dimensions specified as an object with `width`, `height`, and `depthOrArrayLayers`, or as an array `[width, height?, depth?]`. For 2D textures, `depthOrArrayLayers` represents the number of array layers.

**dimension**: One of '1d', '2d', or '3d'. Determines how the texture is indexed and sampled.

**format**: The pixel format (see section below). Must be compatible with the specified usage.

**usage**: Bitwise combination of usage flags determining how the texture can be used:

- `TEXTURE_BINDING`: Can be sampled in shaders
- `STORAGE_BINDING`: Can be used as read-write storage texture
- `RENDER_ATTACHMENT`: Can be rendered to
- `COPY_SRC` / `COPY_DST`: Can be source or destination of copy operations

### Size, mipLevelCount, sampleCount

**Size Constraints**: Texture dimensions must respect device limits:

```javascript
console.log("Max 2D texture size:", adapter.limits.maxTextureDimension2D);
console.log("Max 3D texture size:", adapter.limits.maxTextureDimension3D);

// Typical values: 8192 or 16384 for 2D, 2048 for 3D
```

**mipLevelCount**: Number of mipmap levels. For a full mipmap chain:

```javascript
const maxDimension = Math.max(width, height);
const mipLevelCount = Math.floor(Math.log2(maxDimension)) + 1;

const texture = device.createTexture({
  size: [512, 512],
  format: "rgba8unorm",
  mipLevelCount: mipLevelCount, // 10 levels for 512x512
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

Each mipmap level is half the size of the previous level (512→256→128→64→32→16→8→4→2→1).

**sampleCount**: Number of samples per pixel. Must be 1 (default) or 4 for multisampling. Multisampled textures have restricted usage—they can only be render attachments and cannot be directly sampled.

### Format Selection

Choosing the right format balances quality, memory usage, and performance:

```javascript
// Common format selection pattern
function selectTextureFormat(purpose) {
  switch (purpose) {
    case "color":
      return "rgba8unorm"; // Standard 8-bit color
    case "hdr":
      return "rgba16float"; // HDR rendering
    case "normal-map":
      return "rgba8unorm"; // Normal maps
    case "depth":
      return "depth24plus"; // Depth buffer
    case "single-channel":
      return "r8unorm"; // Grayscale or masks
    default:
      return "rgba8unorm";
  }
}
```

### Usage Flags

Multiple usage flags can be combined:

```javascript
const renderTargetTexture = device.createTexture({
  size: [800, 600],
  format: "rgba8unorm",
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT | // Can render to it
    GPUTextureUsage.TEXTURE_BINDING | // Can sample from it
    GPUTextureUsage.COPY_SRC, // Can copy from it
});
```

Be conservative with usage flags—only specify what you actually need. Some usage combinations may prevent optimizations.

## Common Texture Formats

WebGPU provides a rich set of texture formats optimized for different use cases.

### rgba8unorm: Standard Color

The most common format for color textures:

```javascript
format: "rgba8unorm";
```

- **4 channels**: Red, Green, Blue, Alpha
- **8 bits per channel**: 0-255 mapped to 0.0-1.0
- **Normalized**: Integer values normalized to floating point
- **32 bits per pixel**: 4 bytes total
- **Use for**: Photos, UI elements, sprite sheets, most color data

### bgra8unorm: Platform-Preferred

Often the preferred format for canvas rendering:

```javascript
const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
// Usually returns 'bgra8unorm'

context.configure({
  device: device,
  format: preferredFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

- **Same as rgba8unorm but with channels reordered**: B, G, R, A
- **Platform optimization**: Matches native framebuffer format on many systems
- **Use for**: Canvas render targets, swap chain images

### depth24plus: Depth Buffer

Standard depth format for 3D rendering:

```javascript
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

- **At least 24-bit depth**: Implementation may provide more precision
- **No direct shader access**: Can only be used for depth testing
- **Use for**: Standard depth buffers in 3D rendering

For shader-accessible depth, use `depth32float`:

```javascript
format: "depth32float"; // Can be sampled in shaders
```

### r32float: Single-Channel Float

High-precision single-channel data:

```javascript
const heightMap = device.createTexture({
  size: [1024, 1024],
  format: "r32float",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

- **Single channel**: Red only
- **32-bit float**: Full floating-point precision
- **Use for**: Height maps, distance fields, computational results

Other useful formats:

- `rg16float`: Two-channel half-precision (normals, gradients)
- `r8unorm`: Single channel 8-bit (masks, grayscale)
- `rgba16float`: HDR color with good precision
- `rgba32float`: Maximum precision (expensive)

## Loading Images into Textures

WebGPU provides several methods for getting image data into textures.

### createImageBitmap() from HTMLImageElement

The standard approach for loading image files:

```javascript
async function loadTexture(device, url) {
  // Load the image
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);

  // Create texture
  const texture = device.createTexture({
    size: [imageBitmap.width, imageBitmap.height, 1],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Copy image to texture
  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: texture },
    [imageBitmap.width, imageBitmap.height],
  );

  return texture;
}

// Usage
const diffuseTexture = await loadTexture(device, "textures/brick.jpg");
```

### queue.copyExternalImageToTexture()

This method efficiently copies image data from various sources to GPU textures:

```javascript
device.queue.copyExternalImageToTexture(
  {
    source: imageBitmap,
    flipY: false, // Optional: flip vertical orientation
  },
  {
    texture: texture,
    mipLevel: 0, // Which mip level to write
    origin: [0, 0, 0], // Offset within texture
  },
  [width, height, 1], // Copy size
);
```

The source can be:

- `ImageBitmap`
- `HTMLCanvasElement`
- `OffscreenCanvas`
- `HTMLVideoElement` (for video textures)

### ImageData and Canvas Sources

You can also use Canvas 2D API to generate or manipulate image data:

```javascript
function createProceduralTexture(device, size) {
  // Create canvas
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  // Draw something
  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, "#ff0000");
  gradient.addColorStop(1, "#0000ff");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  // Create texture and copy
  const texture = device.createTexture({
    size: [size, size],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.copyExternalImageToTexture(
    { source: canvas },
    { texture: texture },
    [size, size],
  );

  return texture;
}
```

Or use `ImageData` directly:

```javascript
const imageData = ctx.getImageData(0, 0, width, height);

device.queue.writeTexture(
  { texture: texture },
  imageData.data, // Uint8ClampedArray
  {
    offset: 0,
    bytesPerRow: width * 4,
    rowsPerImage: height,
  },
  [width, height, 1],
);
```

## Samplers

Samplers configure how textures are read in shaders, controlling filtering, addressing, and comparison operations.

### Creating Samplers

Samplers are lightweight configuration objects:

```javascript
const sampler = device.createSampler({
  label: "Linear Repeat Sampler",
  addressModeU: "repeat",
  addressModeV: "repeat",
  addressModeW: "repeat",
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 1,
});
```

### device.createSampler()

Samplers are created once and can be reused across many textures:

```javascript
// Create samplers for different use cases
const nearestSampler = device.createSampler({
  magFilter: "nearest",
  minFilter: "nearest",
});

const linearSampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
});

const anisotropicSampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 16,
});
```

### GPUSamplerDescriptor Options

Complete sampler configuration:

```javascript
const sampler = device.createSampler({
  label: "Full Configuration Sampler",

  // Address modes (what happens at texture edges)
  addressModeU: "repeat", // Horizontal wrapping
  addressModeV: "clamp-to-edge", // Vertical clamping
  addressModeW: "mirror-repeat", // Depth mirroring

  // Filtering
  magFilter: "linear", // Magnification filter
  minFilter: "linear", // Minification filter
  mipmapFilter: "linear", // Mip level blending

  // Anisotropic filtering
  maxAnisotropy: 16,

  // LOD (Level of Detail) control
  lodMinClamp: 0,
  lodMaxClamp: 10,

  // Comparison (for shadow mapping)
  compare: undefined, // or 'less', 'less-equal', etc.
});
```

## Filtering Modes

Filtering determines how the GPU interpolates texture values between discrete texel positions.

### magFilter, minFilter: 'nearest' or 'linear'

**Magnification Filter** (`magFilter`): Used when texture is drawn larger than its native resolution:

```javascript
// Nearest: Sharp, pixelated look (good for pixel art)
magFilter: "nearest";

// Linear: Smooth, blurred interpolation
magFilter: "linear";
```

**Minification Filter** (`minFilter`): Used when texture is drawn smaller than native resolution:

```javascript
// Nearest: Can cause flickering
minFilter: "nearest";

// Linear: Smoother, reduces aliasing
minFilter: "linear";
```

### mipmapFilter: Mipmap Level Selection

Controls blending between mipmap levels:

```javascript
// Nearest: Abrupt transitions between mip levels
mipmapFilter: "nearest";

// Linear (trilinear filtering): Smooth transitions
mipmapFilter: "linear";
```

Trilinear filtering (linear magFilter + linear minFilter + linear mipmapFilter) provides the highest quality for most use cases.

### Visual Differences

```javascript
// Pixel art / retro aesthetic
const pixelArtSampler = device.createSampler({
  magFilter: "nearest",
  minFilter: "nearest",
  mipmapFilter: "nearest",
});

// Standard smooth texturing
const smoothSampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
});

// High quality with anisotropic filtering
const highQualitySampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 16,
});
```

## Address Modes

Address modes determine what happens when texture coordinates fall outside the [0, 1] range.

### 'clamp-to-edge': Clamp Coordinates

The edge texel is repeated infinitely:

```javascript
addressModeU: "clamp-to-edge";
```

- **Coordinates < 0**: Use texel at coordinate 0
- **Coordinates > 1**: Use texel at coordinate 1
- **Use for**: UI elements, skyboxes, any texture that shouldn't repeat

### 'repeat': Tile Texture

Texture repeats infinitely:

```javascript
addressModeU: "repeat";
```

- **Coordinates wrap**: 1.5 becomes 0.5, -0.3 becomes 0.7
- **Creates tiling**: Perfect for repeating patterns
- **Use for**: Brick walls, terrain textures, any seamless patterns

### 'mirror-repeat': Mirror at Edges

Texture mirrors at each integer boundary:

```javascript
addressModeU: "mirror-repeat";
```

- **Alternates direction**: 0→1→0→1→0...
- **Reduces visible seams**: Even if texture isn't perfectly tileable
- **Use for**: Improving tiling of non-seamless textures

Complete example:

```javascript
const tiledSampler = device.createSampler({
  addressModeU: "repeat",
  addressModeV: "repeat",
  magFilter: "linear",
  minFilter: "linear",
});

const clampedSampler = device.createSampler({
  addressModeU: "clamp-to-edge",
  addressModeV: "clamp-to-edge",
  magFilter: "linear",
  minFilter: "linear",
});
```

## Anisotropic Filtering

Anisotropic filtering dramatically improves quality for surfaces viewed at oblique angles.

### maxAnisotropy Setting

```javascript
const sampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 16, // 1, 2, 4, 8, or 16
});
```

**Values**:

- `1`: No anisotropic filtering (default)
- `2`, `4`, `8`, `16`: Increasing quality and cost

Higher values take more samples along the direction of anisotropy, producing sharper detail on surfaces receding into the distance.

### Quality vs Performance

```javascript
// Low quality, high performance
const performanceSampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 1,
});

// High quality, moderate performance
const qualitySampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 16,
});
```

For most 3D scenes, 4x or 8x anisotropic filtering provides an excellent balance of quality and performance. Reserve 16x for scenarios where maximum quality is required.

## Mipmapping

Mipmaps are pre-computed, progressively smaller versions of a texture, solving aliasing and performance problems when textures are viewed at small sizes.

### What Mipmaps Are

A mipmap chain consists of the original texture plus successively half-sized versions:

```
Level 0: 512×512 (original)
Level 1: 256×256
Level 2: 128×128
Level 3: 64×64
Level 4: 32×32
Level 5: 16×16
Level 6: 8×8
Level 7: 4×4
Level 8: 2×2
Level 9: 1×1
```

The GPU automatically selects the appropriate level based on how much the texture is minified, preventing aliasing and improving cache efficiency.

### Automatic Mipmap Generation

While WebGPU doesn't provide built-in mipmap generation, you can implement it using compute shaders or render passes:

```javascript
function generateMipmaps(device, texture, width, height, mipLevelCount) {
  const encoder = device.createCommandEncoder();

  for (let mipLevel = 1; mipLevel < mipLevelCount; mipLevel++) {
    const srcView = texture.createView({
      baseMipLevel: mipLevel - 1,
      mipLevelCount: 1,
    });

    const dstView = texture.createView({
      baseMipLevel: mipLevel,
      mipLevelCount: 1,
    });

    // Use a render pass or compute shader to downsample
    // srcView into dstView (implementation omitted for brevity)
  }

  device.queue.submit([encoder.finish()]);
}
```

### Manual Mipmap Data

You can also provide pre-computed mipmap data:

```javascript
const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

const texture = device.createTexture({
  size: [width, height],
  format: "rgba8unorm",
  mipLevelCount: mipLevelCount,
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// Write each mip level
for (let mipLevel = 0; mipLevel < mipLevelCount; mipLevel++) {
  const mipWidth = Math.max(1, width >> mipLevel);
  const mipHeight = Math.max(1, height >> mipLevel);
  const mipData = generateMipData(mipLevel); // Your mip generation

  device.queue.writeTexture(
    { texture: texture, mipLevel: mipLevel },
    mipData,
    { bytesPerRow: mipWidth * 4 },
    [mipWidth, mipHeight],
  );
}
```

## Sampling in Shaders

WGSL provides several built-in functions for texture sampling.

### textureSample(texture, sampler, coords)

The primary sampling function, performs filtered texture lookup:

```wgsl
@group(0) @binding(0) var textureSampler: sampler;
@group(0) @binding(1) var colorTexture: texture_2d<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  let color = textureSample(colorTexture, textureSampler, texCoord);
  return color;
}
```

The sampler configuration determines filtering and address modes.

### textureLoad(texture, coords, mipLevel)

Loads a single texel without filtering:

```wgsl
@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  let dimensions = textureDimensions(inputTexture);
  let pixelCoord = vec2<i32>(texCoord * vec2f(dimensions));
  let color = textureLoad(inputTexture, pixelCoord, 0);
  return color;
}
```

`textureLoad` is useful when you need exact texel values without interpolation.

### textureDimensions()

Query texture size at a specific mip level:

```wgsl
@group(0) @binding(0) var myTexture: texture_2d<f32>;

fn getTextureDimensions() -> vec2<u32> {
  let dims = textureDimensions(myTexture, 0); // Mip level 0
  return dims;
}
```

Complete shader example:

```wgsl
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) texCoord: vec2f,
};

@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f
) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(position, 1.0);
  output.texCoord = texCoord;
  return output;
}

@group(0) @binding(0) var textureSampler: sampler;
@group(0) @binding(1) var diffuseTexture: texture_2d<f32>;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let textureColor = textureSample(diffuseTexture, textureSampler, input.texCoord);
  return textureColor;
}
```

## TypeGPU Texture Support

TypeGPU provides type-safe abstractions for working with textures and samplers.

### Texture Slots and Binding

TypeGPU simplifies texture binding with named slots:

```typescript
import tgpu from "typegpu";

// Define texture slot
const diffuseTextureSlot = tgpu.textureSlot({
  format: "rgba8unorm",
  dimension: "2d",
});

// Define sampler slot
const samplerSlot = tgpu.samplerSlot();

// Create shader using slots
const fragmentShader = tgpu.fragmentFn((texCoord: vec2f) => {
  const color = diffuseTextureSlot.sample(samplerSlot, texCoord);
  return { color };
});
```

### Using Textures in TGSL

Complete TypeGPU example:

```typescript
import tgpu from "typegpu";
import { vec2f, vec4f } from "typegpu/data";

// Initialize TypeGPU
const root = await tgpu.init();

// Load texture (using standard WebGPU)
const response = await fetch("texture.jpg");
const blob = await response.blob();
const imageBitmap = await createImageBitmap(blob);

const texture = root.device.createTexture({
  size: [imageBitmap.width, imageBitmap.height],
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

root.device.queue.copyExternalImageToTexture(
  { source: imageBitmap },
  { texture },
  [imageBitmap.width, imageBitmap.height],
);

// Create sampler
const sampler = root.device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
});

// Create typed texture resource
const diffuseTexture = tgpu.texture(texture);
const textureSampler = tgpu.sampler(sampler);

// Use in pipeline
const pipeline = root
  .withVertex(vertexShader, attributes)
  .withFragment(fragmentShader, targets)
  .createPipeline();

// Bind and render
pipeline
  .with(diffuseTexture, textureSampler)
  .withColorAttachment({
    /* ... */
  })
  .draw(vertexCount);
```

## Texture Coordinates

Understanding texture coordinate conventions is essential for correct texture mapping.

### UV Space (0 to 1)

Texture coordinates typically use UV naming and range from 0 to 1:

- **(0, 0)**: Top-left corner (in WebGPU)
- **(1, 1)**: Bottom-right corner
- **(0.5, 0.5)**: Center of texture

```javascript
// Vertex data with texture coordinates
const vertices = new Float32Array([
  // Position (x,y,z)  // TexCoord (u,v)
  -1.0,
  -1.0,
  0.0,
  0.0,
  1.0, // Bottom-left
  1.0,
  -1.0,
  0.0,
  1.0,
  1.0, // Bottom-right
  1.0,
  1.0,
  0.0,
  1.0,
  0.0, // Top-right
  -1.0,
  1.0,
  0.0,
  0.0,
  0.0, // Top-left
]);
```

### Origin Conventions

WebGPU uses a top-left origin for textures:

```
(0,0) ────────────> U
  │
  │
  │
  v
  V
```

This differs from some graphics APIs that use bottom-left origins. When loading images, you may need to flip them:

```javascript
device.queue.copyExternalImageToTexture(
  {
    source: imageBitmap,
    flipY: true, // Flip if needed
  },
  { texture: texture },
  [width, height],
);
```

## Best Practices

### Format Selection

Choose formats based on content and performance requirements:

```javascript
// Standard color photos
format: "rgba8unorm";

// HDR content
format: "rgba16float";

// Normal maps (can use 2-channel)
format: "rg8unorm"; // Reconstruct Z in shader

// Single channel data
format: "r8unorm";

// Depth buffers
format: "depth24plus";
```

### Texture Atlases

Combine multiple small textures into larger atlases to reduce bind group changes:

```javascript
// Instead of binding 100 individual textures,
// pack them into a single 2048×2048 atlas
const atlas = device.createTexture({
  size: [2048, 2048],
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// Adjust UVs to reference specific regions
const spriteUVs = calculateAtlasUVs(spriteIndex, atlasLayout);
```

### Mipmap Usage

Always use mipmaps for textures that will be viewed at varying distances:

```javascript
// Calculate mip levels
const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

const texture = device.createTexture({
  size: [width, height],
  format: "rgba8unorm",
  mipLevelCount: mipLevelCount,
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// Use linear mipmap filtering
const sampler = device.createSampler({
  minFilter: "linear",
  mipmapFilter: "linear",
});
```

## Common Pitfalls

### UV Coordinate Ranges

Problem: UVs outside [0, 1] produce unexpected results with `clamp-to-edge`:

```javascript
// Bad: UVs go to 2.0 but sampler clamps
const sampler = device.createSampler({
  addressModeU: "clamp-to-edge", // This clamps!
});

// Good: Use repeat for tiling
const sampler = device.createSampler({
  addressModeU: "repeat", // Allows UVs > 1
});
```

### Missing Mipmaps

Problem: Texture created without mipmaps causes aliasing:

```javascript
// Bad: No mipmaps
const texture = device.createTexture({
  size: [1024, 1024],
  format: "rgba8unorm",
  mipLevelCount: 1, // Only base level!
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// Good: Full mipmap chain
const mipLevels = Math.floor(Math.log2(1024)) + 1;
const texture = device.createTexture({
  size: [1024, 1024],
  format: "rgba8unorm",
  mipLevelCount: mipLevels, // 11 levels
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

### Format Mismatches

Problem: Using incompatible formats for usage:

```javascript
// Bad: r32float cannot be used as RENDER_ATTACHMENT
const texture = device.createTexture({
  format: "r32float",
  usage: GPUTextureUsage.RENDER_ATTACHMENT, // ERROR!
});

// Good: Use renderable format
const texture = device.createTexture({
  format: "rgba8unorm",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

### Sampler/Texture Mismatch

Problem: Using wrong sampler type for texture:

```wgsl
// Bad: Trying to sample a depth texture as color
@group(0) @binding(0) var depthSampler: sampler;
@group(0) @binding(1) var depthTex: texture_depth_2d;

// This won't work - depth textures need comparison samplers
let depth = textureSample(depthTex, depthSampler, uv); // ERROR

// Good: Use comparison sampler for depth
@group(0) @binding(0) var depthSampler: sampler_comparison;
@group(0) @binding(1) var depthTex: texture_depth_2d;

let depth = textureSampleCompare(depthTex, depthSampler, uv, compareValue);
```

### Not Destroying Textures

Problem: Large textures consume significant GPU memory:

```javascript
// Bad: Creating textures in a loop without cleanup
for (let i = 0; i < 100; i++) {
  const texture = device.createTexture({
    size: [2048, 2048],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  // Use texture briefly...
  // No destroy() call - GPU memory leak!
}

// Good: Explicit cleanup
const texture = device.createTexture({
  /* ... */
});
// Use texture...
texture.destroy(); // Free GPU memory immediately
```

---

Textures and samplers are foundational to GPU programming, enabling efficient storage and access to image data with hardware-accelerated filtering and addressing. By understanding texture types, formats, sampling modes, and best practices, you can create high-performance, visually appealing graphics applications with WebGPU and TypeGPU. Always consider memory usage, choose appropriate formats and filtering modes for your use case, and leverage mipmaps to ensure optimal quality and performance across varying viewing distances.
