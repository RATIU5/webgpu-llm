---
title: Textures and Samplers
sidebar:
  order: 20
---

## Overview

Textures are multi-dimensional arrays of data stored in GPU memory, optimized for spatial access patterns. Unlike buffers that store linear arrays, textures provide built-in interpolation, filtering, and addressing modes through samplers.

:::note[Texture vs Buffer]
- **Textures**: Optimized for 2D locality, hardware filtering, compressed formats, and automatic mipmap selection
- **Buffers**: Better for random write access, atomic operations, and data without spatial locality
:::

Textures and samplers work together:
- **Textures** hold pixel data in GPU memory
- **Samplers** define how data is read—filtering quality, coordinate wrapping, mipmap selection

## Texture Types

### 1D, 2D, 3D Textures

<details>
<summary>**1D Textures**</summary>

Single-row arrays for lookup tables and gradients:

```javascript title="1D texture for color ramps"
const gradientTexture = device.createTexture({
  size: [256, 1, 1],
  dimension: "1d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

</details>

<details>
<summary>**2D Textures**</summary>

Standard images—the most common texture type:

```javascript title="2D diffuse texture"
const diffuseTexture = device.createTexture({
  size: [1024, 1024, 1],
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

Used for photographs, UI elements, sprite sheets, and normal maps.

</details>

<details>
<summary>**3D Textures**</summary>

Volumetric data with width, height, and depth:

```javascript title="3D volume texture"
const volumeTexture = device.createTexture({
  size: [128, 128, 128],
  dimension: "3d",
  format: "r32float",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

Unlike 2D arrays, 3D textures support interpolation between depth slices.

</details>

### Cube Maps

Six 2D textures arranged as cube faces for environment mapping:

```javascript title="Cube map for skybox"
const skyboxTexture = device.createTexture({
  size: [512, 512, 6], // Width, height, 6 faces
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

const cubeView = skyboxTexture.createView({
  dimension: "cube",
});
```

Sampled using 3D direction vectors rather than 2D coordinates.

### Texture Arrays

Multiple independent 2D images in a single texture object:

```javascript title="Texture array for sprites"
const textureArray = device.createTexture({
  size: [256, 256, 10], // 10 layers of 256×256 images
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

const arrayView = textureArray.createView({
  dimension: "2d-array",
});
```

Ideal for sprite animations and terrain splatmaps.

### Multisampled Textures

Multiple samples per pixel for antialiasing:

```javascript title="4× MSAA texture"
const msaaTexture = device.createTexture({
  size: [800, 600, 1],
  dimension: "2d",
  format: "rgba8unorm",
  sampleCount: 4,
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

:::caution
Multisampled textures can only be render attachments. Resolve to regular textures before sampling.
:::

## Creating Textures

### GPUTextureDescriptor

```javascript title="Complete texture creation" {3,6-7}
const texture = device.createTexture({
  label: "Main Color Texture",
  size: { width: 512, height: 512, depthOrArrayLayers: 1 },
  dimension: "2d",
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING |
         GPUTextureUsage.COPY_DST |
         GPUTextureUsage.RENDER_ATTACHMENT,
  mipLevelCount: 1,
  sampleCount: 1,
});
```

| Property | Description |
|----------|-------------|
| `label` | Debug string for error messages |
| `size` | Dimensions as object or `[width, height?, depth?]` |
| `dimension` | `'1d'`, `'2d'`, or `'3d'` |
| `format` | Pixel format (see formats section) |
| `usage` | Bitwise combination of usage flags |
| `mipLevelCount` | Number of mipmap levels |
| `sampleCount` | 1 (default) or 4 (MSAA) |

### Usage Flags

| Flag | Purpose |
|------|---------|
| `TEXTURE_BINDING` | Sample in shaders |
| `STORAGE_BINDING` | Read-write storage texture |
| `RENDER_ATTACHMENT` | Render target |
| `COPY_SRC` / `COPY_DST` | Copy operations |

:::tip[Conservative Usage]
Only specify flags you need—some combinations prevent driver optimizations.
:::

### Mipmap Levels

For a full mipmap chain:

```javascript title="Calculate mip levels"
const maxDimension = Math.max(width, height);
const mipLevelCount = Math.floor(Math.log2(maxDimension)) + 1;

// 512×512 → 10 levels: 512→256→128→64→32→16→8→4→2→1
```

## Common Texture Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `rgba8unorm` | 8-bit RGBA, normalized | Standard color |
| `bgra8unorm` | 8-bit BGRA, normalized | Canvas render targets |
| `depth24plus` | ≥24-bit depth | Depth buffers |
| `depth32float` | 32-bit float depth | Shader-accessible depth |
| `r32float` | Single-channel float | Height maps, distance fields |
| `rgba16float` | 16-bit float RGBA | HDR rendering |
| `r8unorm` | Single-channel 8-bit | Masks, grayscale |

```javascript title="Get preferred canvas format"
const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
// 'bgra8unorm' on macOS/Metal, 'rgba8unorm' on Windows/D3D12
```

:::caution[Format Compatibility]
Not all formats work with all usages. `r32float` cannot be a `RENDER_ATTACHMENT` without checking device features.
:::

## Loading Images

### From URL

```javascript title="Load texture from image file" {3-4,17-21}
async function loadTexture(device, url) {
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);

  const texture = device.createTexture({
    size: [imageBitmap.width, imageBitmap.height, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING |
           GPUTextureUsage.COPY_DST |
           GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: texture },
    [imageBitmap.width, imageBitmap.height]
  );

  return texture;
}
```

### From Canvas

```javascript title="Procedural texture from canvas"
function createProceduralTexture(device, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, "#ff0000");
  gradient.addColorStop(1, "#0000ff");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = device.createTexture({
    size: [size, size],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.copyExternalImageToTexture(
    { source: canvas },
    { texture: texture },
    [size, size]
  );

  return texture;
}
```

### copyExternalImageToTexture Sources

- `ImageBitmap`
- `HTMLCanvasElement`
- `OffscreenCanvas`
- `HTMLVideoElement`

```javascript title="Flip image on upload"
device.queue.copyExternalImageToTexture(
  { source: imageBitmap, flipY: true },
  { texture: texture },
  [width, height]
);
```

## Samplers

### Creating Samplers

```javascript title="Complete sampler configuration"
const sampler = device.createSampler({
  label: "Linear Repeat Sampler",
  addressModeU: "repeat",
  addressModeV: "repeat",
  addressModeW: "repeat",
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 1,
  lodMinClamp: 0,
  lodMaxClamp: 10,
});
```

:::tip
Samplers are lightweight—create them once and reuse across many textures.
:::

### Filtering Modes

| Property | `'nearest'` | `'linear'` |
|----------|-------------|------------|
| `magFilter` | Pixelated (good for pixel art) | Smooth interpolation |
| `minFilter` | May cause flickering | Reduces aliasing |
| `mipmapFilter` | Abrupt mip transitions | Smooth mip blending |

```javascript title="Common sampler presets"
// Pixel art / retro
const pixelSampler = device.createSampler({
  magFilter: "nearest",
  minFilter: "nearest",
});

// Smooth texturing
const linearSampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
});

// High quality with anisotropic filtering
const qualitySampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 16,
});
```

### Address Modes

| Mode | Behavior |
|------|----------|
| `'clamp-to-edge'` | Edge texel repeated infinitely |
| `'repeat'` | Texture tiles (1.5 → 0.5) |
| `'mirror-repeat'` | Alternates direction at edges |

```javascript title="Address mode examples"
// Tiling textures (brick walls, terrain)
const tiledSampler = device.createSampler({
  addressModeU: "repeat",
  addressModeV: "repeat",
  magFilter: "linear",
  minFilter: "linear",
});

// UI elements, skyboxes (no repeat)
const clampedSampler = device.createSampler({
  addressModeU: "clamp-to-edge",
  addressModeV: "clamp-to-edge",
  magFilter: "linear",
  minFilter: "linear",
});
```

### Anisotropic Filtering

Improves quality for surfaces viewed at oblique angles:

```javascript title="Enable anisotropic filtering"
const sampler = device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  maxAnisotropy: 16, // 1, 2, 4, 8, or 16
});
```

:::tip[Performance Balance]
4× or 8× anisotropic filtering provides excellent quality/performance balance. Reserve 16× for maximum quality scenarios.
:::

## Mipmapping

Mipmaps are pre-computed, progressively smaller texture versions:

```
Level 0: 512×512 (original)
Level 1: 256×256
Level 2: 128×128
...
Level 9: 1×1
```

:::note[Why Mipmaps Matter]
- Prevent aliasing at small sizes
- Improve cache efficiency
- GPU automatically selects appropriate level
:::

### Manual Mipmap Data

```javascript title="Create texture with mipmaps"
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
  const mipData = generateMipData(mipLevel);

  device.queue.writeTexture(
    { texture: texture, mipLevel: mipLevel },
    mipData,
    { bytesPerRow: mipWidth * 4 },
    [mipWidth, mipHeight]
  );
}
```

:::caution
WebGPU doesn't provide built-in mipmap generation. Use compute shaders or render passes to downsample.
:::

### Generating Mipmaps

Unlike WebGL's `generateMipmap()`, WebGPU requires manual mipmap generation.

<details>
<summary>**Render Pass Approach**</summary>

Draw textured quads from each level to the next:

```javascript title="Mipmap generation with render passes"
const mipmapPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: blitShaderModule, entryPoint: "vertexMain" },
  fragment: {
    module: blitShaderModule,
    entryPoint: "fragmentMain",
    targets: [{ format: "rgba8unorm" }],
  },
});

for (let level = 1; level < mipLevelCount; level++) {
  const srcView = texture.createView({
    baseMipLevel: level - 1,
    mipLevelCount: 1,
  });
  const dstView = texture.createView({
    baseMipLevel: level,
    mipLevelCount: 1,
  });

  const bindGroup = device.createBindGroup({
    layout: mipmapPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: linearSampler },
      { binding: 1, resource: srcView },
    ],
  });

  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: dstView,
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(mipmapPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(6);
  pass.end();
}
```

</details>

<details>
<summary>**Compute Shader Approach**</summary>

Faster for large textures—process 4 source texels to 1 destination:

```wgsl title="Mipmap compute shader"
@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var dstTexture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn generateMip(@builtin(global_invocation_id) id: vec3u) {
  let dstCoord = id.xy;
  let srcCoord = dstCoord * 2u;

  // Average 4 source texels
  let tl = textureLoad(srcTexture, srcCoord, 0);
  let tr = textureLoad(srcTexture, srcCoord + vec2u(1, 0), 0);
  let bl = textureLoad(srcTexture, srcCoord + vec2u(0, 1), 0);
  let br = textureLoad(srcTexture, srcCoord + vec2u(1, 1), 0);

  let avg = (tl + tr + bl + br) * 0.25;
  textureStore(dstTexture, dstCoord, avg);
}
```

Requires texture with both `TEXTURE_BINDING` and `STORAGE_BINDING` usage.

</details>

## Sampling in Shaders

### textureSample

Filtered texture lookup (uses sampler settings):

```wgsl title="Basic texture sampling"
@group(0) @binding(0) var textureSampler: sampler;
@group(0) @binding(1) var colorTexture: texture_2d<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  let color = textureSample(colorTexture, textureSampler, texCoord);
  return color;
}
```

### textureLoad

Direct texel access without filtering:

```wgsl title="Load exact texel value"
@group(0) @binding(0) var inputTexture: texture_2d<f32>;

@fragment
fn fragmentMain(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  let dimensions = textureDimensions(inputTexture);
  let pixelCoord = vec2<i32>(texCoord * vec2f(dimensions));
  let color = textureLoad(inputTexture, pixelCoord, 0);
  return color;
}
```

### Complete Shader Example

```wgsl title="Full texture sampling shader"
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
  return textureSample(diffuseTexture, textureSampler, input.texCoord);
}
```

## TypeGPU Texture Support

```typescript title="TypeGPU texture binding"
import tgpu from "typegpu";

// Define texture and sampler slots
const diffuseTextureSlot = tgpu.textureSlot({
  format: "rgba8unorm",
  dimension: "2d",
});

const samplerSlot = tgpu.samplerSlot();

// Create shader using slots
const fragmentShader = tgpu.fragmentFn((texCoord: vec2f) => {
  const color = diffuseTextureSlot.sample(samplerSlot, texCoord);
  return { color };
});
```

## Texture Coordinates

### UV Space

Coordinates range from 0 to 1:

```
(0,0) ────────────> U
  │   Top-left origin
  │   (WebGPU convention)
  v
  V           (1,1)
```

:::caution[Origin Conventions]
WebGPU uses top-left origin. Some image formats use bottom-left. Use `flipY: true` when loading if needed.
:::

```javascript title="Vertex data with UVs"
const vertices = new Float32Array([
  // Position (x,y,z)    TexCoord (u,v)
  -1.0, -1.0, 0.0,       0.0, 1.0,  // Bottom-left
   1.0, -1.0, 0.0,       1.0, 1.0,  // Bottom-right
   1.0,  1.0, 0.0,       1.0, 0.0,  // Top-right
  -1.0,  1.0, 0.0,       0.0, 0.0,  // Top-left
]);
```

### Texture Atlases

Combine multiple small textures to reduce bind group changes:

```javascript title="Texture atlas approach"
const atlas = device.createTexture({
  size: [2048, 2048],
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// Calculate UVs for specific sprite region
function getSpriteUVs(spriteIndex, atlasLayout) {
  const col = spriteIndex % atlasLayout.columns;
  const row = Math.floor(spriteIndex / atlasLayout.columns);
  const u = col / atlasLayout.columns;
  const v = row / atlasLayout.rows;
  const spriteWidth = 1 / atlasLayout.columns;
  const spriteHeight = 1 / atlasLayout.rows;
  return { u, v, width: spriteWidth, height: spriteHeight };
}
```

## Depth Textures

### Depth Formats

| Format | Bits | Shader Access | Use Case |
|--------|------|---------------|----------|
| `depth24plus` | ≥24 | No | Standard depth buffer |
| `depth32float` | 32 | Yes | Shadow mapping, effects |
| `depth24plus-stencil8` | 24+8 | No | Depth + stencil |
| `depth32float-stencil8` | 32+8 | Depth only | Depth access + stencil |

:::tip[Format Selection]
Use `depth24plus` for render-only depth buffers. Use `depth32float` when you need to sample depth in shaders.
:::

### Creating Depth Textures

```javascript title="Shader-accessible depth texture"
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth32float",
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});
```

### Shadow Mapping

<details>
<summary>**1. Shadow Map Creation**</summary>

Render scene depth from light's perspective:

```javascript title="Shadow map texture"
const shadowMap = device.createTexture({
  size: [1024, 1024],
  format: "depth32float",
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

// Render pass for shadow map
const shadowPass = encoder.beginRenderPass({
  colorAttachments: [],  // Depth only, no color
  depthStencilAttachment: {
    view: shadowMap.createView(),
    depthLoadOp: "clear",
    depthStoreOp: "store",
    depthClearValue: 1.0,
  },
});
```

</details>

<details>
<summary>**2. Comparison Sampler**</summary>

```javascript title="Create comparison sampler"
const shadowSampler = device.createSampler({
  compare: "less",
  magFilter: "linear",
  minFilter: "linear",
});
```

:::danger[Comparison Samplers]
Depth textures require `sampler_comparison` and `textureSampleCompare()`:

```wgsl
@group(0) @binding(0) var shadowSampler: sampler_comparison;
@group(0) @binding(1) var shadowMap: texture_depth_2d;

// Returns 0.0 (in shadow) or 1.0 (lit)
let shadow = textureSampleCompare(shadowMap, shadowSampler, uv, depth);
```
:::

</details>

<details>
<summary>**3. Shadow Sampling Shader**</summary>

```wgsl title="Shadow mapping fragment shader"
@group(0) @binding(0) var shadowSampler: sampler_comparison;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var<uniform> lightViewProj: mat4x4f;

@fragment
fn main(
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f
) -> @location(0) vec4f {
  // Transform to light space
  let lightSpacePos = lightViewProj * vec4f(worldPos, 1.0);
  let projCoords = lightSpacePos.xyz / lightSpacePos.w;

  // Convert to UV space [0,1]
  let shadowUV = projCoords.xy * 0.5 + 0.5;
  let currentDepth = projCoords.z;

  // PCF soft shadows (3×3 kernel)
  var shadow = 0.0;
  let texelSize = 1.0 / 1024.0;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      let offset = vec2f(f32(x), f32(y)) * texelSize;
      shadow += textureSampleCompare(
        shadowMap, shadowSampler,
        shadowUV + offset, currentDepth - 0.005  // Bias
      );
    }
  }
  shadow /= 9.0;

  let lighting = max(dot(normal, lightDir), 0.0) * shadow;
  return vec4f(baseColor * lighting, 1.0);
}
```

</details>

### Reading Depth Without Comparison

For post-processing effects that need raw depth values:

```wgsl title="Read raw depth values"
@group(0) @binding(0) var depthTex: texture_depth_2d;

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let depth = textureLoad(depthTex, vec2i(fragCoord.xy), 0);
  // depth is in [0, 1] range (0 = near, 1 = far)
  return vec4f(depth, depth, depth, 1.0);
}
```

## Resource Cleanup

:::danger[Memory Management]
Large textures consume significant GPU memory. Destroy when no longer needed:

```javascript
texture.destroy(); // Free GPU memory immediately
```
:::

## Resources

:::note[Official Documentation]
- [WebGPU Texture Specification](https://gpuweb.github.io/gpuweb/#textures)
- [TypeGPU Texture Guide](https://docs.swmansion.com/TypeGPU/fundamentals/textures/)
- [WebGPU Fundamentals: Textures](https://webgpufundamentals.org/webgpu/lessons/webgpu-textures.html)
:::
