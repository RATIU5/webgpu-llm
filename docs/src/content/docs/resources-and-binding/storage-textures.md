---
title: Storage Textures
sidebar:
  order: 30
---

## Overview

Storage textures allow shaders to write directly to texture memory using `textureStore()`. Unlike render attachments that write through the rasterization pipeline, storage textures enable arbitrary writes from compute and fragment shaders.

:::note[Storage vs Sampled Textures]
| Aspect | Sampled Texture | Storage Texture |
|--------|-----------------|-----------------|
| Read method | `textureSample()` with filtering | `textureLoad()` exact texel |
| Write method | Render attachment only | `textureStore()` anywhere |
| Filtering | Bilinear, trilinear, aniso | None |
| Mipmaps | Supported | Single level only |
| Use case | Traditional texturing | Compute output, procedural |
:::

## Creating Storage Textures

```javascript title="Create storage texture" {4-5}
const storageTexture = device.createTexture({
  size: [512, 512],
  format: "rgba8unorm",
  usage: GPUTextureUsage.STORAGE_BINDING |
         GPUTextureUsage.TEXTURE_BINDING,
});
```

:::caution[Required Usage Flags]
- `STORAGE_BINDING`: Required for storage texture binding
- `TEXTURE_BINDING`: Add if you also need to sample the result
- `COPY_SRC`: Add if you need to read back to CPU
:::

## Supported Formats

### Write-Only Formats

All storage texture formats support write-only access:

| Format | Components | Bits | WGSL Type |
|--------|------------|------|-----------|
| `rgba8unorm` | 4 | 32 | `vec4<f32>` |
| `rgba8snorm` | 4 | 32 | `vec4<f32>` |
| `rgba8uint` | 4 | 32 | `vec4<u32>` |
| `rgba8sint` | 4 | 32 | `vec4<i32>` |
| `rgba16uint` | 4 | 64 | `vec4<u32>` |
| `rgba16sint` | 4 | 64 | `vec4<i32>` |
| `rgba16float` | 4 | 64 | `vec4<f32>` |
| `r32uint` | 1 | 32 | `u32` |
| `r32sint` | 1 | 32 | `i32` |
| `r32float` | 1 | 32 | `f32` |
| `rg32uint` | 2 | 64 | `vec2<u32>` |
| `rg32sint` | 2 | 64 | `vec2<i32>` |
| `rg32float` | 2 | 64 | `vec2<f32>` |
| `rgba32uint` | 4 | 128 | `vec4<u32>` |
| `rgba32sint` | 4 | 128 | `vec4<i32>` |
| `rgba32float` | 4 | 128 | `vec4<f32>` |

### Read-Write Formats

Only R32 formats support `read_write` access:

| Format | Access Modes |
|--------|--------------|
| `r32float` | `read`, `write`, `read_write` |
| `r32uint` | `read`, `write`, `read_write` |
| `r32sint` | `read`, `write`, `read_write` |

:::danger[Read-Write Limitations]
Read-write storage textures require:
```wgsl
requires readonly_and_readwrite_storage_textures;
```
Other formats cannot be read and written in the same shader invocation.
:::

## WGSL Declaration

### Write-Only

```wgsl title="Write-only storage texture"
@group(0) @binding(0)
var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let color = vec4f(
    f32(id.x) / 512.0,
    f32(id.y) / 512.0,
    0.5,
    1.0
  );
  textureStore(outputTex, id.xy, color);
}
```

### Read-Write

```wgsl title="Read-write storage texture (R32 only)"
requires readonly_and_readwrite_storage_textures;

@group(0) @binding(0)
var tex: texture_storage_2d<r32float, read_write>;

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) id: vec3u) {
  let current = textureLoad(tex, id.xy);
  let left = textureLoad(tex, id.xy - vec2u(1, 0));
  let right = textureLoad(tex, id.xy + vec2u(1, 0));

  let avg = (current + left + right) / 3.0;

  textureBarrier();  // Sync before writing
  textureStore(tex, id.xy, vec4f(avg, 0, 0, 1));
}
```

## Binding Configuration

```javascript title="Bind group layout for storage texture" {5-9}
const bindGroupLayout = device.createBindGroupLayout({
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.COMPUTE,
    storageTexture: {
      access: "write-only",  // or "read-only", "read-write"
      format: "rgba8unorm",
      viewDimension: "2d",
    },
  }],
});

const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{
    binding: 0,
    resource: storageTexture.createView(),
  }],
});
```

## Use Cases

### Procedural Generation

```wgsl title="Procedural noise texture"
@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

@compute @workgroup_size(8, 8)
fn generateNoise(@builtin(global_invocation_id) id: vec3u) {
  let uv = vec2f(id.xy) / 512.0;
  let n = hash(uv * 10.0);
  textureStore(output, id.xy, vec4f(n, n, n, 1.0));
}
```

### Image Processing

```wgsl title="Grayscale conversion"
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn grayscale(@builtin(global_invocation_id) id: vec3u) {
  let color = textureLoad(inputTex, id.xy, 0);
  let gray = dot(color.rgb, vec3f(0.299, 0.587, 0.114));
  textureStore(outputTex, id.xy, vec4f(gray, gray, gray, 1.0));
}
```

### Compute Shader Output

```wgsl title="Physics simulation visualization"
@group(0) @binding(0) var<storage, read> particles: array<vec4f>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(64)
fn renderParticles(@builtin(global_invocation_id) id: vec3u) {
  let p = particles[id.x];
  let pixel = vec2u(u32(p.x), u32(p.y));
  textureStore(output, pixel, vec4f(1.0, 0.5, 0.0, 1.0));
}
```

## Synchronization

### Within Workgroup

```wgsl title="Workgroup synchronization"
textureBarrier();  // Ensure all writes visible within workgroup
```

### Between Dispatches

Separate compute passes automatically synchronize:

```javascript title="Multi-pass processing"
// Pass 1: Generate
computePass1.setPipeline(generatePipeline);
computePass1.dispatchWorkgroups(64, 64);
computePass1.end();

// Pass 2: Process (automatically waits for Pass 1)
computePass2.setPipeline(processPipeline);
computePass2.dispatchWorkgroups(64, 64);
computePass2.end();
```

## TypeGPU Storage Textures

```typescript title="TypeGPU storage texture binding"
import tgpu from "typegpu";
import * as d from "typegpu/data";

const outputTexture = root.createTexture({
  size: [512, 512],
  format: "rgba8unorm",
}).$usage("storage", "sampled");

const processImage = tgpu
  .computeFn({ workgroupSize: [8, 8] })
  .does`(@builtin(global_invocation_id) id: vec3u) {
    let color = vec4f(f32(id.x) / 512.0, f32(id.y) / 512.0, 0.5, 1.0);
    textureStore(${outputTexture}, id.xy, color);
  }`;
```

## Limitations

:::caution[Storage Texture Constraints]
- **No multisampling**: Storage textures cannot have `sampleCount > 1`
- **No mipmaps**: Only mip level 0 accessible
- **Format in shader**: Must specify exact format in WGSL declaration
- **No filtering**: Use `textureLoad()` for exact texel access only
- **Limited read-write**: Only R32 formats support simultaneous read/write
:::
