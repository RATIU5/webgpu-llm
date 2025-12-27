---
title: WebGPU vs WebGL Migration Guide
sidebar:
  order: 20
---

## Overview

WebGPU represents a fundamental shift from WebGL's OpenGL-based design toward a modern, low-level graphics API inspired by Vulkan, Metal, and Direct3D 12. It offers explicit control over GPU resources, better performance through reduced CPU overhead, and compute shader support.

:::note[Key Differences]
| Aspect | WebGL | WebGPU |
|--------|-------|--------|
| State Model | Global state machine | Stateless pipeline objects |
| Resource Management | Implicit, driver-managed | Explicit, application-managed |
| Command Model | Immediate mode | Command buffers |
| Shading Language | GLSL ES | WGSL |
| Compute Shaders | Not supported | Full support |
| Depth Range (NDC) | -1 to 1 | 0 to 1 |
| Framebuffer Origin | Bottom-left | Top-left |
:::

## Architectural Differences

### State Management

**WebGL** uses a global state machine where settings persist until changed:

```javascript title="WebGL global state"
gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
gl.enableVertexAttribArray(0);
// State remains for all subsequent draws
```

**WebGPU** uses immutable pipeline objects created upfront:

```javascript title="WebGPU stateless pipeline" {1-10}
const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: shaderModule, entryPoint: "vertexMain", buffers: [...] },
  fragment: { module: shaderModule, entryPoint: "fragmentMain", targets: [...] },
  primitive: { topology: "triangle-list" },
});

// Switch pipelines explicitly
pass.setPipeline(pipeline);
```

### Command Model

**WebGL** executes commands immediately:

```javascript
gl.drawArrays(gl.TRIANGLES, 0, 3); // Executes immediately
```

**WebGPU** records commands into buffers, then submits:

```javascript title="WebGPU command buffer pattern" {1,8}
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass(descriptor);

pass.setPipeline(pipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.draw(3);
pass.end();

device.queue.submit([encoder.finish()]); // Actual execution
```

:::tip[Performance Benefit]
WebGPU validates commands once during encoding, not per draw call. This enables 50-70% reduction in CPU overhead for draw-call-heavy applications.
:::

## Shading Languages

### GLSL to WGSL

```glsl title="WebGL GLSL"
attribute vec3 position;
uniform mat4 modelViewProjection;
varying vec2 vTexCoord;

void main() {
    gl_Position = modelViewProjection * vec4(position, 1.0);
}
```

```wgsl title="WebGPU WGSL"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
}

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;

@vertex
fn vertexMain(@location(0) position: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = mvp * vec4<f32>(position, 1.0);
    return output;
}
```

### Key Translation Points

| GLSL | WGSL |
|------|------|
| `attribute` | `@location(n)` input |
| `varying` | `@location(n)` output |
| `uniform sampler2D` | `texture_2d` + `sampler` |
| `texture2D()` | `textureSample()` |
| `gl_Position` | `@builtin(position)` |
| `gl_FragColor` | Return with `@location(0)` |
| `vec4` | `vec4<f32>` |
| `main()` | Named function with `@vertex`/`@fragment` |

## Coordinate System Changes

### Depth Range

:::danger[Critical Migration Issue]
WebGPU uses 0 to 1 depth range (not -1 to 1). Using WebGL projection matrices causes incorrect clipping.
:::

```javascript title="WebGPU perspective matrix (0-1 depth)"
function perspectiveZO(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const rangeInv = 1.0 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * rangeInv, -1,
    0, 0, near * far * rangeInv, 0,
  ];
}
```

:::tip[Use wgpu-matrix]
Libraries like `wgpu-matrix` provide WebGPU-compatible matrix functions out of the box.
:::

### Framebuffer Origin

| Aspect | WebGL | WebGPU |
|--------|-------|--------|
| Origin | Bottom-left | Top-left |
| Y Direction | Up | Down |

**Texture coordinate fix:**

```wgsl title="Flip Y in shader"
let flippedUV = vec2<f32>(texCoord.x, 1.0 - texCoord.y);
let color = textureSample(tex, samp, flippedUV);
```

## Compute Shaders

WebGPU's major feature addition over WebGL:

```wgsl title="WebGPU compute shader"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&input)) {
        output[id.x] = input[id.x] * 2.0;
    }
}
```

:::note[GPGPU Capabilities]
Compute shaders enable:
- Machine learning inference
- Physics simulations
- Image/video processing
- Data-parallel algorithms
:::

## Common Migration Issues

<details>
<summary>**Objects Not Visible**</summary>

**Cause:** Wrong projection matrix depth range.

**Fix:** Use `perspectiveZO` or `wgpu-matrix`:

```javascript
import { mat4 } from "wgpu-matrix";
const proj = mat4.perspective(fov, aspect, near, far);
```

</details>

<details>
<summary>**Textures Upside Down**</summary>

**Cause:** Different Y-axis origins.

**Fix:** Flip UV in shader or flip image data on load:

```javascript
createImageBitmap(blob, { imageOrientation: "flipY" });
```

</details>

<details>
<summary>**Depth Test Failures**</summary>

**Cause:** Missing explicit depth buffer configuration.

**Fix:** Create and configure depth texture:

```javascript title="Explicit depth buffer" {1-5,12-16}
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const renderPass = {
  colorAttachments: [{ /* ... */ }],
  depthStencilAttachment: {
    view: depthTexture.createView(),
    depthLoadOp: "clear",
    depthStoreOp: "store",
    depthClearValue: 1.0,
  },
};
```

</details>

<details>
<summary>**Binding Errors**</summary>

**Cause:** Missing explicit `@group` and `@binding` declarations.

**Fix:** Ensure WGSL bindings match JavaScript bind group:

```wgsl
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
```

</details>

## Side-by-Side Comparison

### WebGL Triangle

```javascript title="WebGL triangle"
const gl = canvas.getContext("webgl");

// Compile shaders
const vs = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vs, vertexSource);
gl.compileShader(vs);

const fs = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fs, fragmentSource);
gl.compileShader(fs);

// Link program
const program = gl.createProgram();
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);
gl.useProgram(program);

// Create buffer
const buffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

// Configure attributes
const loc = gl.getAttribLocation(program, "position");
gl.enableVertexAttribArray(loc);
gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

// Draw
gl.drawArrays(gl.TRIANGLES, 0, 3);
```

### WebGPU Triangle

```javascript title="WebGPU triangle" {6-20,28-29}
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu");
context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat() });

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: device.createShaderModule({ code: shaderCode }),
    entryPoint: "vertexMain",
    buffers: [{
      arrayStride: 8,
      attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
    }],
  },
  fragment: {
    module: device.createShaderModule({ code: shaderCode }),
    entryPoint: "fragmentMain",
    targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
  },
});

const vertexBuffer = device.createBuffer({
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    loadOp: "clear",
    storeOp: "store",
  }],
});
pass.setPipeline(pipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.draw(3);
pass.end();

device.queue.submit([encoder.finish()]);
```

## When to Migrate

### Good Candidates

| Scenario | Reason |
|----------|--------|
| Compute-heavy applications | GPGPU not possible in WebGL |
| Performance-critical projects | Reduced CPU overhead |
| New projects | Better architecture, future-proof |
| Complex rendering | Explicit control, fewer state bugs |

### Stay with WebGL

| Scenario | Reason |
|----------|--------|
| Maximum browser compatibility | WebGL works on older devices |
| Simple 2D graphics | WebGL is simpler for basic use |
| Stable existing codebases | Migration cost may outweigh benefits |
| Limited development resources | WebGPU has steeper learning curve |

## Migration Strategies

### Gradual Migration

```javascript title="Feature detection with fallback"
if (navigator.gpu) {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter) {
    return initWebGPURenderer(await adapter.requestDevice());
  }
}
return initWebGLRenderer();
```

### Feature-by-Feature

1. Start with compute shaders (new functionality)
2. Move particle systems (performance benefit)
3. Convert main rendering pipeline (largest impact)
4. Migrate post-processing effects
5. Update UI rendering last

:::caution[Alignment Requirements]
WebGPU has strict buffer alignment (typically 256 bytes for uniforms). Calculate sizes carefully:

```javascript
const alignedSize = Math.ceil(dataSize / 256) * 256;
```
:::

## Resources

:::note[Official Documentation]
- [Chrome: From WebGL to WebGPU](https://developer.chrome.com/docs/web-platform/webgpu/from-webgl-to-webgpu)
- [WebGPU Fundamentals: From WebGL](https://webgpufundamentals.org/webgpu/lessons/webgpu-from-webgl.html)
- [MDN WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
:::
