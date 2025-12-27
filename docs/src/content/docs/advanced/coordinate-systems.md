---
title: Coordinate Systems and Clip Space
sidebar:
  order: 30
---

## Overview

WebGPU adopts coordinate conventions aligned with modern APIs (Vulkan, Metal, Direct3D 12), differing from WebGL/OpenGL in two critical areas:

:::note[Key Differences from WebGL]
- **Depth range**: 0 to 1 (not -1 to 1)
- **Framebuffer origin**: Top-left (not bottom-left)
:::

## Coordinate Spaces Pipeline

| Stage | Description |
|-------|-------------|
| **Object Space** | Local coordinates relative to object origin |
| **World Space** | Global scene coordinates via model matrix |
| **View Space** | Camera-relative coordinates via view matrix |
| **Clip Space** | 4D homogeneous output from vertex shader |
| **NDC** | After perspective divide (x/w, y/w, z/w) |
| **Screen Space** | Final pixel coordinates |

## WebGPU Coordinate Convention

WebGPU uses a **right-handed coordinate system**:

```
         +Y (up)
          |
          |
          +-------- +X (right)
         /
        +Z (toward viewer)
```

### API Comparison

| API | Depth Range (NDC) | Framebuffer Origin |
|-----|-------------------|-------------------|
| **WebGPU** | 0 to 1 | Top-left |
| **Vulkan/Metal/D3D12** | 0 to 1 | Top-left |
| **WebGL/OpenGL** | -1 to 1 | Bottom-left |

## Clip Space and NDC

### Clip Space Output

The vertex shader outputs 4D homogeneous coordinates:

```wgsl title="Vertex shader clip space output"
@vertex
fn vertexMain(input: Vertex) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = uniforms.modelMatrix * vec4f(input.position, 1.0);
  let viewPos = uniforms.viewMatrix * worldPos;
  output.position = uniforms.projectionMatrix * viewPos;
  return output;
}
```

### NDC Ranges in WebGPU

After the perspective divide (automatic after vertex shader):

| Axis | Range | Description |
|------|-------|-------------|
| X | -1 to +1 | Left to right |
| Y | -1 to +1 | Bottom to top |
| Z | **0 to +1** | Near to far |

:::danger[WebGPU Depth Range]
Z ranges from 0 (near) to 1 (far), not -1 to 1 like WebGL. This requires different projection matrices.
:::

## Screen Space

Origin at **top-left**, Y increases **downward**:

```
(0,0) +---------------------+
      |    Y increases      |
      |    downward ↓       |
      +---------------------+ (width, height)
   X increases rightward →
```

## Projection Matrices

:::danger[Don't Use WebGL Libraries]
Libraries like `gl-matrix` produce -1 to 1 depth range. Use `wgpu-matrix` for WebGPU.
:::

### Using wgpu-matrix

```javascript title="Correct projection for WebGPU"
import { mat4 } from "wgpu-matrix";

// Perspective (0-to-1 depth)
const proj = mat4.perspective(fov, aspect, near, far);

// Orthographic (0-to-1 depth)
const ortho = mat4.ortho(left, right, bottom, top, near, far);
```

### Manual Perspective Matrix (WebGPU)

```javascript title="WebGPU perspective matrix"
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

### Converting WebGL Matrices

If you must convert existing WebGL matrices:

```javascript title="Convert -1..1 depth to 0..1"
function convertWebGLToWebGPU(glMatrix) {
  const gpuMatrix = glMatrix.slice();
  gpuMatrix[8] = (gpuMatrix[8] + gpuMatrix[12]) * 0.5;
  gpuMatrix[9] = (gpuMatrix[9] + gpuMatrix[13]) * 0.5;
  gpuMatrix[10] = (gpuMatrix[10] + gpuMatrix[14]) * 0.5;
  gpuMatrix[11] = (gpuMatrix[11] + gpuMatrix[15]) * 0.5;
  return gpuMatrix;
}
```

## Texture Coordinates

UV coordinates range 0 to 1:

```
(0,0) +----------+ (1,0)
      |          |
      | Texture  |
      |          |
(0,1) +----------+ (1,1)
```

:::caution[Texture Flipping]
Most image formats store data top-to-bottom, which may appear upside down. Solutions:

1. **Flip in shader**: `vec2f(uv.x, 1.0 - uv.y)`
2. **Flip on load**: `createImageBitmap(blob, { imageOrientation: "flipY" })`
:::

## Viewport and Scissor

```javascript title="Viewport configuration"
// Full framebuffer
renderPass.setViewport(0, 0, canvas.width, canvas.height, 0, 1);

// Split-screen (left half)
renderPass.setViewport(0, 0, canvas.width / 2, canvas.height, 0, 1);

// Scissor rectangle (clips fragments)
renderPass.setScissorRect(x, y, width, height);
```

| Function | Behavior |
|----------|----------|
| `setViewport` | Scales and transforms NDC to screen |
| `setScissorRect` | Discards fragments outside rectangle |

## Camera Setup

```javascript title="Complete camera setup"
import { mat4 } from "wgpu-matrix";

const cameraPosition = [0, 10, 20];
const target = [0, 0, 0];
const up = [0, 1, 0];

// View matrix (world → view space)
const viewMatrix = mat4.lookAt(cameraPosition, target, up);

// Projection matrix (view → clip space)
const projectionMatrix = mat4.perspective(
  Math.PI / 4,              // 45° FOV
  canvas.width / canvas.height,
  0.1,                      // Near plane
  100.0                     // Far plane
);

// Combined view-projection
const viewProjectionMatrix = mat4.multiply(projectionMatrix, viewMatrix);
```

## Depth Precision

:::tip[Avoid Z-Fighting]
Keep near/far ratio reasonable:

```javascript
// Poor precision (ratio 10,000,000)
mat4.perspective(fov, aspect, 0.001, 10000);

// Better precision (ratio 1,000)
mat4.perspective(fov, aspect, 0.1, 100);
```
:::

For extreme distances, consider **reverse-Z** (near=1, far=0):

```javascript title="Reverse-Z for better precision"
const reverseZProj = mat4.perspective(fov, aspect, far, near);
// Requires depthCompare: 'greater' instead of 'less'
```

## Migration from WebGL

| Issue | Cause | Fix |
|-------|-------|-----|
| Objects not visible | Wrong projection matrix | Use wgpu-matrix |
| Textures upside down | Different framebuffer origin | Flip UV or texture data |
| Depth test failures | Depth outside [0,1] | Fix projection matrix |
| Y-coordinates inverted | Assuming bottom-left origin | Top-left origin in WebGPU |

## Debugging Coordinates

```wgsl title="Visualize coordinate spaces"
@fragment
fn debugCoordinates(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let depth = fragCoord.z;  // [0, 1] in WebGPU
  let screenNorm = fragCoord.xy / vec2f(800.0, 600.0);
  return vec4f(screenNorm.x, screenNorm.y, depth, 1.0);
}
```

## Resources

:::note[Official Documentation]
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [wgpu-matrix Documentation](https://wgpu-matrix.org/docs/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
:::
