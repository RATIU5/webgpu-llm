---
title: Coordinate Systems and Clip Space
sidebar:
  order: 30
---

Understanding coordinate systems is fundamental to working with WebGPU. Unlike WebGL, which follows OpenGL conventions, WebGPU adopts coordinate system conventions that align with modern graphics APIs like Vulkan, Metal, and Direct3D 12. This document provides a comprehensive guide to WebGPU's coordinate systems, their transformations, and practical implications for graphics programming.

## Overview

In 3D graphics rendering, geometry data passes through several coordinate spaces before appearing on screen. Each transformation serves a specific purpose in the rendering pipeline:

1. **Object/Model Space** → Local coordinates relative to the object
2. **World Space** → Global scene coordinates
3. **View/Camera Space** → Coordinates relative to the camera
4. **Clip Space** → Output of the vertex shader, ready for clipping
5. **Normalized Device Coordinates (NDC)** → After perspective divide
6. **Screen/Viewport Space** → Final pixel coordinates

WebGPU's coordinate conventions differ significantly from WebGL/OpenGL in two critical areas: **depth range** (0 to 1 instead of -1 to 1) and **framebuffer Y-axis direction** (origin at top-left). These differences require careful attention when writing shaders, setting up cameras, and migrating from WebGL.

## WebGPU Coordinate Convention

WebGPU uses a **right-handed coordinate system** for world and view space calculations. In a right-handed system:

- X-axis points right
- Y-axis points up
- Z-axis points toward the viewer (out of the screen)

```
         +Y (up)
          |
          |
          |
          +-------- +X (right)
         /
        /
       +Z (toward viewer)
```

### Comparison to Other APIs

| API              | Depth Range (NDC) | Framebuffer Origin | Y-Axis in NDC           |
| ---------------- | ----------------- | ------------------ | ----------------------- |
| **WebGPU**       | 0 to 1            | Top-left           | -1 (bottom) to +1 (top) |
| **Vulkan**       | 0 to 1            | Top-left           | -1 (bottom) to +1 (top) |
| **Metal**        | 0 to 1            | Top-left           | -1 (bottom) to +1 (top) |
| **D3D12**        | 0 to 1            | Top-left           | -1 (bottom) to +1 (top) |
| **WebGL/OpenGL** | -1 to 1           | Bottom-left        | -1 (bottom) to +1 (top) |

WebGPU's alignment with modern APIs simplifies cross-platform development and takes advantage of hardware optimizations on modern GPUs.

## Coordinate Spaces

### Object/Model Space

Object space (also called model space or local space) represents coordinates relative to an object's own origin. Each 3D model has its own local coordinate system, typically centered at a logical point like the object's center or base.

**Characteristics:**

- Origin (0, 0, 0) is at a meaningful point for the object
- Coordinates are independent of the object's position in the scene
- Allows reusing the same model in different locations

**Example:**

```wgsl
// Vertex positions in object space
struct Vertex {
  @location(0) position: vec3f,  // Object-space coordinates
  @location(1) normal: vec3f,    // Object-space normal vector
}
```

### World Space

World space is the global coordinate system for the entire scene. All objects are positioned, oriented, and scaled within this unified space using a **model matrix** (also called world matrix).

**Transformation:**

```javascript
// Model matrix transforms from object space to world space
const modelMatrix = mat4.create();
mat4.translate(modelMatrix, [x, y, z], modelMatrix); // Position
mat4.rotateY(modelMatrix, angleY, modelMatrix); // Rotation
mat4.scale(modelMatrix, [sx, sy, sz], modelMatrix); // Scale
```

**In the shader:**

```wgsl
@vertex
fn vertexMain(input: Vertex) -> VertexOutput {
  // Transform from object space to world space
  let worldPosition = uniforms.modelMatrix * vec4f(input.position, 1.0);
  // ...
}
```

### View/Camera Space

View space (also called camera space or eye space) represents coordinates relative to the camera. The camera is positioned at the origin (0, 0, 0) and typically looks down the **negative Z-axis** in a right-handed system.

**Characteristics:**

- Camera at origin (0, 0, 0)
- Camera looks down -Z axis
- X-axis points right from camera's perspective
- Y-axis points up from camera's perspective

```
Camera View Space (Right-handed):

    +Y (up)
     |
     |  -Z (view direction)
     | /
     |/
     +-------- +X (right)
```

**Transformation:**
The view matrix transforms from world space to view space. It's typically constructed using a "lookAt" function:

```javascript
// Create view matrix using camera position, target, and up vector
const viewMatrix = mat4.lookAt(
  [eyeX, eyeY, eyeZ], // Camera position
  [targetX, targetY, targetZ], // Look-at target
  [0, 1, 0], // Up direction
);
```

The view matrix is the **inverse** of the camera's world transform. Instead of moving the camera through the scene, we move the scene relative to the camera.

### Clip Space

Clip space is the four-dimensional coordinate system (x, y, z, w) that vertex shaders output. It's called "clip space" because coordinates outside the viewing frustum can be clipped (culled) by the GPU.

**Key Properties:**

- 4D homogeneous coordinates (x, y, z, w)
- Output as `@builtin(position)` from vertex shader
- Values are in the range that will map to NDC after division by w
- WebGPU clips primitives outside the viewing frustum

**Range in Clip Space (before perspective divide):**

- X: -w to +w (will become -1 to +1 in NDC)
- Y: -w to +w (will become -1 to +1 in NDC)
- Z: 0 to +w (will become 0 to +1 in NDC) ⚠️ **WebGPU-specific**
- W: positive value (typically distance from camera)

**In the shader:**

```wgsl
@vertex
fn vertexMain(input: Vertex) -> VertexOutput {
  var output: VertexOutput;

  // Transform through all spaces to clip space
  let worldPos = uniforms.modelMatrix * vec4f(input.position, 1.0);
  let viewPos = uniforms.viewMatrix * worldPos;
  let clipPos = uniforms.projectionMatrix * viewPos;

  output.position = clipPos;  // Clip space coordinates (x, y, z, w)
  return output;
}
```

### Normalized Device Coordinates (NDC)

NDC is the result of the **perspective divide** — dividing clip space coordinates (x, y, z) by w. This conversion happens automatically after the vertex shader.

**NDC Ranges in WebGPU:**

- X: -1 (left) to +1 (right)
- Y: -1 (bottom) to +1 (top)
- Z: 0 (near) to +1 (far) ⚠️ **Different from WebGL**

```
NDC Space (looking at screen):

    +1 __________ +1
       |        |
   -1  |        | +1 (X-axis)
       |        |
    -1 ----------
      -1        +1
   (Y-axis)

   Z-axis: 0 (near) → +1 (far)
```

**The Perspective Divide:**

```
NDC.x = clipSpace.x / clipSpace.w
NDC.y = clipSpace.y / clipSpace.w
NDC.z = clipSpace.z / clipSpace.w
```

This division creates the perspective effect where distant objects appear smaller.

### Screen/Viewport Space

Screen space (also called viewport space) represents the final pixel coordinates in the framebuffer.

**Characteristics:**

- Integer pixel coordinates
- Origin at **top-left** corner (0, 0) ⚠️ **Different from WebGL**
- X increases to the right
- Y increases **downward**
- Depth remains in range [0, 1]

```
Framebuffer/Screen Space:

(0,0) +---------------------+
      |                     |
      |    Y increases      |
      |    downward         |
      |         ↓           |
      |                     |
      +---------------------+ (width, height)
   X increases rightward →
```

**Transformation from NDC:**

```javascript
// Viewport transformation (handled by GPU)
screenX = (ndc.x * 0.5 + 0.5) * viewport.width + viewport.x;
screenY = (-ndc.y * 0.5 + 0.5) * viewport.height + viewport.y; // Note Y flip
screenZ = ndc.z * (maxDepth - minDepth) + minDepth;
```

Note the Y-axis flip: NDC Y=-1 maps to the bottom of the viewport, but in screen space, this becomes the maximum Y value because the origin is at the top.

## Depth Range Differences

The most significant difference between WebGPU and WebGL is the **depth range** in clip space and NDC.

### WebGPU: 0 to 1

In WebGPU, clip space Z ranges from **0 to +w**, and after perspective divide, NDC Z ranges from **0 to 1**:

- **Near plane**: Z = 0
- **Far plane**: Z = 1
- **Advantages**: Better depth precision, matches modern APIs
- **Depth clear value**: Typically 1.0 (farthest)
- **Depth test**: Usually `'less'` (closer values pass)

This convention provides **better floating-point precision** in the depth buffer, especially for distant objects, reducing z-fighting artifacts.

### WebGL: -1 to 1

WebGL follows OpenGL conventions where NDC Z ranges from **-1 to 1**:

- **Near plane**: Z = -1
- **Far plane**: Z = +1
- **Depth clear value**: Typically 1.0
- **Depth test**: Usually `gl.LESS`

### Projection Matrix Adjustments

The depth range difference requires different projection matrices. You cannot use WebGL projection matrices directly in WebGPU.

**Perspective Projection (WebGPU):**

```javascript
// Using wgpu-matrix (automatically handles 0-to-1 depth)
const projectionMatrix = mat4.perspective(
  fieldOfViewRadians,
  aspectRatio,
  near,
  far,
);
```

**Manual Projection Matrix Construction:**

For WebGPU (0 to 1 depth):

```javascript
function perspectiveZO(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const rangeInv = 1.0 / (near - far);

  return [
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    far * rangeInv,
    -1,
    0,
    0,
    near * far * rangeInv,
    0,
  ];
}
```

For WebGL (-1 to 1 depth):

```javascript
function perspectiveGL(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const rangeInv = 1.0 / (near - far);

  return [
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (near + far) * rangeInv,
    -1,
    0,
    0,
    2 * near * far * rangeInv,
    0,
  ];
}
```

**Key Differences:**

- Row 2 (Z calculation) differs significantly
- WebGPU: `Z_clip = far / (near - far) * Z_view + (near * far) / (near - far)`
- WebGL: `Z_clip = (near + far) / (near - far) * Z_view + (2 * near * far) / (near - far)`

**Orthographic Projection:**

For WebGPU:

```javascript
function ortho(left, right, bottom, top, near, far) {
  const dst = new Float32Array(16);

  dst[0] = 2 / (right - left);
  dst[5] = 2 / (top - bottom);
  dst[10] = 1 / (near - far); // WebGPU: maps to [0, 1]

  dst[12] = (right + left) / (left - right);
  dst[13] = (top + bottom) / (bottom - top);
  dst[14] = near / (near - far); // WebGPU-specific
  dst[15] = 1;

  return dst;
}
```

For WebGL:

```javascript
function orthoGL(left, right, bottom, top, near, far) {
  const dst = new Float32Array(16);

  dst[0] = 2 / (right - left);
  dst[5] = 2 / (top - bottom);
  dst[10] = 2 / (near - far); // WebGL: maps to [-1, 1]

  dst[12] = (right + left) / (left - right);
  dst[13] = (top + bottom) / (bottom - top);
  dst[14] = (near + far) / (near - far); // WebGL-specific
  dst[15] = 1;

  return dst;
}
```

## Y-Axis Direction

WebGPU's Y-axis direction varies depending on the coordinate space, which can be confusing when migrating from WebGL.

### World Space

In world space and view space, WebGPU conventionally uses **Y-up**:

- +Y points upward
- -Y points downward
- Standard for 3D scene graphs

```javascript
// Typical world-space camera setup
const cameraPosition = [0, 5, 10]; // Camera 5 units above ground
const target = [0, 0, 0]; // Looking at origin
const up = [0, 1, 0]; // Y-up orientation
```

### Framebuffer/Viewport

In framebuffer coordinates, **Y points down** with origin at top-left:

- Origin: (0, 0) at **top-left** corner
- +X points right
- +Y points **down**
- Different from OpenGL (bottom-left origin)

```wgsl
// Fragment shader - builtin position in screen space
@fragment
fn fragmentMain(@builtin(position) pixelPos: vec4f) -> @location(0) vec4f {
  // pixelPos.x: 0 at left edge
  // pixelPos.y: 0 at TOP edge (increases downward)
  // pixelPos.z: depth value [0, 1]
}
```

### NDC to Screen Transformation

The GPU automatically handles the Y-flip when transforming from NDC to screen space:

```
NDC Y = +1 (top)    →  Screen Y = 0 (top)
NDC Y = -1 (bottom) →  Screen Y = height (bottom)
```

This is different from WebGL, where:

```
WebGL NDC Y = +1 (top)    →  Screen Y = height (top)
WebGL NDC Y = -1 (bottom) →  Screen Y = 0 (bottom)
```

## Texture Coordinate Conventions

Texture coordinates (UV coordinates) in WebGPU follow specific conventions that affect how textures are sampled and displayed.

### UV Origin Position

Texture coordinates range from **0.0 to 1.0** in both U (horizontal) and V (vertical) directions:

- U = 0: left edge of texture
- U = 1: right edge of texture
- V = 0: **first texel** in texture data
- V = 1: **last texel** in texture data

**Important:** The coordinate (0, 0) maps to the **first texel** in the texture data array, regardless of how you interpret "top" or "bottom" visually.

```
Texture Coordinates:

(0,0) +----------+ (1,0)
      |          |
      | Texture  |
      |          |
(0,1) +----------+ (1,1)
```

### Data Organization and Flipping

Whether V=0 represents the "top" or "bottom" visually depends on how texture data is organized. Most image formats store data with the first row at the top, which can cause textures to appear **upside down** in WebGPU.

**Two common solutions:**

1. **Flip texture coordinates in shader:**

```wgsl
@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
  // Flip V coordinate
  let flippedUV = vec2f(uv.x, 1.0 - uv.y);
  return textureSample(myTexture, mySampler, flippedUV);
}
```

2. **Flip texture data when loading:**

```javascript
// Many image loaders provide flip options
const bitmap = await createImageBitmap(blob, {
  imageOrientation: "flipY",
});
```

Or manually flip the pixel data:

```javascript
function flipImageData(data, width, height) {
  const bytesPerRow = width * 4; // RGBA
  const temp = new Uint8Array(bytesPerRow);

  for (let y = 0; y < height / 2; y++) {
    const topOffset = y * bytesPerRow;
    const bottomOffset = (height - y - 1) * bytesPerRow;

    // Swap rows
    temp.set(data.subarray(topOffset, topOffset + bytesPerRow));
    data.copyWithin(topOffset, bottomOffset, bottomOffset + bytesPerRow);
    data.set(temp, bottomOffset);
  }
}
```

### Comparison with WebGL

Both WebGL and WebGPU use the same UV coordinate range (0 to 1), but the interpretation can differ:

- **WebGPU**: Framebuffer Y origin at top-left; texture V=0 maps to first data element
- **WebGL**: Framebuffer Y origin at bottom-left; texture V=0 maps to first data element

The consistent approach: **V=0 always refers to the first element in texture data**. How you organize that data determines visual orientation.

## Viewport Transformation

The viewport defines how NDC coordinates map to screen pixels. WebGPU provides fine-grained control through the `setViewport()` method.

### GPURenderPassEncoder.setViewport()

```javascript
renderPass.setViewport(
  x, // X offset in pixels
  y, // Y offset in pixels (from top-left)
  width, // Viewport width
  height, // Viewport height
  minDepth, // Minimum depth (typically 0.0)
  maxDepth, // Maximum depth (typically 1.0)
);
```

**Parameters:**

- `x`, `y`: Viewport origin in pixels, relative to framebuffer top-left
- `width`, `height`: Viewport dimensions in pixels
- `minDepth`, `maxDepth`: Depth range mapping (usually 0.0 to 1.0)

**Example:**

```javascript
// Full framebuffer viewport
renderPass.setViewport(0, 0, canvas.width, canvas.height, 0, 1);

// Quarter-screen viewport (top-left quadrant)
renderPass.setViewport(0, 0, canvas.width / 2, canvas.height / 2, 0, 1);

// Split-screen (left half)
renderPass.setViewport(0, 0, canvas.width / 2, canvas.height, 0, 1);
```

### Scissor Rectangle

The scissor rectangle clips rendering to a specific region:

```javascript
renderPass.setScissorRect(x, y, width, height);
```

Unlike the viewport (which scales NDC), the scissor rectangle **discards** fragments outside the specified rectangle. Both can be used together for efficient multi-view rendering.

**Difference:**

- **Viewport**: Scales and transforms NDC to screen space
- **Scissor**: Hard clips fragments outside rectangle

## Practical Implications

Understanding coordinate systems has direct implications for common rendering tasks.

### Camera Setup

Setting up a camera requires careful attention to coordinate conventions:

```javascript
import { mat4 } from "wgpu-matrix";

// Camera parameters
const cameraPosition = [0, 10, 20]; // Y-up world space
const target = [0, 0, 0];
const up = [0, 1, 0];

// View matrix (world → view space)
const viewMatrix = mat4.lookAt(cameraPosition, target, up);

// Projection matrix (view → clip space)
const projectionMatrix = mat4.perspective(
  Math.PI / 4, // 45-degree field of view
  canvas.width / canvas.height, // Aspect ratio
  0.1, // Near plane
  100.0, // Far plane
);

// Combined view-projection matrix
const viewProjectionMatrix = mat4.multiply(projectionMatrix, viewMatrix);
```

**View Matrix Construction:**

The `lookAt` function creates a view matrix that:

1. Positions the camera at `cameraPosition`
2. Orients it to look at `target`
3. Uses `up` to determine roll orientation

Internally, it computes three perpendicular axes:

- **Forward** (Z-axis): from eye to target (negated for right-handed view space)
- **Right** (X-axis): cross product of forward and up
- **Up** (Y-axis): cross product of right and forward

Then inverts this camera transform to get the view matrix.

### Projection Matrices

Always use projection matrix functions designed for WebGPU's 0-to-1 depth range.

**Using wgpu-matrix:**

```javascript
import { mat4 } from "wgpu-matrix";

// Perspective projection (0-to-1 depth, correct for WebGPU)
const perspectiveMatrix = mat4.perspective(fov, aspect, near, far);

// Orthographic projection (0-to-1 depth)
const orthoMatrix = mat4.ortho(left, right, bottom, top, near, far);
```

**⚠️ Warning:** Do not use gl-matrix, glMatrix, or other WebGL libraries without modification, as they produce -1-to-1 depth range matrices.

### Converting WebGL Matrices

If you must use WebGL matrices, you can convert them:

```javascript
// Convert WebGL projection matrix to WebGPU
function convertWebGLToWebGPU(glMatrix) {
  const gpuMatrix = glMatrix.slice(); // Copy

  // Adjust Z mapping from [-1, 1] to [0, 1]
  // Row 2 (indices 8-11) needs modification
  gpuMatrix[8] = (gpuMatrix[8] + gpuMatrix[12]) * 0.5;
  gpuMatrix[9] = (gpuMatrix[9] + gpuMatrix[13]) * 0.5;
  gpuMatrix[10] = (gpuMatrix[10] + gpuMatrix[14]) * 0.5;
  gpuMatrix[11] = (gpuMatrix[11] + gpuMatrix[15]) * 0.5;

  return gpuMatrix;
}
```

However, it's **strongly recommended** to use wgpu-matrix or similar libraries designed for WebGPU.

### Texture Sampling

Be consistent with texture coordinate conventions:

```wgsl
@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
  // If your texture appears upside down, flip V:
  // let correctedUV = vec2f(uv.x, 1.0 - uv.y);

  // Sample texture with correct coordinates
  let color = textureSample(myTexture, mySampler, uv);
  return color;
}
```

**Vertex shader UV generation:**

```wgsl
@vertex
fn vertexMain(@location(0) position: vec3f) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.mvpMatrix * vec4f(position, 1.0);

  // Generate UVs from vertex position
  // Ensure V=0 at top if that's your convention
  output.uv = vec2f(
    position.x * 0.5 + 0.5,  // Map -1..1 to 0..1
    position.y * 0.5 + 0.5
  );

  return output;
}
```

## wgpu-matrix Helpers

The `wgpu-matrix` library provides WebGPU-specific matrix and vector operations.

### Installation

```bash
npm install wgpu-matrix
```

### Key Features

**Correct Depth Range:**
All projection functions use 0-to-1 depth range:

```javascript
import { mat4, vec3 } from "wgpu-matrix";

// Perspective projection (0-to-1 depth)
const proj = mat4.perspective(fov, aspect, near, far);

// Orthographic projection (0-to-1 depth)
const ortho = mat4.ortho(left, right, bottom, top, near, far);
```

**Matrix3 Padding:**
WebGPU requires mat3 uniforms to use 12 floats (not 9):

```javascript
// wgpu-matrix mat3 automatically uses 12 elements
const normalMatrix = mat4.toMat3(modelViewMatrix);
// Safe to upload directly to WebGPU uniform buffer
```

**View Matrix Creation:**

```javascript
// LookAt function
const viewMatrix = mat4.lookAt(
  [eyeX, eyeY, eyeZ], // Camera position
  [targetX, targetY, targetZ], // Look-at target
  [upX, upY, upZ], // Up direction
);

// Or manually construct camera transform and invert
const cameraMatrix = mat4.identity();
mat4.translate(cameraMatrix, cameraPosition, cameraMatrix);
mat4.rotateY(cameraMatrix, yaw, cameraMatrix);
mat4.rotateX(cameraMatrix, pitch, cameraMatrix);
const viewMatrix = mat4.inverse(cameraMatrix);
```

**Concise API:**

```javascript
// Operations return results directly (no pre-allocation needed)
const m1 = mat4.translation([1, 2, 3]);
const m2 = mat4.rotationY(Math.PI / 4);
const combined = mat4.multiply(m1, m2);

// Or use destination parameter for performance
const dst = mat4.create();
mat4.multiply(m1, m2, dst); // Result in dst
```

### Integration Example

```javascript
import { mat4 } from "wgpu-matrix";

class Camera {
  constructor(canvas) {
    this.position = [0, 5, 10];
    this.target = [0, 0, 0];
    this.up = [0, 1, 0];

    this.fov = Math.PI / 4;
    this.near = 0.1;
    this.far = 100.0;
    this.aspect = canvas.width / canvas.height;
  }

  getViewMatrix() {
    return mat4.lookAt(this.position, this.target, this.up);
  }

  getProjectionMatrix() {
    return mat4.perspective(this.fov, this.aspect, this.near, this.far);
  }

  getViewProjectionMatrix() {
    const view = this.getViewMatrix();
    const proj = this.getProjectionMatrix();
    return mat4.multiply(proj, view);
  }
}

// Usage
const camera = new Camera(canvas);
const vpMatrix = camera.getViewProjectionMatrix();

// Upload to uniform buffer
device.queue.writeBuffer(
  uniformBuffer,
  0,
  vpMatrix.buffer,
  vpMatrix.byteOffset,
  vpMatrix.byteLength,
);
```

## Migration from WebGL

Migrating from WebGL to WebGPU requires coordinate system adjustments.

### Coordinate System Changes

| Aspect                  | WebGL           | WebGPU             | Action Required                |
| ----------------------- | --------------- | ------------------ | ------------------------------ |
| **Clip Space Z**        | -w to +w        | 0 to +w            | ✅ Update projection matrices  |
| **NDC Z Range**         | -1 to +1        | 0 to +1            | ✅ Use perspectiveZO functions |
| **Framebuffer Origin**  | Bottom-left     | Top-left           | ⚠️ May affect Y calculations   |
| **Depth Clear Value**   | 1.0             | 1.0                | ✅ No change                   |
| **Texture Coordinates** | V=0 at bottom\* | V=0 at first texel | ⚠️ May need flipping           |

\*In WebGL convention, though V=0 still maps to first texel in data

### Matrix Library Adjustments

**Replace WebGL libraries:**

```javascript
// OLD (WebGL)
import { mat4 } from "gl-matrix";
const proj = mat4.perspective(mat4.create(), fov, aspect, near, far);

// NEW (WebGPU)
import { mat4 } from "wgpu-matrix";
const proj = mat4.perspective(fov, aspect, near, far);
```

**Or use conversion if necessary:**

```javascript
// If you must use gl-matrix
import { mat4 as glMat4 } from "gl-matrix";

function glToWebGPUProjection(glProj) {
  // Convert -1..1 depth to 0..1 depth
  const gpuProj = glProj.slice();
  gpuProj[8] = (gpuProj[8] + gpuProj[12]) / 2;
  gpuProj[9] = (gpuProj[9] + gpuProj[13]) / 2;
  gpuProj[10] = (gpuProj[10] + gpuProj[14]) / 2;
  gpuProj[11] = (gpuProj[11] + gpuProj[15]) / 2;
  return gpuProj;
}
```

### Shader Adjustments

**Vertex shader output:**

```wgsl
// WebGPU shader
@vertex
fn vertexMain(@location(0) position: vec3f) -> @builtin(position) vec4f {
  // Projection matrix must use 0-to-1 depth
  return uniforms.projectionMatrix * uniforms.viewMatrix *
         uniforms.modelMatrix * vec4f(position, 1.0);
}
```

No changes needed in shader code itself, but ensure matrices are correct.

### Common Fixes

**Issue: Objects not appearing**

- **Cause**: Using WebGL projection matrix (wrong depth range)
- **Fix**: Use wgpu-matrix or perspectiveZO function

**Issue: Depth test failures**

- **Cause**: Depth values outside [0, 1] range
- **Fix**: Verify near/far planes and projection matrix

**Issue: Textures upside down**

- **Cause**: Different framebuffer origin or texture data organization
- **Fix**: Flip V coordinate in shader or flip texture data

**Issue: Y-coordinates inverted**

- **Cause**: Assuming bottom-left origin in screen space
- **Fix**: Remember framebuffer origin is top-left in WebGPU

## Best Practices

### Consistent Conventions

1. **Use Y-up for 3D world space** (camera up = [0, 1, 0])
2. **Use wgpu-matrix or equivalent** for WebGPU-compatible matrices
3. **Document coordinate spaces** in comments
4. **Establish texture coordinate conventions** early in project

### Documentation in Code

```javascript
/**
 * Creates a perspective projection matrix for WebGPU.
 * Maps view space to clip space with 0-to-1 depth range.
 *
 * @param {number} fov - Field of view in radians (Y-axis)
 * @param {number} aspect - Width / height
 * @param {number} near - Near plane distance (Z > 0)
 * @param {number} far - Far plane distance (Z > near)
 * @returns {Float32Array} 4x4 projection matrix (column-major)
 */
function createProjectionMatrix(fov, aspect, near, far) {
  return mat4.perspective(fov, aspect, near, far);
}
```

### Coordinate Space Debugging

```wgsl
// Visualize coordinate space in fragment shader
@fragment
fn debugCoordinates(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  // Visualize depth
  let depth = fragCoord.z;  // [0, 1] in WebGPU

  // Visualize screen position
  let screenNorm = fragCoord.xy / vec2f(800.0, 600.0);  // Normalize to [0, 1]

  return vec4f(screenNorm.x, screenNorm.y, depth, 1.0);
}
```

### Matrix Validation

```javascript
// Validate projection matrix produces correct depth range
function validateProjectionMatrix(projMatrix, testPoint) {
  // Transform a point
  const clip = vec4.transformMat4(testPoint, projMatrix);
  const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];

  console.log("NDC Z:", ndc[2]); // Should be in [0, 1] for WebGPU

  if (ndc[2] < 0 || ndc[2] > 1) {
    console.warn("⚠️ Depth value outside [0, 1] - wrong projection matrix?");
  }
}
```

## Common Pitfalls

### Incorrect Projection Matrices

**Problem:** Using WebGL/OpenGL projection matrices

```javascript
// ❌ WRONG - produces -1 to +1 depth
import { perspective } from "gl-matrix";
const proj = perspective(mat4.create(), fov, aspect, near, far);
```

**Solution:** Use WebGPU-compatible library

```javascript
// ✅ CORRECT - produces 0 to +1 depth
import { mat4 } from "wgpu-matrix";
const proj = mat4.perspective(fov, aspect, near, far);
```

### Depth Precision Issues

**Problem:** Poor depth precision causing z-fighting

**Causes:**

- Near plane too close to 0
- Far/near ratio too large

**Solutions:**

```javascript
// ❌ Poor precision
const proj = mat4.perspective(fov, aspect, 0.001, 10000); // Ratio: 10,000,000

// ✅ Better precision
const proj = mat4.perspective(fov, aspect, 0.1, 100); // Ratio: 1,000
```

WebGPU's 0-to-1 depth range provides better precision than -1-to-1, but still benefits from reasonable near/far ratios.

**Advanced:** Consider reverse-Z for extreme view distances:

```javascript
// Reverse-Z: near=1, far=0 for better precision
const reverseZProj = mat4.perspective(fov, aspect, far, near);
// Requires depth compare function change: 'greater' instead of 'less'
```

### Texture Flipping Confusion

**Problem:** Textures appear upside down

**Diagnosis:**

- Check if V=0 is at top or bottom in your texture data
- Verify UV coordinates in vertex data

**Solutions:**

Option 1 - Flip in shader:

```wgsl
@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
  let flippedUV = vec2f(uv.x, 1.0 - uv.y);
  return textureSample(myTexture, mySampler, flippedUV);
}
```

Option 2 - Flip texture data:

```javascript
const bitmap = await createImageBitmap(blob, { imageOrientation: "flipY" });
```

Option 3 - Adjust vertex UVs:

```javascript
// Generate UVs with V=1 at top instead of V=0
const uvs = new Float32Array([
  0,
  1, // Top-left
  1,
  1, // Top-right
  0,
  0, // Bottom-left
  1,
  0, // Bottom-right
]);
```

### Framebuffer Origin Assumptions

**Problem:** Assuming bottom-left origin like WebGL

**Example issue:**

```javascript
// ❌ WRONG - assumes bottom-left origin
const y = canvas.height - mouseY;
```

**Solution:** Remember top-left origin

```javascript
// ✅ CORRECT - top-left origin
const y = mouseY; // Already correct for WebGPU
```

### Near/Far Plane Confusion

**Problem:** Swapping near and far planes

```javascript
// ❌ WRONG - far plane smaller than near
const proj = mat4.perspective(fov, aspect, 100, 0.1);

// ✅ CORRECT - near < far
const proj = mat4.perspective(fov, aspect, 0.1, 100);
```

### Matrix Multiplication Order

**Problem:** Incorrect matrix multiplication order

```javascript
// ❌ WRONG - matrices in wrong order
const mvp = mat4.multiply(model, mat4.multiply(view, projection));

// ✅ CORRECT - projection * view * model
const vp = mat4.multiply(projection, view);
const mvp = mat4.multiply(vp, model);

// In shader: mvp * position
// Order: position → model space → world space → view space → clip space
```

---

## Conclusion

Understanding WebGPU's coordinate systems is essential for correct rendering. Key takeaways:

1. **Depth range is 0 to 1**, not -1 to 1 (different from WebGL)
2. **Framebuffer origin is top-left**, not bottom-left
3. **Use wgpu-matrix** or similar libraries for correct projection matrices
4. **Right-handed Y-up convention** for world/view space
5. **Texture coordinate V=0** maps to first texel in data
6. **Viewport Y increases downward** in screen space

By following these conventions and using appropriate tools, you can avoid common pitfalls and create robust WebGPU applications that render correctly across all platforms.

---

**Further Reading:**

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [wgpu-matrix Documentation](https://wgpu-matrix.org/docs/)
- [WebGPU Best Practices](https://toji.dev/webgpu-best-practices/)
