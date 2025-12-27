---
title: Camera and View Matrices
sidebar:
  order: 25
---

## Overview

Cameras in WebGPU are implemented through matrix transformations. There's no built-in camera—you construct matrices that transform vertices from world space through camera space into clip space.

:::note[Camera Matrix Pipeline]
```
Model Matrix → World Space → View Matrix → Camera Space → Projection Matrix → Clip Space
```
The GPU pipeline expects vertices in clip space. Cameras are the view + projection portion.
:::

## Matrix Types

### View Matrix

The **view matrix** transforms world coordinates into camera-relative coordinates. It's the inverse of where the camera is positioned and oriented:

```javascript title="View matrix concept"
// Camera at position (10, 5, 20), looking at origin
const cameraPosition = [10, 5, 20];
const cameraTarget = [0, 0, 0];
const up = [0, 1, 0];

// View matrix moves everything opposite to camera
// So camera effectively sits at origin, looking down -Z
```

### Projection Matrix

The **projection matrix** transforms camera-space coordinates into clip space:

| Type | Use Case | Characteristics |
|------|----------|-----------------|
| **Perspective** | 3D scenes, games | Objects shrink with distance |
| **Orthographic** | 2D, CAD, UI | No depth foreshortening |

## wgpu-matrix Library

The recommended library for WebGPU matrix math:

```bash
npm install wgpu-matrix
```

### Basic Usage

```javascript title="wgpu-matrix basics"
import { mat4, vec3 } from "wgpu-matrix";

// Perspective projection
const fov = Math.PI / 4;  // 45 degrees
const aspect = canvas.width / canvas.height;
const near = 0.1;
const far = 1000;
const projection = mat4.perspective(fov, aspect, near, far);

// View matrix using lookAt
const eye = [10, 5, 20];
const target = [0, 0, 0];
const up = [0, 1, 0];
const view = mat4.lookAt(eye, target, up);

// Combined view-projection
const viewProjection = mat4.multiply(projection, view);
```

:::tip[WebGPU Clip Space]
wgpu-matrix uses WebGPU's Z clip space (0 to 1), unlike WebGL (-1 to 1). Don't use gl-matrix directly—its projection functions target the wrong Z range.
:::

### Performance Pattern

```javascript title="Pre-allocated matrices"
// Allocate once
const view = mat4.create();
const projection = mat4.create();
const viewProjection = mat4.create();

function updateCamera() {
  // Reuse existing arrays (no allocation)
  mat4.lookAt(eye, target, up, view);
  mat4.perspective(fov, aspect, near, far, projection);
  mat4.multiply(projection, view, viewProjection);
}
```

## The lookAt Function

`lookAt` constructs a view matrix from camera parameters:

```javascript title="lookAt parameters"
mat4.lookAt(
  eye,     // Camera position [x, y, z]
  target,  // Point camera looks at [x, y, z]
  up       // Which direction is up [x, y, z], typically [0, 1, 0]
);
```

### How lookAt Works

1. Compute forward vector: `normalize(target - eye)`
2. Compute right vector: `cross(forward, up)`
3. Compute true up: `cross(right, forward)`
4. Build rotation from these three perpendicular axes
5. Combine with translation to camera position

```javascript title="lookAt internals (simplified)"
function lookAt(eye, target, up) {
  const forward = vec3.normalize(vec3.subtract(target, eye));
  const right = vec3.normalize(vec3.cross(forward, up));
  const trueUp = vec3.cross(right, forward);

  // 4x4 matrix with rotation and translation
  return mat4.fromValues(
    right[0],    right[1],    right[2],    0,
    trueUp[0],   trueUp[1],   trueUp[2],   0,
    -forward[0], -forward[1], -forward[2], 0,
    -vec3.dot(right, eye), -vec3.dot(trueUp, eye), vec3.dot(forward, eye), 1
  );
}
```

## Projection Matrices

### Perspective Projection

```javascript title="Perspective parameters"
mat4.perspective(
  fovY,    // Vertical field of view in radians
  aspect,  // Width / height ratio
  near,    // Near clipping plane (> 0)
  far      // Far clipping plane (> near)
);
```

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `fovY` | π/4 (45°) to π/2 (90°) | Wider = more visible, more distortion |
| `near` | 0.1 to 1.0 | Too small causes Z-fighting |
| `far` | 100 to 10000 | Limits visible distance |

:::danger[Near Plane Precision]
A near plane too close to 0 wastes depth buffer precision. Use the largest near value your scene allows. Ratio of far/near affects Z-fighting: keep it under 10000 when possible.
:::

### Orthographic Projection

```javascript title="Orthographic projection"
mat4.ortho(
  left,    // Left edge of view
  right,   // Right edge of view
  bottom,  // Bottom edge of view
  top,     // Top edge of view
  near,    // Near clipping plane
  far      // Far clipping plane
);
```

```javascript title="Common orthographic setups"
// 2D UI (pixel coordinates)
const uiProjection = mat4.ortho(0, canvas.width, canvas.height, 0, -1, 1);

// Centered orthographic
const size = 10;
const aspect = canvas.width / canvas.height;
const ortho = mat4.ortho(-size * aspect, size * aspect, -size, size, 0.1, 100);
```

## Camera Patterns

### Orbit Camera

Rotates around a target point—ideal for 3D viewers and editors:

```javascript title="Orbit camera implementation" {17-21}
class OrbitCamera {
  constructor() {
    this.target = [0, 0, 0];
    this.distance = 10;
    this.azimuth = 0;      // Horizontal angle (radians)
    this.elevation = 0.3;  // Vertical angle (radians)
  }

  getViewMatrix() {
    // Clamp elevation to avoid flipping
    const el = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.elevation));

    // Convert spherical to Cartesian
    const x = this.distance * Math.cos(el) * Math.sin(this.azimuth);
    const y = this.distance * Math.sin(el);
    const z = this.distance * Math.cos(el) * Math.cos(this.azimuth);

    const eye = [
      this.target[0] + x,
      this.target[1] + y,
      this.target[2] + z,
    ];

    return mat4.lookAt(eye, this.target, [0, 1, 0]);
  }

  rotate(deltaX, deltaY) {
    this.azimuth += deltaX * 0.01;
    this.elevation += deltaY * 0.01;
  }

  zoom(delta) {
    this.distance *= 1 + delta * 0.1;
    this.distance = Math.max(1, Math.min(100, this.distance));
  }
}
```

### First-Person Camera

Free-moving camera controlled by position and look direction:

```javascript title="First-person camera" {22-27}
class FirstPersonCamera {
  constructor() {
    this.position = [0, 1.7, 5];  // Eye height
    this.yaw = 0;    // Horizontal look angle
    this.pitch = 0;  // Vertical look angle
  }

  getViewMatrix() {
    // Calculate look direction from angles
    const forward = [
      Math.cos(this.pitch) * Math.sin(this.yaw),
      Math.sin(this.pitch),
      Math.cos(this.pitch) * Math.cos(this.yaw),
    ];

    const target = vec3.add(this.position, forward);
    return mat4.lookAt(this.position, target, [0, 1, 0]);
  }

  look(deltaX, deltaY) {
    this.yaw += deltaX * 0.002;
    this.pitch -= deltaY * 0.002;
    // Clamp pitch to avoid gimbal lock
    this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
  }

  move(forward, right) {
    // Move in look direction
    const dir = [
      Math.sin(this.yaw),
      0,
      Math.cos(this.yaw),
    ];

    this.position[0] += dir[0] * forward + dir[2] * right;
    this.position[2] += dir[2] * forward - dir[0] * right;
  }
}
```

### Turntable Camera

Like orbit, but rotation is around a fixed up axis:

```javascript title="Turntable (modeling software style)"
class TurntableCamera {
  constructor() {
    this.target = [0, 0, 0];
    this.distance = 10;
    this.theta = 0;     // Horizontal rotation
    this.phi = Math.PI / 4;  // Vertical angle from top
  }

  getViewMatrix() {
    const x = this.distance * Math.sin(this.phi) * Math.cos(this.theta);
    const y = this.distance * Math.cos(this.phi);
    const z = this.distance * Math.sin(this.phi) * Math.sin(this.theta);

    const eye = [
      this.target[0] + x,
      this.target[1] + y,
      this.target[2] + z,
    ];

    return mat4.lookAt(eye, this.target, [0, 1, 0]);
  }
}
```

## Mouse and Touch Controls

```javascript title="Camera input handling"
let isDragging = false;
let lastX = 0, lastY = 0;

canvas.addEventListener("mousedown", (e) => {
  isDragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDragging) return;

  const deltaX = e.clientX - lastX;
  const deltaY = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;

  camera.rotate(deltaX, deltaY);
});

canvas.addEventListener("mouseup", () => isDragging = false);
canvas.addEventListener("mouseleave", () => isDragging = false);

// Zoom with scroll wheel
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  camera.zoom(e.deltaY > 0 ? 1 : -1);
});
```

:::tip[Pointer Lock]
For first-person controls, use the Pointer Lock API for unlimited mouse movement:
```javascript
canvas.requestPointerLock();
document.addEventListener("pointerlockchange", () => {
  const locked = document.pointerLockElement === canvas;
});
```
:::

## Uploading to GPU

```javascript title="Camera uniform buffer" {17-21}
import { mat4 } from "wgpu-matrix";

// Create uniform buffer for camera matrices
const cameraBuffer = device.createBuffer({
  size: 64 * 3,  // 3 mat4x4 matrices
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

function updateCamera() {
  const view = camera.getViewMatrix();
  const projection = mat4.perspective(fov, aspect, near, far);
  const viewProjection = mat4.multiply(projection, view);

  // Upload all matrices
  device.queue.writeBuffer(cameraBuffer, 0, view);
  device.queue.writeBuffer(cameraBuffer, 64, projection);
  device.queue.writeBuffer(cameraBuffer, 128, viewProjection);
}
```

### WGSL Usage

```wgsl title="Camera in shader"
struct Camera {
  view: mat4x4f,
  projection: mat4x4f,
  viewProjection: mat4x4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn main(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return camera.viewProjection * vec4f(position, 1.0);
}
```

## TypeGPU Camera Buffer

```typescript title="TypeGPU camera setup"
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { mat4 } from "wgpu-matrix";

const CameraSchema = d.struct({
  view: d.mat4x4f,
  projection: d.mat4x4f,
  viewProjection: d.mat4x4f,
});

const cameraBuffer = root
  .createBuffer(CameraSchema)
  .$usage("uniform");

function updateCamera() {
  const view = camera.getViewMatrix();
  const projection = mat4.perspective(fov, aspect, 0.1, 1000);
  const viewProjection = mat4.multiply(projection, view);

  cameraBuffer.write({
    view: view as Float32Array,
    projection: projection as Float32Array,
    viewProjection: viewProjection as Float32Array,
  });
}
```

## Common Issues

:::danger[Gimbal Lock]
When pitch approaches ±90°, the camera can flip unexpectedly. Clamp pitch angles:
```javascript
this.pitch = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, this.pitch));
```
For unrestricted rotation, use quaternions instead of Euler angles.
:::

:::danger[Z-Fighting]
Objects at similar depths flicker. Solutions:
- Increase near plane distance
- Decrease far/near ratio
- Use logarithmic depth buffer for large scenes
:::

:::caution[Coordinate Handedness]
WebGPU uses a left-handed coordinate system in NDC (normalized device coordinates) with +Z pointing into the screen. Ensure your camera math matches.
:::

## Resources

:::note[References]
- [wgpu-matrix Documentation](https://wgpu-matrix.org/docs/)
- [wgpu-matrix GitHub](https://github.com/greggman/wgpu-matrix)
- [WebGPU Fundamentals: Cameras](https://webgpufundamentals.org/webgpu/lessons/webgpu-cameras.html)
- [WebGPU Fundamentals: Perspective Projection](https://webgpufundamentals.org/webgpu/lessons/webgpu-perspective-projection.html)
- [Learn WebGPU: Camera Control](https://eliemichel.github.io/LearnWebGPU/basic-3d-rendering/some-interaction/camera-control.html)
:::
