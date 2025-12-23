---
title: Real-time Graphics Rendering
sidebar:
  order: 20
---

Real-time graphics rendering is the foundation of interactive 3D applications, games, and visualizations. With the advent of WebGPU and modern libraries like TypeGPU, developers can now build high-performance graphics applications directly in the browser with unprecedented control over the GPU pipeline.

This guide explores how to build interactive 3D visualizations using WebGPU and TypeGPU, focusing on the fundamental concepts, mathematical foundations, and practical implementations needed to create compelling real-time graphics.

## Overview

WebGPU is the next-generation graphics API for the web, providing low-level access to GPU capabilities with a modern, efficient design. TypeGPU builds on WebGPU by offering a type-safe, ergonomic TypeScript interface that makes GPU programming more accessible while maintaining performance.

Building interactive 3D visualizations requires understanding several key concepts:

- **Coordinate Systems**: How 3D space is represented and transformed
- **Transformation Matrices**: Mathematical tools for positioning, rotating, and projecting objects
- **Camera Systems**: Controlling the viewpoint and perspective
- **Lighting Models**: Simulating how light interacts with surfaces
- **Render Pipeline**: The sequence of operations that transforms 3D data into pixels

The combination of WebGPU's power and TypeGPU's developer-friendly abstractions creates an ideal environment for modern graphics programming. Unlike traditional graphics programming, WebGPU embraces explicit state management and async operations, giving you precise control over GPU resources and execution.

## 3D Graphics Fundamentals

### Coordinate Systems

Understanding coordinate systems is crucial for 3D graphics. WebGPU uses a **right-handed coordinate system** where:

- **X-axis**: Points right
- **Y-axis**: Points up
- **Z-axis**: Points toward the viewer (out of the screen)

This is the convention used by OpenGL and Vulkan, and it's important to maintain consistency throughout your application. In a right-handed system, if you point your right thumb along the positive X-axis and your index finger along the positive Y-axis, your middle finger points along the positive Z-axis.

The **clip space** in WebGPU ranges from -1 to +1 for X and Y axes, but crucially, **Z ranges from 0 to 1** (not -1 to 1 like in WebGL). This is a key difference that affects depth calculations and projection matrices.

### Transformation Matrices

Matrices are the fundamental tool for transforming objects in 3D space. A 4x4 matrix can represent translation, rotation, scaling, and projection operations. The beauty of matrices is that multiple transformations can be combined through multiplication.

In WebGPU's memory layout, a 4x4 matrix is stored in column-major order, which means each column is stored contiguously in memory. This affects how you pass data to shaders and how you think about matrix multiplication.

### The Rendering Equation (Simplified)

At its core, rendering is about solving the rendering equation, which describes how light interacts with surfaces. While the full equation is complex and used in path tracing, real-time graphics use simplified approximations.

The basic idea: **The color of a pixel depends on the surface properties and the light reaching it.**

For real-time applications, we typically use local illumination models like Phong or Blinn-Phong, which consider:

- **Emissive**: Light emitted by the surface itself
- **Ambient**: Constant background illumination
- **Diffuse**: Light scattered equally in all directions
- **Specular**: Light reflected in a preferred direction (shininess)

The complete rendering pipeline transforms vertices through several spaces: object space → world space → view space → clip space → screen space.

## Transformation Matrices

Transformation matrices are the workhorses of 3D graphics, enabling you to position, orient, and project 3D objects. The three primary transformation matrices form what's commonly called the **MVP** (Model-View-Projection) matrix chain.

### Model Matrix

The model matrix transforms vertices from **object space** (also called model space or local space) to **world space**. Object space is where your 3D model is defined—for example, a cube might be centered at the origin with vertices at ±1 units.

The model matrix typically combines three basic transformations:

**Translation**: Moving the object in 3D space. A translation matrix looks like:

```
[1  0  0  tx]
[0  1  0  ty]
[0  0  1  tz]
[0  0  0  1 ]
```

**Rotation**: Rotating the object around an axis. Rotation matrices are more complex and can be around X, Y, or Z axes, or around an arbitrary axis.

**Scale**: Changing the size of the object. A scale matrix:

```
[sx 0  0  0]
[0  sy 0  0]
[0  0  sz 0]
[0  0  0  1]
```

These transformations are typically applied in the order: Scale → Rotate → Translate (SRT). This order matters because matrix multiplication is not commutative. Applying them in this order ensures the object scales and rotates around its local origin, then moves to its world position.

### View Matrix

The view matrix transforms from **world space** to **view space** (also called camera space or eye space). This matrix represents the inverse of the camera's transformation—if the camera moves right, the world appears to move left.

The view matrix is typically constructed using a camera position (eye point), a target point to look at, and an up vector that defines which direction is "up" for the camera. This is often called a "look-at" matrix.

The view matrix effectively positions all objects in the world relative to the camera, placing the camera at the origin looking down the negative Z-axis (in a right-handed system).

### Projection Matrix

The projection matrix transforms from **view space** to **clip space**, applying perspective foreshortening—making distant objects appear smaller.

**Perspective Projection** is what creates realistic 3D depth. It requires four parameters:

- **Field of View (FOV)**: The angular extent of the view, typically 45-90 degrees
- **Aspect Ratio**: Width divided by height of the viewport
- **Near Plane**: The closest distance that will be rendered
- **Far Plane**: The farthest distance that will be rendered

Objects between the near and far planes, within the field of view, are mapped to clip space. The GPU then performs perspective division (dividing by the W component) to create normalized device coordinates.

**PerspectiveZO for WebGPU**: Because WebGPU uses a 0-1 depth range (unlike WebGL's -1 to 1), you must use projection matrix functions designed for this. The `wgpu-matrix` library provides `perspectiveZO` specifically for WebGPU, ensuring depth values map correctly to the 0-1 range.

## Using wgpu-matrix

The `wgpu-matrix` library is specifically designed for WebGPU, handling the nuances of WebGPU's coordinate systems and memory layouts. It provides efficient, well-tested implementations of common matrix and vector operations.

### Installation

Installing wgpu-matrix is straightforward via npm:

```bash
npm install wgpu-matrix
```

Or using other package managers:

```bash
yarn add wgpu-matrix
pnpm add wgpu-matrix
```

You can also use it directly via ES modules from a CDN:

```typescript
import {
  vec3,
  mat4,
} from "https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js";
```

### Integration with TypeGPU

TypeGPU provides typed GPU data structures that work seamlessly with wgpu-matrix. When defining uniforms or vertex data, you can use TypeGPU's `d.mat4x4f` type for matrices and pass wgpu-matrix generated matrices directly.

The key is that wgpu-matrix produces `Float32Array` instances by default, which are exactly what WebGPU expects. Here's how the types align:

```typescript
import { mat4, vec3 } from "wgpu-matrix";
import { d } from "typegpu";

// TypeGPU schema definition
const UniformsSchema = d.struct({
  modelMatrix: d.mat4x4f,
  viewMatrix: d.mat4x4f,
  projectionMatrix: d.mat4x4f,
});

// Generate matrices with wgpu-matrix
const modelMatrix = mat4.identity();
const viewMatrix = mat4.lookAt([0, 5, 10], [0, 0, 0], [0, 1, 0]);
const projectionMatrix = mat4.perspective(
  Math.PI / 4, // 45 degree FOV
  canvas.width / canvas.height,
  0.1, // near
  100, // far
);

// These can be passed directly to TypeGPU buffer writes
uniformBuffer.write({ modelMatrix, viewMatrix, projectionMatrix });
```

### Common Operations

The wgpu-matrix library provides a clean, functional API. Here are essential operations:

**Creating Matrices:**

```typescript
import { mat4, vec3 } from "wgpu-matrix";

// Identity matrix
const identity = mat4.identity();

// Translation
const translation = mat4.translation([5, 2, -3]);

// Rotation (45 degrees around Y-axis)
const rotation = mat4.rotationY(Math.PI / 4);

// Scaling
const scale = mat4.scaling([2, 2, 2]);

// Combined transformations
const modelMatrix = mat4.create();
mat4.translate(modelMatrix, [5, 2, -3], modelMatrix);
mat4.rotateY(modelMatrix, Math.PI / 4, modelMatrix);
mat4.scale(modelMatrix, [2, 2, 2], modelMatrix);
```

**View and Projection Matrices:**

```typescript
// Look-at view matrix
const viewMatrix = mat4.lookAt(
  [0, 5, 10], // eye position
  [0, 0, 0], // target position
  [0, 1, 0], // up vector
);

// Perspective projection (WebGPU style, 0-1 depth)
const projectionMatrix = mat4.perspective(
  Math.PI / 3, // 60 degree field of view
  canvas.width / canvas.height, // aspect ratio
  0.1, // near plane
  1000, // far plane
);
```

**Vector Operations:**

```typescript
// Create vectors
const position = vec3.create(1, 2, 3);
const direction = vec3.create(0, 0, -1);

// Normalize a vector
const normalized = vec3.normalize(direction);

// Dot product
const dot = vec3.dot(position, direction);

// Cross product
const cross = vec3.cross([1, 0, 0], [0, 1, 0]); // Results in [0, 0, 1]

// Add vectors
const sum = vec3.add(position, direction);
```

**Matrix Multiplication:**

```typescript
// Combine transformations
const modelViewMatrix = mat4.multiply(viewMatrix, modelMatrix);
const mvpMatrix = mat4.multiply(projectionMatrix, modelViewMatrix);
```

**Important**: Matrix multiplication order matters! In the example above, to transform a vertex, you multiply: projection × view × model × vertex. This reads right-to-left: first transform by model, then view, then projection.

## Setting Up the Render Loop

Real-time graphics require continuous rendering to create smooth animation and respond to user input. The render loop is the heart of any interactive graphics application.

### requestAnimationFrame Pattern

The `requestAnimationFrame` API is the standard way to create a render loop that synchronizes with the browser's refresh rate (typically 60 Hz).

```typescript
let lastTime = 0;

function renderLoop(currentTime: number) {
  // Calculate delta time in seconds
  const deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  // Update scene
  updateScene(deltaTime);

  // Render frame
  renderFrame();

  // Schedule next frame
  requestAnimationFrame(renderLoop);
}

// Start the loop
requestAnimationFrame(renderLoop);

function updateScene(deltaTime: number) {
  // Update animations, physics, etc.
  // deltaTime ensures consistent animation speed regardless of frame rate
  rotationAngle += rotationSpeed * deltaTime;
}

function renderFrame() {
  // Update transformation matrices
  const modelMatrix = mat4.rotationY(rotationAngle);

  // Encode render commands
  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });

  // Record render commands
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, uniformBindGroup);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.draw(vertexCount);
  renderPass.end();

  // Submit commands
  device.queue.submit([commandEncoder.finish()]);
}
```

**Delta Time**: Using delta time (the time elapsed since the last frame) ensures animations run at consistent speeds regardless of frame rate. Without delta time, animations would run faster on high-refresh displays and slower when performance drops.

### Handling Canvas Resize

Responsive graphics applications must handle canvas resizing gracefully:

```typescript
const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const canvas = entry.target as HTMLCanvasElement;
    const width = entry.contentBoxSize[0].inlineSize;
    const height = entry.contentBoxSize[0].blockSize;

    // Update canvas size
    canvas.width = Math.max(
      1,
      Math.min(width, device.limits.maxTextureDimension2D),
    );
    canvas.height = Math.max(
      1,
      Math.min(height, device.limits.maxTextureDimension2D),
    );

    // Update projection matrix with new aspect ratio
    projectionMatrix = mat4.perspective(
      fieldOfView,
      canvas.width / canvas.height, // Updated aspect ratio
      nearPlane,
      farPlane,
    );

    // Recreate depth buffer if using one
    depthTexture?.destroy();
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
});

resizeObserver.observe(canvas);
```

## Camera Systems

Cameras control the viewpoint in your 3D scene. Different camera systems suit different applications.

### Orbit Camera

An orbit camera rotates around a target point, perfect for viewing 3D models or architectural scenes:

```typescript
class OrbitCamera {
  target: vec3 = [0, 0, 0];
  distance: number = 10;
  azimuth: number = 0; // Horizontal rotation
  elevation: number = Math.PI / 4; // Vertical rotation

  constructor(canvas: HTMLCanvasElement) {
    this.setupControls(canvas);
  }

  setupControls(canvas: HTMLCanvasElement) {
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });

    canvas.addEventListener("mousemove", (e) => {
      if (!isDragging) return;

      const deltaX = e.clientX - lastX;
      const deltaY = e.clientY - lastY;

      this.azimuth -= deltaX * 0.01;
      this.elevation = Math.max(
        -Math.PI / 2,
        Math.min(Math.PI / 2, this.elevation - deltaY * 0.01),
      );

      lastX = e.clientX;
      lastY = e.clientY;
    });

    canvas.addEventListener("mouseup", () => {
      isDragging = false;
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.distance = Math.max(
        1,
        Math.min(100, this.distance + e.deltaY * 0.01),
      );
    });
  }

  getViewMatrix(): mat4 {
    // Calculate camera position
    const x =
      this.target[0] +
      this.distance * Math.cos(this.elevation) * Math.sin(this.azimuth);
    const y = this.target[1] + this.distance * Math.sin(this.elevation);
    const z =
      this.target[2] +
      this.distance * Math.cos(this.elevation) * Math.cos(this.azimuth);

    return mat4.lookAt([x, y, z], this.target, [0, 1, 0]);
  }
}
```

### First-Person Camera

A first-person camera moves through the scene, ideal for games and virtual walkthroughs:

```typescript
class FirstPersonCamera {
  position: vec3 = [0, 0, 5];
  yaw: number = 0; // Horizontal rotation
  pitch: number = 0; // Vertical rotation
  moveSpeed: number = 5;
  lookSpeed: number = 0.002;

  keys: Set<string> = new Set();

  constructor(canvas: HTMLCanvasElement) {
    this.setupControls(canvas);
  }

  setupControls(canvas: HTMLCanvasElement) {
    document.addEventListener("keydown", (e) => this.keys.add(e.code));
    document.addEventListener("keyup", (e) => this.keys.delete(e.code));

    canvas.addEventListener("click", () => canvas.requestPointerLock());

    document.addEventListener("mousemove", (e) => {
      if (document.pointerLockElement === canvas) {
        this.yaw -= e.movementX * this.lookSpeed;
        this.pitch = Math.max(
          -Math.PI / 2,
          Math.min(Math.PI / 2, this.pitch - e.movementY * this.lookSpeed),
        );
      }
    });
  }

  update(deltaTime: number) {
    // Calculate forward and right vectors
    const forward: vec3 = [
      Math.sin(this.yaw) * Math.cos(this.pitch),
      Math.sin(this.pitch),
      -Math.cos(this.yaw) * Math.cos(this.pitch),
    ];

    const right: vec3 = vec3.normalize(vec3.cross(forward, [0, 1, 0]));

    const speed = this.moveSpeed * deltaTime;

    // WASD movement
    if (this.keys.has("KeyW")) {
      this.position = vec3.add(this.position, vec3.scale(forward, speed));
    }
    if (this.keys.has("KeyS")) {
      this.position = vec3.subtract(this.position, vec3.scale(forward, speed));
    }
    if (this.keys.has("KeyD")) {
      this.position = vec3.add(this.position, vec3.scale(right, speed));
    }
    if (this.keys.has("KeyA")) {
      this.position = vec3.subtract(this.position, vec3.scale(right, speed));
    }
  }

  getViewMatrix(): mat4 {
    const forward: vec3 = [
      Math.sin(this.yaw),
      Math.sin(this.pitch),
      -Math.cos(this.yaw),
    ];

    const target = vec3.add(this.position, forward);
    return mat4.lookAt(this.position, target, [0, 1, 0]);
  }
}
```

## Basic Lighting

Lighting breathes life into 3D scenes, revealing form and creating atmosphere.

### The Phong Model

The Phong lighting model is a classic approach that balances quality and performance. It consists of three components:

**Ambient Light**: Constant illumination that simulates indirect light bouncing around the environment. Without ambient light, surfaces facing away from lights would be completely black.

```glsl
vec3 ambient = ambientColor * ambientStrength;
```

**Diffuse Light**: Directional light that's scattered equally in all directions. The intensity depends on the angle between the surface normal and the light direction. Surfaces facing the light are brighter.

```glsl
float diffuseStrength = max(dot(normal, lightDirection), 0.0);
vec3 diffuse = lightColor * diffuseStrength;
```

**Specular Light**: Reflected light that creates highlights. The intensity depends on the viewer's position, creating shiny spots on surfaces.

```glsl
vec3 viewDirection = normalize(cameraPosition - fragPosition);
vec3 reflectDirection = reflect(-lightDirection, normal);
float specularStrength = pow(max(dot(viewDirection, reflectDirection), 0.0), shininess);
vec3 specular = lightColor * specularStrength * specularColor;
```

### Lighting in Shaders

Here's a complete fragment shader implementing Phong lighting:

```wgsl
struct Uniforms {
  modelMatrix: mat4x4f,
  viewMatrix: mat4x4f,
  projectionMatrix: mat4x4f,
  normalMatrix: mat4x4f,
  cameraPosition: vec3f,
  lightPosition: vec3f,
  lightColor: vec3f,
  ambientStrength: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) normal: vec3f,
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // Normalize the interpolated normal
  let normal = normalize(input.normal);

  // Calculate light direction
  let lightDirection = normalize(uniforms.lightPosition - input.worldPosition);

  // Ambient component
  let ambient = uniforms.ambientStrength * uniforms.lightColor;

  // Diffuse component
  let diffuseStrength = max(dot(normal, lightDirection), 0.0);
  let diffuse = diffuseStrength * uniforms.lightColor;

  // Specular component
  let viewDirection = normalize(uniforms.cameraPosition - input.worldPosition);
  let reflectDirection = reflect(-lightDirection, normal);
  let specularStrength = pow(max(dot(viewDirection, reflectDirection), 0.0), 32.0);
  let specular = specularStrength * uniforms.lightColor * 0.5;

  // Combine lighting components
  let objectColor = vec3f(0.8, 0.2, 0.3);  // Material color
  let finalColor = (ambient + diffuse + specular) * objectColor;

  return vec4f(finalColor, 1.0);
}
```

### Normal Transformation

A critical detail: normals must be transformed differently than positions. When you scale an object non-uniformly, the normals can become skewed if transformed by the same matrix.

The correct transformation for normals is the **normal matrix**, which is the transpose of the inverse of the model matrix (or model-view matrix if transforming to view space):

```typescript
const normalMatrix = mat4.transpose(mat4.inverse(modelMatrix));
```

In the vertex shader:

```wgsl
@vertex
fn vertexMain(
  @location(0) position: vec3f,
  @location(1) normal: vec3f
) -> VertexOutput {
  var output: VertexOutput;

  // Transform position to world space
  let worldPosition = uniforms.modelMatrix * vec4f(position, 1.0);
  output.worldPosition = worldPosition.xyz;

  // Transform normal to world space using normal matrix
  output.normal = (uniforms.normalMatrix * vec4f(normal, 0.0)).xyz;

  // Transform to clip space
  output.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;

  return output;
}
```

Note that normals are transformed with a W component of 0, because they're directions, not positions.

## Render Pass Configuration

WebGPU render passes define how rendering operations are performed and what happens to the output.

### Color Attachments

Color attachments specify where and how to write color data:

```typescript
const renderPassDescriptor: GPURenderPassDescriptor = {
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),

      // Clear color (dark gray)
      clearValue: { r: 0.2, g: 0.2, b: 0.2, a: 1.0 },

      // 'clear' wipes the texture before rendering
      // 'load' preserves existing contents
      loadOp: "clear",

      // 'store' writes results back to texture
      // 'discard' throws away results (useful for intermediate passes)
      storeOp: "store",
    },
  ],
};
```

### Depth Buffer

Depth testing ensures objects are drawn in the correct order based on their distance from the camera:

```typescript
// Create depth texture
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const renderPassDescriptor: GPURenderPassDescriptor = {
  colorAttachments: [
    /* ... */
  ],
  depthStencilAttachment: {
    view: depthTexture.createView(),

    depthClearValue: 1.0, // Far plane value
    depthLoadOp: "clear",
    depthStoreOp: "store",

    // Optional stencil operations
    // stencilClearValue: 0,
    // stencilLoadOp: 'clear',
    // stencilStoreOp: 'store',
  },
};

// Enable depth testing in pipeline
const pipelineDescriptor: GPURenderPipelineDescriptor = {
  // ... other configuration
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "less", // Closer objects win
  },
};
```

### Multiple Render Targets

For advanced techniques like deferred rendering, you can output to multiple textures simultaneously:

```typescript
const gBufferTextures = {
  albedo: device.createTexture({
    size: [canvas.width, canvas.height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  }),
  normal: device.createTexture({
    size: [canvas.width, canvas.height],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  }),
  position: device.createTexture({
    size: [canvas.width, canvas.height],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  }),
};

const renderPassDescriptor: GPURenderPassDescriptor = {
  colorAttachments: [
    { view: gBufferTextures.albedo.createView() /* ... */ },
    { view: gBufferTextures.normal.createView() /* ... */ },
    { view: gBufferTextures.position.createView() /* ... */ },
  ],
  depthStencilAttachment: {
    /* ... */
  },
};
```

The fragment shader outputs to multiple targets:

```wgsl
struct FragmentOutput {
  @location(0) albedo: vec4f,
  @location(1) normal: vec4f,
  @location(2) position: vec4f,
}

@fragment
fn fragmentMain(input: VertexOutput) -> FragmentOutput {
  var output: FragmentOutput;
  output.albedo = vec4f(materialColor, 1.0);
  output.normal = vec4f(normalize(input.normal) * 0.5 + 0.5, 1.0);
  output.position = vec4f(input.worldPosition, 1.0);
  return output;
}
```

## Complete Example: Rotating Lit Cube

Here's a complete TypeGPU implementation of a rotating, lit cube:

```typescript
import { mat4, vec3 } from "wgpu-matrix";
import tgpu, { d } from "typegpu";

// Cube geometry with normals
const cubeVertices = new Float32Array([
  // Front face
  -1, -1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, -1, 1, 1, 0, 0, 1,
  // Back face
  -1, -1, -1, 0, 0, -1, -1, 1, -1, 0, 0, -1, 1, 1, -1, 0, 0, -1, 1, -1, -1, 0,
  0, -1,
  // Top face
  -1, 1, -1, 0, 1, 0, -1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, -1, 0, 1, 0,
  // Bottom face
  -1, -1, -1, 0, -1, 0, 1, -1, -1, 0, -1, 0, 1, -1, 1, 0, -1, 0, -1, -1, 1, 0,
  -1, 0,
  // Right face
  1, -1, -1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, -1, 1, 1, 0, 0,
  // Left face
  -1, -1, -1, -1, 0, 0, -1, -1, 1, -1, 0, 0, -1, 1, 1, -1, 0, 0, -1, 1, -1, -1,
  0, 0,
]);

const cubeIndices = new Uint16Array([
  0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14,
  15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
]);

// Initialize TypeGPU
const root = await tgpu.init();

// Define uniform structure
const Uniforms = d.struct({
  modelMatrix: d.mat4x4f,
  viewMatrix: d.mat4x4f,
  projectionMatrix: d.mat4x4f,
  normalMatrix: d.mat4x4f,
  cameraPosition: d.vec3f,
  lightPosition: d.vec3f,
  lightColor: d.vec3f,
  ambientStrength: d.f32,
});

// Create buffers
const vertexBuffer = root.createBuffer(cubeVertices).$usage("vertex");
const indexBuffer = root.createBuffer(cubeIndices).$usage("index");
const uniformBuffer = root.createBuffer(Uniforms).$usage("uniform");

// Shader code
const vertexShader = `
@vertex
fn main(
  @location(0) position: vec3f,
  @location(1) normal: vec3f
) -> VertexOutput {
  var output: VertexOutput;
  let worldPos = uniforms.modelMatrix * vec4f(position, 1.0);
  output.worldPosition = worldPos.xyz;
  output.normal = (uniforms.normalMatrix * vec4f(normal, 0.0)).xyz;
  output.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
  return output;
}
`;

const fragmentShader = `
@fragment
fn main(input: VertexOutput) -> @location(0) vec4f {
  let normal = normalize(input.normal);
  let lightDir = normalize(uniforms.lightPosition - input.worldPosition);
  let viewDir = normalize(uniforms.cameraPosition - input.worldPosition);

  let ambient = uniforms.ambientStrength * uniforms.lightColor;
  let diffuse = max(dot(normal, lightDir), 0.0) * uniforms.lightColor;

  let reflectDir = reflect(-lightDir, normal);
  let specular = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * uniforms.lightColor * 0.5;

  let objectColor = vec3f(0.8, 0.3, 0.2);
  let finalColor = (ambient + diffuse + specular) * objectColor;

  return vec4f(finalColor, 1.0);
}
`;

// Animation state
let rotationAngle = 0;
const camera = new OrbitCamera(canvas);

// Render loop
function render(time: number) {
  rotationAngle = time * 0.001;

  // Update matrices
  const modelMatrix = mat4.rotationY(rotationAngle);
  const viewMatrix = camera.getViewMatrix();
  const projectionMatrix = mat4.perspective(
    Math.PI / 4,
    canvas.width / canvas.height,
    0.1,
    100,
  );
  const normalMatrix = mat4.transpose(mat4.inverse(modelMatrix));

  // Update uniforms
  uniformBuffer.write({
    modelMatrix,
    viewMatrix,
    projectionMatrix,
    normalMatrix,
    cameraPosition: [
      camera.position[0],
      camera.position[1],
      camera.position[2],
    ],
    lightPosition: [5, 5, 5],
    lightColor: [1, 1, 1],
    ambientStrength: 0.2,
  });

  // Render
  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  renderPass.setPipeline(pipeline);
  renderPass.setBindGroup(0, bindGroup);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.setIndexBuffer(indexBuffer, "uint16");
  renderPass.drawIndexed(cubeIndices.length);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);
  requestAnimationFrame(render);
}

requestAnimationFrame(render);
```

## Best Practices

Building performant real-time graphics requires attention to several key areas:

**Frame Budgeting**: At 60 FPS, you have only 16.67ms per frame. Monitor your frame times and profile GPU operations. Use browser developer tools' performance profiler to identify bottlenecks.

**Frustum Culling**: Don't render objects outside the camera's view. Test each object's bounding box against the view frustum before adding it to render lists. This can dramatically reduce draw calls.

```typescript
function isInFrustum(boundingBox: BoundingBox, mvpMatrix: mat4): boolean {
  // Test bounding box corners against frustum planes
  // Return false if all corners are outside any plane
  // This prevents rendering invisible objects
}
```

**LOD (Level of Detail)**: Use simpler geometry for distant objects. Switch between high, medium, and low polygon models based on distance from camera.

**Batching**: Group similar objects to reduce state changes. Each pipeline switch and bind group change has overhead.

**Texture Management**: Use texture atlases to reduce texture binding changes. Compress textures to reduce memory bandwidth.

**Occlusion Culling**: For complex scenes, don't render objects hidden behind others. This is more advanced but crucial for large environments.

**Buffer Management**: Reuse buffers where possible. Creating and destroying GPU resources is expensive.

## Common Pitfalls

**Depth Precision Issues**: The depth buffer has finite precision. The near/far plane ratio affects quality—keeping them close together improves precision. A near plane of 0.001 and far plane of 10000 will have terrible precision. Try near=0.1, far=100 instead.

**Matrix Order**: WebGPU uses column-major matrices. When multiplying matrices, remember the order matters: `projection × view × model`, not `model × view × projection`.

**Coordinate Handedness**: WebGPU uses right-handed coordinates but 0-1 depth. Mixing conventions from WebGL or other APIs will cause frustration. Always use `perspective` functions designed for WebGPU (like wgpu-matrix's default perspective function).

**Normal Transformation**: Never transform normals with the model matrix directly. Always use the normal matrix (transpose of inverse). Non-uniform scaling will break lighting otherwise.

**Uniform Alignment**: WebGPU has strict alignment requirements. Vec3 in uniforms is padded to vec4 size. Always check the specification or use tools that handle alignment automatically.

**Depth Format**: Use 'depth24plus' for general use. Don't assume exact format availability—always check device capabilities.

**Texture Mipmaps**: Forgetting to generate mipmaps causes texture aliasing and shimmering. Always generate mipmaps for textures viewed at varying distances.

**Memory Leaks**: WebGPU resources must be explicitly destroyed. Use `.destroy()` on textures, buffers, and pipelines when done. Monitor GPU memory usage in development.

## Conclusion

Real-time graphics rendering with WebGPU and TypeGPU opens up powerful possibilities for interactive 3D applications on the web. By understanding coordinate systems, transformation matrices, lighting models, and the rendering pipeline, you can create compelling visual experiences.

The combination of wgpu-matrix's mathematical foundations with TypeGPU's type-safe abstractions provides an excellent development experience. Start with simple scenes, understand each component deeply, and gradually add complexity as you master the fundamentals.

Remember that real-time graphics is an iterative process. Profile early, optimize when needed, and always prioritize understanding over premature optimization. The techniques covered here form the foundation for more advanced topics like shadows, post-processing effects, physically-based rendering, and compute shader integration.

With these tools and knowledge, you're well-equipped to build impressive real-time graphics applications that run smoothly in any modern browser.
