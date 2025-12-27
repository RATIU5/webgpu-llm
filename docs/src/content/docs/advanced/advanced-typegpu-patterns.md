---
title: Advanced TypeGPU Patterns
sidebar:
  order: 10
---

## Overview

TypeGPU provides advanced features for building production-ready GPU applications: externals for dependency management, the Resolve API for debugging, external textures for video processing, WebGPU interoperability, and React Native support.

## Externals and $uses

### Automatic Dependency Management

Externals allow TGSL functions to reference resources and other functions without manual binding management:

```typescript title="Externals in template literals"
const timeUniform = root.createBuffer(d.f32).$usage("uniform");
const particleBuffer = root.createBuffer(d.arrayOf(d.vec3f, 1000)).$usage("storage");

const updateParticles = tgpu.fn([d.u32], d.void).does`(idx: u32) {
    let dt = ${timeUniform};
    particles[idx] = particles[idx] + velocities[idx] * dt;
  }`.$uses({
  particles: particleBuffer,
  velocities: velocityBuffer,
});
```

:::note[Template Literal Capture]
Resources in `${}` are captured automatically. Resources referenced by name in WGSL need explicit `$uses()` declaration.
:::

### Nested Dependencies

TypeGPU walks the dependency tree automatically:

```typescript title="Dependency chain"
const computeGravity = tgpu.fn([d.vec3f, d.vec3f], d.vec3f)
  .does`(pos1: vec3f, pos2: vec3f) -> vec3f {
    let diff = pos2 - pos1;
    return normalize(diff) * ${d.f32(9.81)} / dot(diff, diff);
  }`;

const physicsUpdate = tgpu.fn([d.u32], d.void).does`(idx: u32) {
    totalForce += ${computeGravity}(positions[idx], otherPos);
  }`.$uses({ positions: positionBuffer });
// computeGravity is automatically included
```

## Resolve API

### Debugging Generated WGSL

Use `tgpu.resolve()` to inspect generated shader code:

```typescript title="Inspect WGSL output"
const wgsl = tgpu.resolve(myShaderFunction);
console.log(wgsl);

// With context for bind group information
const resolved = tgpu.resolveWithContext({
  main: shaderFunction,
  uniforms: uniformBuffer,
});
console.log("WGSL:", resolved.code);
console.log("Bind Groups:", resolved.bindGroupLayouts);
```

:::tip[Use Cases]
- Debug unexpected shader behavior
- Integrate with external profiling tools
- Pre-compile shaders for production
- Verify generated code in tests
:::

## External Textures

### Video Frame Processing

External textures enable zero-copy GPU access to video frames:

```typescript title="Process video frames" {5-6,11-12}
const video = document.createElement("video");
video.src = "video.mp4";
video.play();

function frame() {
  const externalTexture = device.importExternalTexture({ source: video });

  // Use in shader (must complete in same frame)
  processFrame(externalTexture);

  requestAnimationFrame(frame);
}
```

:::danger[Frame Lifetime]
External textures are destroyed automatically when JavaScript returns to the browser. Create and consume in the same callback.
:::

### Camera Input

```typescript title="Access camera stream"
const stream = await navigator.mediaDevices.getUserMedia({
  video: { width: 1920, height: 1080 },
});

const video = document.createElement("video");
video.srcObject = stream;
video.play();

// Use same pattern as video files
```

## WebGPU Interoperability

### Access Raw Resources

```typescript title="Unwrap TypeGPU to WebGPU"
const root = await tgpu.init();

// Get raw GPUDevice
const device = root.unwrap().device;

// Get raw GPUBuffer from typed buffer
const typedBuffer = root.createBuffer(schema, data).$usage("storage");
const rawBuffer = root.unwrap(typedBuffer);

// Use in vanilla WebGPU
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{ binding: 0, resource: { buffer: rawBuffer } }],
});
```

### Gradual Migration

TypeGPU is non-contagious—use it incrementally:

```typescript title="Mix TypeGPU with vanilla WebGPU"
// TypeGPU for type-safe buffer management
const typedBuffer = root.createBuffer(schema, data).$usage("storage");

// Vanilla WebGPU for custom pipeline
const rawBuffer = root.unwrap(typedBuffer);
const vanillaPipeline = device.createComputePipeline({ ... });

// TypeGPU for type-safe writes
await typedBuffer.write(newData);
```

## React Native Support

### Setup

```bash
npm install react-native-wgpu typegpu
npm install --save-dev @webgpu/types unplugin-typegpu
```

Configure Babel:

```javascript title="babel.config.js"
module.exports = (api) => {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
    plugins: ["unplugin-typegpu/babel"],
  };
};
```

### Mobile Rendering

```typescript title="React Native canvas" {16}
import { Canvas, useCanvasEffect } from 'react-native-wgpu';

export default function App() {
  const ref = useCanvasEffect(async () => {
    const context = ref.current!.getContext('webgpu')!;
    const device = await navigator.gpu.requestAdapter()
      .then(a => a!.requestDevice());

    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat() });

    const frame = () => {
      // ... render commands ...
      device.queue.submit([commandEncoder.finish()]);
      context.present();  // Required on React Native!
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  });

  return <Canvas ref={ref} style={{ flex: 1 }} />;
}
```

:::caution[Mobile Differences]
- Requires `context.present()` after submit
- iOS: Disable Metal validation in Xcode
- Expo: Must use `expo prebuild`, not Expo Go
:::

## Generator CLI (tgpu-gen)

Convert WGSL to TypeGPU:

```bash
# Single file
npx tgpu-gen src/shaders/compute.wgsl

# Glob pattern
npx tgpu-gen "src/shaders/**/*.wgsl"

# Watch mode
npx tgpu-gen "src/shaders/**/*.wgsl" --watch
```

:::tip[Integration]
Add to package.json:
```json
{
  "scripts": {
    "generate-shaders": "tgpu-gen \"src/shaders/**/*.wgsl\"",
    "prebuild": "npm run generate-shaders"
  }
}
```
:::

## CPU Simulation

Test shaders on CPU without GPU:

```typescript title="Unit test GPU functions"
import { test, expect } from "vitest";

const normalize2D = (x: number, y: number) => {
  "use gpu";
  const length = Math.sqrt(x * x + y * y);
  return { x: x / length, y: y / length };
};

test("normalize2D produces unit vectors", () => {
  const result = tgpu["~unstable"].simulate(normalize2D, [3, 4]);
  expect(result.x).toBeCloseTo(0.6);
  expect(result.y).toBeCloseTo(0.8);
});
```

:::caution[Limitations]
- No buffer/texture access
- CPU/GPU floating-point may differ slightly
- For correctness testing, not performance
:::

## Custom Extensions

### Building Domain Libraries

```typescript title="Particle system library"
import tgpu from "typegpu";
import * as d from "typegpu/data";

export const ParticleSchema = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

export class ParticleSystem {
  private particleBuffer: TgpuBuffer;

  constructor(root: TgpuRoot, maxParticles: number) {
    this.particleBuffer = root
      .createBuffer(d.arrayOf(ParticleSchema, maxParticles))
      .$usage("storage");
  }

  getBuffer() {
    return this.particleBuffer;
  }
}
```

### Custom Data Types

```typescript title="Physics type library"
import * as d from "typegpu/data";

export const Quaternion = d.vec4f;

export const RigidBody = d.struct({
  position: d.vec3f,
  rotation: Quaternion,
  linearVelocity: d.vec3f,
  angularVelocity: d.vec3f,
  mass: d.f32,
});
```

## Advanced Slots

### Slot Hierarchies

```typescript title="Nested configuration"
const simulationConfig = tgpu.slot({
  physics: tgpu.slot({
    gravity: d.f32,
    damping: d.f32,
  }),
  rendering: tgpu.slot({
    particleSize: d.f32,
  }),
});

const pipeline = root
  .makeComputePipeline(shader)
  .$with(simulationConfig, {
    physics: { gravity: -9.81, damping: 0.99 },
    rendering: { particleSize: 5.0 },
  });
```

### Runtime Configuration

```typescript title="Debug visualization modes"
const shaderMode = tgpu.slot(d.u32);

const shader = tgpu.fn([d.u32], d.vec4f).does`(idx: u32) -> vec4f {
    switch (${shaderMode}) {
      case 0u: { return normalVisualization(idx); }
      case 1u: { return depthVisualization(idx); }
      default: { return fullRendering(idx); }
    }
  }`;

// Switch at runtime
pipeline.$with(shaderMode, 1);
```

## Common Pitfalls

:::danger[Unwrapped Resources Lose Type Safety]
```typescript
const rawBuffer = root.unwrap(typedBuffer);
await rawBuffer.write([1, 2, 3]);  // TypeError!

// Use typed buffer for write operations
await typedBuffer.write([1, 2, 3]);  // Correct
```
:::

:::danger[External Texture Lifetime]
```typescript
// Wrong: texture destroyed before use
const tex = device.importExternalTexture({ source: video });
requestAnimationFrame(() => renderWithTexture(tex));  // Error!

// Correct: create and use in same frame
requestAnimationFrame(() => {
  const tex = device.importExternalTexture({ source: video });
  renderWithTexture(tex);
});
```
:::

:::danger[Editing Generated Files]
Never edit files generated by tgpu-gen. Import and extend instead:

```typescript
// compute.ts (generated - don't edit!)
export const updateParticles = /* ... */;

// particle-system.ts (your code)
import { updateParticles } from './compute';
export class ParticleSystem { /* custom logic */ }
```
:::

## Framework Integration

### Three.js with TypeGPU

Use TypeGPU for type-safe compute shaders alongside Three.js rendering:

```typescript title="Three.js + TypeGPU compute" {5-6,15-19}
import * as THREE from "three/webgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Share device between Three.js and TypeGPU
const renderer = new THREE.WebGPURenderer();
await renderer.init();

const root = await tgpu.init({
  device: renderer.backend.device,  // Reuse Three.js device
});

// TypeGPU compute for physics
const positions = root.createBuffer(d.arrayOf(d.vec3f, 1000)).$usage("storage");

// Three.js reads TypeGPU buffer
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(
  root.unwrap(positions),
  3
));

function animate() {
  // TypeGPU: Update physics
  physicsComputePipeline.dispatchWorkgroups(64);

  // Three.js: Render scene
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
```

:::tip[Shared Device]
Reuse the same `GPUDevice` between TypeGPU and Three.js to avoid synchronization overhead and enable zero-copy buffer sharing.
:::

### Babylon.js with TypeGPU

```typescript title="Babylon.js + TypeGPU"
import { Engine, Scene } from "@babylonjs/core";
import tgpu from "typegpu";

// Initialize Babylon with WebGPU
const engine = new Engine(canvas, true, {
  preserveDrawingBuffer: true,
  stencil: true,
});
await engine.initAsync();

// Initialize TypeGPU with Babylon's device
const root = await tgpu.init({
  device: engine._device,  // Access internal device
});

// Use TypeGPU for compute workloads
const computeBuffer = root
  .createBuffer(d.arrayOf(d.f32, 1024))
  .$usage("storage");

// Babylon handles rendering, TypeGPU handles compute
engine.runRenderLoop(() => {
  computePipeline.execute();  // TypeGPU compute
  scene.render();             // Babylon render
});
```

### Pattern: Hybrid Rendering

```typescript title="Framework renders, TypeGPU computes"
// 1. TypeGPU owns compute resources
const particleData = root.createBuffer(particleSchema, particles);
const computePipeline = root
  .withCompute(updateParticlesShader)
  .createPipeline();

// 2. Framework reads results
const rawBuffer = root.unwrap(particleData);
framework.setInstanceBuffer(rawBuffer);

// 3. Render loop
function frame() {
  computePipeline.execute();        // TypeGPU: physics
  framework.render(scene, camera);  // Framework: graphics
  requestAnimationFrame(frame);
}
```

:::caution[WebGPU Renderer Required]
Framework integration requires WebGPU backends:
- Three.js: `import * as THREE from "three/webgpu"`
- Babylon.js: Enable WebGPU engine mode

WebGL renderers cannot share resources with TypeGPU.
:::

## Resources

:::note[Official Documentation]
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/)
- [WebGPU Interoperability](https://docs.swmansion.com/TypeGPU/integration/webgpu-interoperability/)
- [React Native Support](https://docs.swmansion.com/TypeGPU/integration/react-native/)
- [Generator CLI](https://docs.swmansion.com/TypeGPU/tooling/tgpu-gen/)
- [Three.js WebGPU](https://threejs.org/docs/#manual/en/introduction/How-to-use-WebGPU)
:::
