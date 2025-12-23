---
title: Advanced TypeGPU Patterns
sidebar:
  order: 10
---

## Overview

TypeGPU is a modular and open-ended toolkit for WebGPU that brings type safety and advanced features to GPU programming in JavaScript and TypeScript. While the fundamentals of TypeGPU cover data schemas, buffers, and basic shader functions, mastering advanced patterns enables building complex, production-ready GPU applications with improved modularity, performance, and maintainability.

This guide explores advanced TypeGPU features designed for sophisticated use cases: managing complex shader dependencies with externals and `$uses`, debugging transpilation with the Resolve API, processing video and camera streams with external textures, integrating TypeGPU with vanilla WebGPU code, deploying to mobile platforms via React Native, automating migrations with the Generator CLI, testing shaders on the CPU with simulation, building custom wrapper libraries, and creating dynamic pipelines with advanced slot patterns.

These patterns are essential when building real-world applications like video processing pipelines, cross-platform graphics engines, physics simulations, or machine learning inference systems on the GPU. Understanding these advanced capabilities allows developers to leverage TypeGPU's full potential while maintaining the flexibility to drop down to vanilla WebGPU when needed.

## Externals and $uses

### What are Externals?

Externals are dependencies that TGSL (TypeGPU Shading Language) functions can reference to access resources, constants, or other functions. In traditional WGSL development, every shader must explicitly declare all resources it uses through binding declarations. TypeGPU's externals system automates dependency management, allowing shaders to reference GPU resources and other functions without manual binding index coordination.

When you define a function that needs access to buffers, textures, constants, or other GPU functions, you can pass these as externals. TypeGPU automatically resolves the entire dependency graph, embedding all necessary code and resources into the final shader without duplication or name clashes. This enables true modular shader development where utility functions can be distributed across multiple files or even npm packages.

Externals can include:

- Numbers, vectors, matrices, and other primitive constants
- Buffers and buffer views
- Textures and samplers
- Other TypeGPU functions
- Struct definitions and type schemas
- Module-level constants created with `tgpu.const()`

The key advantage is that externals are resolved automatically during shader compilation. TypeGPU analyzes the dependency graph, determines what needs to be included, and generates the appropriate WGSL code with all necessary declarations.

### Using $uses

The `$uses()` method explicitly declares dependencies for WGSL-implemented functions. While TGSL functions written in JavaScript automatically track external references through AST analysis, functions implemented directly in WGSL require manual dependency declaration.

Here's a basic example:

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Create resources
const particleBuffer = root
  .createBuffer(arrayOf(vec3f, 1000))
  .$usage("storage")
  .$name("particles");

const velocityBuffer = root
  .createBuffer(arrayOf(vec3f, 1000))
  .$usage("storage")
  .$name("velocities");

const timeUniform = root.createBuffer(f32).$usage("uniform").$name("deltaTime");

// Define a function with external dependencies
const updateParticles = tgpu.fn([d.u32], d.void).does`(idx: u32) {
    let dt = ${timeUniform};
    particles[idx] = particles[idx] + velocities[idx] * dt;
  }`.$uses({
  particles: particleBuffer,
  velocities: velocityBuffer,
});
```

In this example, `timeUniform` is captured automatically from the template literal, but `particles` and `velocities` need explicit declaration via `$uses()` since they're referenced by name in the WGSL code.

### Advanced $uses Patterns

For complex applications with nested dependencies, `$uses()` creates a clean dependency graph:

```typescript
// Utility function with dependencies
const computeGravity = tgpu.fn([d.vec3f, d.vec3f], d.vec3f)
  .does`(pos1: vec3f, pos2: vec3f) -> vec3f {
    let diff = pos2 - pos1;
    let distSq = dot(diff, diff) + ${d.f32(0.0001)};
    let force = ${d.f32(9.81)} / distSq;
    return normalize(diff) * force;
  }`;

// Main compute function using the utility
const physicsUpdate = tgpu.fn([d.u32], d.void).does`(idx: u32) {
    var totalForce = vec3f(0.0);
    for (var i = 0u; i < ${d.u32(100)}; i++) {
      if (i != idx) {
        totalForce += ${computeGravity}(positions[idx], positions[i]);
      }
    }
    velocities[idx] += totalForce * deltaTime;
    positions[idx] += velocities[idx] * deltaTime;
  }`.$uses({
  positions: positionBuffer,
  velocities: velocityBuffer,
  deltaTime: timeBuffer,
});
```

The `computeGravity` function is automatically included when `physicsUpdate` is resolved, even though it's not explicitly listed in `$uses()`. TypeGPU walks the dependency tree and includes everything needed.

### When to Use Externals and $uses

Use externals and `$uses()` when:

**Building Modular Shader Libraries**: Create reusable shader utilities that can be imported across projects. Since externals are resolved automatically, users don't need to manually copy dependencies.

**Managing Complex Dependency Graphs**: For shaders with many interconnected resources and utility functions, externals prevent binding index conflicts and make code more maintainable.

**Cross-Package Shader Composition**: When publishing TypeGPU-based libraries to npm, externals enable other developers to use your functions without understanding internal implementation details.

**Incremental Migration**: When migrating existing WGSL code to TypeGPU, `$uses()` allows keeping performance-critical shaders in WGSL while still benefiting from TypeGPU's resource management.

Avoid `$uses()` when:

- Working with simple, standalone shaders with few dependencies
- All code is written in TGSL (JavaScript), which tracks dependencies automatically
- Performance profiling shows dependency resolution overhead (rare but possible)

## Resolve API

### tgpu.resolve()

The Resolve API provides manual control over WGSL code generation from TypeGPU objects. While TypeGPU typically handles shader compilation transparently when creating pipelines, the Resolve API exposes this process for debugging, inspection, and integration with other tools.

The basic `tgpu.resolve()` function converts any TypeGPU object into WGSL code:

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

const computeDistance = tgpu.fn([d.vec3f, d.vec3f], d.f32)
  .does`(a: vec3f, b: vec3f) -> f32 {
    let diff = b - a;
    return sqrt(dot(diff, diff));
  }`;

// Generate WGSL code
const wgslCode = tgpu.resolve(computeDistance);

console.log(wgslCode);
// Output:
// fn computeDistance_0(a: vec3f, b: vec3f) -> f32 {
//   let diff = b - a;
//   return sqrt(dot(diff, diff));
// }
```

TypeGPU automatically assigns unique names to avoid conflicts. The `_0` suffix ensures that if you resolve the same function multiple times, each gets a unique identifier.

### Advanced Resolution with Context

For more control, use `tgpu.resolveWithContext()`, which returns both the WGSL code and metadata about bind group layouts and dependencies:

```typescript
const particleShader = tgpu.fn([d.u32], d.void).does`(idx: u32) {
    positions[idx] += velocities[idx] * ${deltaTime};
  }`.$uses({
  positions: positionBuffer,
  velocities: velocityBuffer,
});

const resolved = tgpu.resolveWithContext({
  main: particleShader,
  deltaTime: timeUniform,
});

console.log("WGSL:", resolved.code);
console.log("Bind Groups:", resolved.bindGroupLayouts);
console.log("Dependencies:", resolved.dependencies);
```

This provides essential information for pipeline construction, showing exactly which resources need to be bound and at which indices.

### Use Cases for the Resolve API

**Inspecting Generated Code**: When debugging shader behavior, examine the exact WGSL that TypeGPU generates. This helps understand performance characteristics and catch potential issues.

```typescript
// Debug mode: print generated shader
if (DEBUG) {
  const wgsl = tgpu.resolve(myComplexShader);
  console.log("Generated WGSL:\n", wgsl);
}
```

**Integration with Other Tools**: Some WebGPU tools or debugging utilities expect raw WGSL. The Resolve API bridges TypeGPU with these tools:

```typescript
// Export shader for external tools
const wgslForAnalysis = tgpu.resolve(shaderFunction);
await sendToProfilingTool(wgslForAnalysis);
```

**Caching and Optimization**: In production applications, pre-resolve shaders and cache the WGSL to avoid runtime resolution overhead:

```typescript
// Build-time shader compilation
const precompiledShaders = {
  physics: tgpu.resolve(physicsShader),
  rendering: tgpu.resolve(renderShader),
  postProcess: tgpu.resolve(postProcessShader),
};

// Save to file for loading at runtime
fs.writeFileSync("shaders.json", JSON.stringify(precompiledShaders));
```

**Testing and Validation**: Verify that shader code generation produces expected output:

```typescript
import { test, expect } from "vitest";

test("shader generates correct WGSL", () => {
  const wgsl = tgpu.resolve(myShader);
  expect(wgsl).toContain("fn myShader");
  expect(wgsl).toContain("@compute");
  expect(wgsl).not.toContain("undefined");
});
```

**Learning and Documentation**: Generate WGSL examples for documentation or educational materials:

```typescript
// Generate example code for docs
const examples = [
  { name: "Basic Compute", fn: basicComputeShader },
  { name: "Particle Physics", fn: particleShader },
  { name: "Ray Tracing", fn: rayTraceShader },
].map(({ name, fn }) => ({
  name,
  typescript: fn.toString(),
  wgsl: tgpu.resolve(fn),
}));
```

## External Textures

### Video Frame Processing

External textures enable efficient video frame processing by allowing direct GPU access to video data without CPU-side copies. WebGPU's `GPUExternalTexture` provides zero-copy access to video frames, making it ideal for real-time video effects, computer vision, and media processing.

To use video frames in TypeGPU:

```typescript
import tgpu from "typegpu";

// HTML video element
const video = document.createElement("video");
video.src = "video.mp4";
video.loop = true;
video.play();

// Create shader that samples from external texture
const processVideoFrame = tgpu.fn([], d.void)
  .does`(@builtin(global_invocation_id) id: vec3u) {
    let coords = vec2f(f32(id.x), f32(id.y)) / ${d.vec2f(1920, 1080)};

    // Sample from external texture
    let color = textureSampleBaseClampToEdge(
      videoTexture,
      videoSampler,
      coords
    );

    // Apply effect (e.g., grayscale)
    let gray = dot(color.rgb, vec3f(0.299, 0.587, 0.114));
    outputTexture[id.xy] = vec4f(gray, gray, gray, 1.0);
  }`;

// Render loop
function frame() {
  // Import video frame as external texture
  const externalTexture = device.importExternalTexture({
    source: video,
  });

  // Bind and dispatch shader
  // Note: Must complete GPU work in same frame
  // External textures are destroyed automatically as a microtask

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
```

**Important Characteristics of External Textures**:

1. **Zero-Copy Access**: External textures reference video memory directly without copying frame data to a GPU texture. This significantly improves performance for video processing.

2. **Automatic Destruction**: External textures are destroyed automatically as a microtask when JavaScript returns to the browser. You must complete all rendering using an external texture in the same callback where you create it.

3. **Requires New Bind Groups Each Frame**: Since external textures are recreated every frame, you must also create new bind groups containing them each frame.

4. **Special Shader Syntax**: External textures use `texture_external` type and `textureSampleBaseClampToEdge()` function in WGSL.

### Camera Input with MediaStream

Camera input follows the same pattern as video, but uses `getUserMedia()` to access the camera stream:

```typescript
async function setupCamera() {
  // Request camera access
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 60 },
    },
  });

  // Apply stream to video element
  const video = document.createElement("video");
  video.srcObject = stream;
  video.play();

  return video;
}

// Face detection / AR effects example
const video = await setupCamera();

function processFrame() {
  const externalTexture = device.importExternalTexture({
    source: video,
  });

  // Run face detection shader
  detectFaces(externalTexture);

  // Apply AR effects
  applyAREffects(externalTexture);

  requestAnimationFrame(processFrame);
}
```

### Use Cases for External Textures

**Video Conferencing**: Apply background blur, replacement, or virtual effects in real-time:

```typescript
const backgroundBlur = tgpu.fn([], d.void)
  .does`(@builtin(global_invocation_id) id: vec3u) {
    let coords = vec2f(id.xy) / resolution;

    // Sample person segmentation mask
    let mask = textureSampleBaseClampToEdge(segmentationMask, sampler, coords).r;

    // Sample video
    let color = textureSampleBaseClampToEdge(videoTexture, sampler, coords);

    // Sample blurred background
    let blurred = textureSampleBaseClampToEdge(blurredBg, sampler, coords);

    // Composite
    outputTexture[id.xy] = mix(blurred, color, mask);
  }`;
```

**Real-Time Computer Vision**: Process camera frames for object detection, pose estimation, or optical flow without expensive CPU transfers.

**Live Streaming Effects**: Apply color grading, filters, or artistic effects to live video streams with minimal latency.

**AR/VR Applications**: Process camera input for AR overlays, environment understanding, or mixed reality experiences.

## WebGPU Interoperability

### Accessing Raw GPUDevice

TypeGPU maintains a 1-to-1 mapping with WebGPU primitives, allowing seamless interoperability. Access the raw `GPUDevice` from the root object:

```typescript
const root = await tgpu.init();

// Access underlying WebGPU device
const device = root.unwrap().device;

// Now you can use vanilla WebGPU APIs
const rawBuffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const commandEncoder = device.createCommandEncoder();
// ... vanilla WebGPU commands ...
```

The `unwrap()` operation exposes TypeGPU's internal WebGPU resources. This is crucial when integrating with libraries that expect raw WebGPU objects, or when you need features not yet abstracted by TypeGPU.

### Mixing TypeGPU with Native WebGPU

TypeGPU's non-contagious design enables incremental adoption. You can use TypeGPU for some parts of your application while using vanilla WebGPU for others:

```typescript
// TypeGPU buffer
const typedBuffer = root
  .createBuffer(particleSchema, initialData)
  .$usage("storage")
  .$name("particles");

// Extract raw GPUBuffer
const rawBuffer = root.unwrap(typedBuffer);

// Use in vanilla WebGPU pipeline
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: rawBuffer }, // Raw GPUBuffer
    },
  ],
});
```

The unwrap operation is idempotent—calling `root.unwrap(typedBuffer)` multiple times returns the exact same `GPUBuffer` instance.

### Using TypeGPU Types in Vanilla WebGPU

Even if you're not using TypeGPU's shader features, you can leverage its type system for buffer management:

```typescript
import * as d from "typegpu/data";

// Define schema for buffer layout
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Calculate buffer size automatically
const bufferSize = d.sizeOf(Particle) * particleCount;

// Create vanilla WebGPU buffer with correct size
const buffer = device.createBuffer({
  size: bufferSize,
  usage: GPUBufferUsage.STORAGE,
});

// Wrap in TypeGPU for type-safe read/write
const typedBuffer = root.createBuffer(Particle, buffer);
await typedBuffer.write(particleData); // Type-checked!
```

This approach provides type safety and automatic memory layout handling while maintaining full control over WebGPU resource creation.

### Gradual Migration Strategy

When migrating existing WebGPU code to TypeGPU, start with the highest-value areas:

**Phase 1: Buffer Management**

```typescript
// Before: Manual buffer creation and data copying
const buffer = device.createBuffer({ size: 1024, usage: ... });
const data = new Float32Array([...]);
device.queue.writeBuffer(buffer, 0, data);

// After: TypeGPU with automatic layout
const buffer = root.createBuffer(schema, data).$usage('storage');
```

**Phase 2: Bind Group Layouts**

```typescript
// Before: Index-based bindings
const layout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
  ],
});

// After: Named, typed bindings
const layout = tgpu.bindGroupLayout({
  particles: { storage: particleSchema },
  forces: { storage: forceSchema },
});
```

**Phase 3: Shader Functions**

```typescript
// Before: String-based WGSL
const shaderCode = `
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    // shader code
  }
`;

// After: TGSL with type safety
const shader = tgpu.fn([]).does(() => {
  "use gpu";
  // shader code with TypeScript autocomplete
});
```

This incremental approach minimizes risk and allows measuring benefits at each stage.

## React Native Support

### react-native-wgpu Overview

TypeGPU works on React Native thanks to `react-native-wgpu`, which provides a WebGPU implementation for iOS and Android. This brings GPU compute and graphics capabilities to mobile devices with the same API as web browsers.

**Platform Support**:

- iOS (Metal backend)
- Android (Vulkan backend)
- Expo (with prebuild, not Expo Go)

The package name is `react-native-wgpu` (not `react-native-webgpu`), and it implements the WebGPU specification using native graphics APIs.

### Setup Requirements

Install required packages:

```bash
npm install react-native-wgpu typegpu
npm install --save-dev @webgpu/types unplugin-typegpu
```

Configure TypeScript in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "types": ["@webgpu/types"],
    "strict": true,
    "esModuleInterop": true
  }
}
```

### Babel Plugin Configuration

For TGSL support in React Native, configure Babel:

```javascript
// babel.config.js
module.exports = (api) => {
  api.cache(true);

  return {
    presets: ["babel-preset-expo"],
    plugins: [
      "unplugin-typegpu/babel", // Add TypeGPU plugin
    ],
  };
};
```

After adding the plugin, clear Metro cache:

```bash
npx expo start --clear
```

### TypeGPU on Mobile

React Native contexts require special handling for frame presentation:

```typescript
import { View } from 'react-native';
import { Canvas, useCanvasEffect } from 'react-native-wgpu';

export default function App() {
  const ref = useCanvasEffect(async () => {
    const context = ref.current!.getContext('webgpu')!;
    const device = await navigator.gpu.requestAdapter().then(a =>
      a!.requestDevice()
    );

    // Configure context
    context.configure({
      device,
      format: navigator.gpu.getPreferredCanvasFormat(),
    });

    // Render loop
    const frame = () => {
      const commandEncoder = device.createCommandEncoder();
      const textureView = context.getCurrentTexture().createView();

      // ... rendering commands ...

      device.queue.submit([commandEncoder.finish()]);

      // React Native specific: manual frame presentation
      context.present();

      requestAnimationFrame(frame);
    };

    requestAnimationFrame(frame);
  });

  return (
    <View style={{ flex: 1 }}>
      <Canvas ref={ref} style={{ flex: 1 }} />
    </View>
  );
}
```

### Platform Differences

**iOS Simulator**: Disable Metal validation before running. In Xcode, go to "Product > Scheme > Edit Scheme" and uncheck "Metal API Validation" under the "Run" section.

**Frame Presentation**: Unlike web browsers where presentation is automatic, React Native requires calling `context.present()` manually after submitting commands. This provides more control over rendering timing.

**Expo Compatibility**: React Native WebGPU doesn't work with Expo Go. Run `npx expo prebuild` to generate native projects, then:

```bash
# iOS
cd ios && pod install && cd ..
npx expo run:ios

# Android
npx expo run:android
```

### Integration with React Native Worklets

For advanced use cases, integrate with Reanimated Worklets to run GPU code on the UI thread:

```bash
npm install react-native-webgpu-worklets
```

This requires Reanimated 4.1.0+ and enables smooth 3D animations with GPU acceleration integrated with gesture handling.

## Generator CLI (tgpu-gen)

### WGSL to TypeGPU Conversion

The `tgpu-gen` CLI automates converting existing WGSL shader files into TypeGPU definitions. This is invaluable when migrating established WebGPU projects or integrating community WGSL shaders.

Install the generator:

```bash
npm install --save-dev tgpu-gen
```

### Command-Line Usage

Basic conversion of a single file:

```bash
npx tgpu-gen src/shaders/compute.wgsl
```

This generates `src/shaders/compute.ts` with TypeGPU equivalents of the WGSL code.

**Glob Pattern Support**:

```bash
# Convert all WGSL files in a directory
npx tgpu-gen "src/shaders/*.wgsl"

# Recursive conversion
npx tgpu-gen "src/**/*.wgsl" --output "src/**/*.ts"
```

When using `**` in the input pattern, also use it in the output path to preserve directory structure and avoid name conflicts.

### Generated Code Structure

Given this WGSL shader:

```wgsl
struct Particle {
  position: vec3f,
  velocity: vec3f,
  lifetime: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= arrayLength(&particles)) {
    return;
  }

  particles[idx].position += particles[idx].velocity * 0.016;
}
```

The generator produces:

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Schema generated from WGSL struct
export const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Buffer definition
export const particles = tgpu
  .slot<d.ArrayBuffer<typeof Particle>>()
  .$name("particles");

// Shader function
export const updateParticles = tgpu.fn([], d.void)
  .does`(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&particles)) {
      return;
    }

    particles[idx].position += particles[idx].velocity * 0.016;
  }`.$uses({ particles });
```

### Workflow Integration

**Build Pipeline Setup**:

Add to `package.json`:

```json
{
  "scripts": {
    "generate-shaders": "tgpu-gen \"src/shaders/**/*.wgsl\"",
    "prebuild": "npm run generate-shaders",
    "build": "vite build"
  }
}
```

Now shader generation runs automatically before builds.

**Watch Mode**:

For development, use watch mode to regenerate TypeGPU code when WGSL files change:

```bash
npx tgpu-gen "src/shaders/**/*.wgsl" --watch
```

This continuously monitors shader files and updates TypeGPU definitions, enabling a workflow where you edit WGSL and immediately see changes in your TypeScript application.

### Migration Best Practices

When using `tgpu-gen` for migration:

1. **Review Generated Code**: The generator produces reasonable output, but review for optimization opportunities or TypeGPU-specific improvements.

2. **Refactor for Modularity**: After generation, consider breaking monolithic shaders into smaller, reusable TypeGPU functions.

3. **Add Type Safety**: Enhance generated code with stronger TypeScript types and schema validations.

4. **Test Equivalence**: Verify that generated TypeGPU code produces identical results to original WGSL.

## tgpu.simulate()

### CPU Simulation for Testing

The `tgpu.simulate()` API enables running shader functions on the CPU, providing a powerful tool for testing and debugging GPU code without requiring GPU hardware or context.

While this feature is in the `~unstable` namespace, it's invaluable for:

- Unit testing shader logic
- Debugging complex algorithms
- Developing shaders on systems without GPU access
- Verifying shader behavior before GPU deployment

Basic simulation usage:

```typescript
const computeDistance = (a: number, b: number) => {
  "use gpu";
  return Math.sqrt(a * a + b * b);
};

// Run on CPU
const result = tgpu["~unstable"].simulate(computeDistance, [3, 4]);
console.log(result); // 5
```

### Testing Shaders with Simulation

Create comprehensive unit tests for GPU functions:

```typescript
import { test, expect } from "vitest";
import { simulate } from "typegpu/unstable";

const normalize2D = (x: number, y: number) => {
  "use gpu";
  const length = Math.sqrt(x * x + y * y);
  return { x: x / length, y: y / length };
};

test("normalize2D produces unit vectors", () => {
  const result = simulate(normalize2D, [3, 4]);

  expect(result.x).toBeCloseTo(0.6);
  expect(result.y).toBeCloseTo(0.8);

  const magnitude = Math.sqrt(result.x ** 2 + result.y ** 2);
  expect(magnitude).toBeCloseTo(1.0);
});

test("normalize2D handles edge cases", () => {
  const zero = simulate(normalize2D, [0, 0]);
  expect(zero.x).toBeNaN();
  expect(zero.y).toBeNaN();
});
```

### Verification Strategies

**Property-Based Testing**: Use simulation for property-based tests:

```typescript
import { fc, test } from "@fast-check/vitest";

const clampValue = (value: number, min: number, max: number) => {
  "use gpu";
  return Math.max(min, Math.min(max, value));
};

test("clamp always produces values in range", () => {
  fc.assert(
    fc.property(fc.float(), fc.float(), fc.float(), (value, a, b) => {
      const min = Math.min(a, b);
      const max = Math.max(a, b);
      const result = simulate(clampValue, [value, min, max]);

      return result >= min && result <= max;
    }),
  );
});
```

**Comparing CPU and GPU Results**: Verify that GPU execution matches CPU simulation:

```typescript
async function verifyGPUAgainstCPU(fn, inputs) {
  // Simulate on CPU
  const cpuResult = simulate(fn, inputs);

  // Execute on GPU
  const gpuResult = await executeOnGPU(fn, inputs);

  // Compare with tolerance for floating-point differences
  expect(cpuResult).toBeCloseTo(gpuResult, 5);
}
```

### Limitations

Simulation has important limitations:

- **No Buffer/Texture Access**: Shaders using GPU-exclusive features like buffers, textures, or atomic operations cannot be simulated.

- **Performance Differences**: CPU and GPU have different performance characteristics. Simulation is for correctness testing, not performance benchmarking.

- **Floating-Point Precision**: GPUs may handle floating-point operations differently than CPUs, leading to small numerical differences.

- **Undefined Behavior**: Some WGSL behaviors (like out-of-bounds access) may differ between simulation and GPU execution.

## Custom Extensions

### Building on TypeGPU

TypeGPU's modular architecture makes it an excellent foundation for building domain-specific libraries and custom extensions. You can create wrapper libraries that provide higher-level abstractions while maintaining full TypeGPU compatibility.

Example: Building a particle system library:

```typescript
// particle-system.ts
import tgpu from "typegpu";
import * as d from "typegpu/data";

export const ParticleSchema = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
  color: d.vec4f,
});

export class ParticleSystem {
  private root: TgpuRoot;
  private particleBuffer: TgpuBuffer;
  private updatePipeline: TgpuComputePipeline;

  constructor(root: TgpuRoot, maxParticles: number) {
    this.root = root;

    // Create particle buffer
    this.particleBuffer = root
      .createBuffer(d.arrayOf(ParticleSchema, maxParticles))
      .$usage("storage")
      .$name("particles");

    // Create update pipeline
    this.updatePipeline = this.createUpdatePipeline();
  }

  private createUpdatePipeline() {
    const updateShader = tgpu.fn([d.u32], d.void)
      .does`(@builtin(global_invocation_id) id: vec3u) {
        let idx = id.x;
        particles[idx].position += particles[idx].velocity * deltaTime;
        particles[idx].lifetime -= deltaTime;
      }`.$uses({ particles: this.particleBuffer });

    return this.root.makeComputePipeline(updateShader);
  }

  update(deltaTime: number) {
    // Update particles on GPU
    // ... implementation ...
  }

  getBuffer() {
    return this.particleBuffer;
  }
}

// Usage
const particleSystem = new ParticleSystem(root, 10000);
particleSystem.update(0.016);
```

### Creating Custom Data Types

Build specialized data types for your domain:

```typescript
// physics-types.ts
import * as d from "typegpu/data";

export const Quaternion = d.vec4f;

export const RigidBody = d.struct({
  position: d.vec3f,
  rotation: Quaternion,
  linearVelocity: d.vec3f,
  angularVelocity: d.vec3f,
  mass: d.f32,
  inverseMass: d.f32,
  inertiaTensor: d.mat3x3f,
});

export const Constraint = d.struct({
  bodyA: d.u32,
  bodyB: d.u32,
  localPivotA: d.vec3f,
  localPivotB: d.vec3f,
  compliance: d.f32,
});

// Helper functions
export function createRigidBody(mass: number): d.Infer<typeof RigidBody> {
  return {
    position: [0, 0, 0],
    rotation: [0, 0, 0, 1],
    linearVelocity: [0, 0, 0],
    angularVelocity: [0, 0, 0],
    mass,
    inverseMass: mass > 0 ? 1 / mass : 0,
    inertiaTensor: new Float32Array(9),
  };
}
```

### Community Packages

The TypeGPU ecosystem includes helper packages for common use cases:

**@typegpu/color**: Color manipulation utilities compatible with TypeGPU shaders.

**@typegpu/noise**: Procedural noise functions (Perlin, Simplex, etc.) for both CPU and GPU.

While the ecosystem is still growing, the modular design encourages community contributions. When building packages:

1. Export reusable schemas for data structures
2. Provide both TGSL and WGSL implementations
3. Include simulation-compatible functions for testing
4. Document resource requirements and usage patterns
5. Use semantic versioning for API stability

## Advanced Slot Patterns

### Slot Hierarchies

Slots provide configurable placeholders for values that can be set at pipeline creation or runtime. Advanced applications use nested slots for complex configuration hierarchies:

```typescript
// Define configuration hierarchy
const simulationConfig = tgpu.slot({
  physics: tgpu.slot({
    gravity: d.f32,
    damping: d.f32,
    timestep: d.f32,
  }),
  rendering: tgpu.slot({
    particleSize: d.f32,
    colorIntensity: d.f32,
  }),
});

// Use in shader
const updateParticle = tgpu.fn([d.u32], d.void).does`(idx: u32) {
    let gravity = ${simulationConfig.physics.gravity};
    let damping = ${simulationConfig.physics.damping};

    particles[idx].velocity.y += gravity * ${simulationConfig.physics.timestep};
    particles[idx].velocity *= damping;
  }`;

// Configure at pipeline creation
const pipeline = root
  .makeComputePipeline(updateParticle)
  .$with(simulationConfig, {
    physics: {
      gravity: -9.81,
      damping: 0.99,
      timestep: 0.016,
    },
    rendering: {
      particleSize: 5.0,
      colorIntensity: 1.0,
    },
  });
```

### Complex Binding Patterns

Slots enable sophisticated resource binding patterns:

```typescript
// Multiple material slots for instanced rendering
const materialSlot = tgpu.slot({
  albedo: d.vec3f,
  roughness: d.f32,
  metallic: d.f32,
  emissive: d.vec3f,
});

// Create multiple instances with different materials
const materials = [
  { albedo: [1, 0, 0], roughness: 0.8, metallic: 0.1, emissive: [0, 0, 0] },
  { albedo: [0, 1, 0], roughness: 0.2, metallic: 0.9, emissive: [0, 0, 0] },
  {
    albedo: [0.5, 0.5, 0.5],
    roughness: 0.5,
    metallic: 0.5,
    emissive: [1, 1, 1],
  },
];

const pipelines = materials.map((material) =>
  root.makeRenderPipeline(shader).$with(materialSlot, material),
);
```

### Dynamic Pipelines and Runtime Configuration

Slots enable hot-swapping shader behavior without recompiling pipelines:

```typescript
// Shader mode selection
const shaderMode = tgpu.slot(d.u32);

const conditionalShader = tgpu.fn([d.u32], d.vec4f).does`(idx: u32) -> vec4f {
    switch (${shaderMode}) {
      case 0u: {
        return normalVisualization(idx);
      }
      case 1u: {
        return depthVisualization(idx);
      }
      case 2u: {
        return lightingOnly(idx);
      }
      default: {
        return fullRendering(idx);
      }
    }
  }`;

// Change mode at runtime
pipeline.$with(shaderMode, 1); // Switch to depth visualization
```

This pattern is useful for debug visualizations, A/B testing different algorithms, or exposing shader parameters to end users.

## Best Practices

### When to Use Advanced Features

**Use Externals and $uses**:

- Building reusable shader libraries across projects
- Managing complex shader dependency graphs
- Publishing TypeGPU-based packages to npm

**Use Resolve API**:

- Debugging unexpected shader behavior
- Integrating with external profiling tools
- Pre-compiling shaders for production deployments

**Use External Textures**:

- Processing video or camera streams in real-time
- Building video conferencing or streaming applications
- Computer vision pipelines requiring minimal latency

**Use WebGPU Interoperability**:

- Migrating existing WebGPU codebases incrementally
- Integrating with libraries that expect raw WebGPU objects
- Accessing WebGPU features not yet abstracted by TypeGPU

**Use React Native**:

- Building cross-platform GPU applications
- Bringing GPU compute to mobile devices
- Creating mobile AR/VR experiences

**Use tgpu-gen**:

- Migrating established WebGPU projects to TypeGPU
- Integrating community WGSL shaders
- Maintaining WGSL as the source of truth during gradual adoption

**Use Simulation**:

- Unit testing shader logic
- Developing on systems without GPU access
- Verifying correctness before GPU deployment

**Use Slots**:

- Creating configurable, reusable shaders
- Building shader libraries with tuneable parameters
- Implementing runtime shader configuration UI

### Performance Considerations

**Minimize Resolve Overhead**: Cache resolved WGSL when possible. Resolving large dependency graphs has overhead—do it once during initialization, not in hot loops.

**External Texture Limitations**: Remember that external textures require new bind groups each frame. Minimize other dynamic bind group creation to reduce overhead.

**Slot Performance**: Slots add indirection. For performance-critical paths, consider baking values into specialized pipelines instead of using dynamic slot configuration.

**Interop Costs**: Unwrapping TypeGPU objects to raw WebGPU is essentially free (it's just an accessor), but repeatedly switching between TypeGPU and vanilla WebGPU patterns can make code harder to optimize.

**React Native Overhead**: Mobile GPUs have different performance characteristics than desktop. Profile on actual devices and be mindful of power consumption.

## Common Pitfalls

### Overcomplication

**Pitfall**: Using advanced features when simple solutions suffice.

```typescript
// Overcomplicated: Using slots for static configuration
const gravity = tgpu.slot(d.f32);
const pipeline = root.makeComputePipeline(shader).$with(gravity, -9.81);

// Better: Use constants directly
const shader = tgpu.fn([]).does(() => {
  "use gpu";
  const gravity = d.f32(-9.81);
  // ...
});
```

**Solution**: Choose the simplest approach that meets your requirements. Advanced features add value for complex scenarios but introduce unnecessary complexity for simple use cases.

### Interop Edge Cases

**Pitfall**: Forgetting that unwrapped resources lose type safety.

```typescript
const typedBuffer = root.createBuffer(schema, data).$usage("storage");
const rawBuffer = root.unwrap(typedBuffer);

// Error: rawBuffer has no .write() method
await rawBuffer.write([1, 2, 3]); // TypeError!

// Correct: Use typed buffer for type-safe operations
await typedBuffer.write([1, 2, 3]);
```

**Solution**: Use TypeGPU wrappers for operations requiring type safety. Only unwrap when interfacing with code that explicitly needs raw WebGPU objects.

### External Texture Timing

**Pitfall**: Attempting to use external textures across multiple frames.

```typescript
// Wrong: External texture destroyed before use
const externalTexture = device.importExternalTexture({ source: video });

requestAnimationFrame(() => {
  // Error: External texture already destroyed!
  renderWithTexture(externalTexture);
});

// Correct: Create and use in same frame
requestAnimationFrame(() => {
  const externalTexture = device.importExternalTexture({ source: video });
  renderWithTexture(externalTexture);
});
```

**Solution**: Always create and consume external textures within the same callback. They're destroyed automatically when JavaScript returns to the browser.

### React Native Frame Presentation

**Pitfall**: Forgetting to call `context.present()` in React Native.

```typescript
// Wrong: Frames won't display
device.queue.submit([commandEncoder.finish()]);
// Missing: context.present()

// Correct: Manual presentation required
device.queue.submit([commandEncoder.finish()]);
context.present(); // Essential for React Native!
```

**Solution**: Unlike web browsers where presentation is automatic, React Native requires explicitly calling `context.present()` after submitting work.

### Generator Overwrites

**Pitfall**: Manually editing generated TypeGPU files, then losing changes when regenerating.

```typescript
// You edit src/shaders/compute.ts (generated file)
// Later, regenerate shaders
npx tgpu-gen "src/shaders/**/*.wgsl" // Overwrites your changes!
```

**Solution**: Never edit generated files directly. Instead, import generated code and build abstractions around it:

```typescript
// src/shaders/compute.ts (generated - don't edit!)
export const updateParticles = /* ... generated code ... */;

// src/shaders/particle-system.ts (custom wrapper)
import { updateParticles } from './compute';

export class ParticleSystem {
  // Your custom logic using generated shaders
}
```

---

## Conclusion

Advanced TypeGPU patterns unlock sophisticated GPU programming capabilities while maintaining type safety and developer experience. By mastering externals and dependency management, leveraging the Resolve API for debugging and integration, efficiently processing video with external textures, seamlessly interoperating with vanilla WebGPU, deploying to mobile platforms via React Native, automating migrations with the Generator CLI, testing on CPU with simulation, building custom extensions, and creating dynamic pipelines with slots, you can build production-ready GPU applications that are maintainable, performant, and portable.

The key to successful advanced TypeGPU usage is knowing when to apply these patterns. Start with simple approaches and introduce advanced features as complexity demands. TypeGPU's modular, non-contagious design ensures you're never locked in—you can always eject to vanilla WebGPU for specific optimizations or unsupported features.

As the TypeGPU ecosystem grows, expect more community packages, better tooling, and expanded capabilities. The patterns covered in this guide provide a foundation for exploring these advances and building the next generation of GPU-accelerated web and mobile applications.

---

## Sources

- [TypeGPU Official Documentation](https://docs.swmansion.com/TypeGPU/)
- [GitHub - software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- [WebGPU Interoperability | TypeGPU](https://docs.swmansion.com/TypeGPU/integration/webgpu-interoperability/)
- [Functions | TypeGPU](https://docs.swmansion.com/TypeGPU/fundamentals/functions/)
- [tgpu API Reference | TypeGPU](https://docs.swmansion.com/TypeGPU/api/typegpu/variables/tgpu/)
- [React Native | TypeGPU](https://docs.swmansion.com/TypeGPU/integration/react-native/)
- [Generator CLI | TypeGPU](https://docs.swmansion.com/TypeGPU/tooling/tgpu-gen/)
- [Build Plugin | TypeGPU](https://docs.swmansion.com/TypeGPU/tooling/unplugin-typegpu/)
- [GPUDevice: importExternalTexture() - MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture)
- [WebGPU Using Video Efficiently](https://webgpufundamentals.org/webgpu/lessons/webgpu-textures-external-video.html)
- [GitHub - wcandillon/react-native-webgpu](https://github.com/wcandillon/react-native-webgpu)
- [tgpu-gen - npm](https://www.npmjs.com/package/tgpu-gen)
