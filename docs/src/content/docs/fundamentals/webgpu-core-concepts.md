---
title: WebGPU Core Concepts
sidebar:
  order: 10
---

## Overview

WebGPU is a modern graphics and compute API for the web that provides low-level, high-performance access to GPU hardware. Designed from the ground up to align with contemporary GPU architecture, WebGPU maps efficiently to native APIs including Vulkan (cross-platform), Metal (Apple), and Direct3D 12 (Windows).

As of 2025, WebGPU is supported across all major browsers: Chrome, Edge, Firefox 141+, and Safari 26+. This broad support makes it a viable choice for production web applications requiring GPU acceleration.

WebGPU enables high-fidelity 3D rendering with advanced techniques like physically-based rendering and complex post-processing pipelines. It treats general-purpose GPU computation (GPGPU) as a first-class feature, making it suitable for machine learning inference, scientific simulations, data visualization, and video processing.

## GPUAdapter: Hardware Abstraction

The GPUAdapter represents physical GPU hardware and serves as the entry point for capability discovery. It abstracts the underlying graphics hardware and driver while exposing machine capabilities through a privacy-conscious binning mechanism.

```typescript
const adapter = await navigator.gpu.requestAdapter({
  powerPreference: "high-performance",
  forceFallbackAdapter: false,
});

if (!adapter) {
  throw new Error("WebGPU adapter not available");
}
```

The `powerPreference` option specifies whether to prefer low-power (integrated GPU) or high-performance (discrete GPU) hardware.

### Querying Capabilities

Adapters expose capabilities through two interfaces:

```typescript
// Feature enumeration - optional capabilities as strings
console.log("Features:", Array.from(adapter.features));

if (adapter.features.has("texture-compression-bc")) {
  // BC compressed textures supported
}

// Capability limits - numeric constraints
console.log("Max texture size:", adapter.limits.maxTextureDimension2D);
console.log("Max buffer size:", adapter.limits.maxBufferSize);
```

## GPUDevice: Logical Connection

The GPUDevice is a logical connection to the GPU through which almost all WebGPU operations are performed. While the adapter represents physical hardware, the device provides an isolated, secure interface for creating resources and submitting work.

```typescript
const device = await adapter.requestDevice({
  requiredFeatures: ["texture-compression-bc"],
  requiredLimits: {
    maxStorageBufferBindingSize: 512 * 1024 * 1024,
  },
});
```

If any requested feature is unavailable or any limit exceeds the adapter's maximum, the Promise rejects. Check adapter capabilities before requesting.

The device manages all GPU resources:
- Buffers (GPU-accessible memory)
- Textures (image data)
- Shader modules (compiled WGSL)
- Pipelines (complete rendering/compute state)
- Bind groups (resource binding configuration)

### Device Lost Handling

A device can become "lost" due to GPU driver crashes, system sleep/wake cycles, or hardware removal. Handle this by monitoring the `device.lost` promise and setting up error handlers immediately after device creation:

```typescript
const device = await adapter.requestDevice();

device.lost.then((info) => {
  console.error(`Device lost: ${info.message}`);
  if (info.reason !== "destroyed") {
    initializeWebGPU(); // Attempt recovery
  }
});

device.addEventListener("uncapturederror", (event) => {
  console.error("Uncaptured WebGPU error:", event.error);
});
```

## GPUQueue: Command Execution

The GPUQueue controls submission and execution of GPU commands. Each device has a default queue accessible via `device.queue`.

### Submitting Commands

```typescript
const commandEncoder = device.createCommandEncoder();
// ... encode commands ...
const commandBuffer = commandEncoder.finish();

device.queue.submit([commandBuffer]);
```

Multiple command buffers in a single submit execute in sequence.

### Direct Data Writes

For small to medium updates, use direct write methods:

```typescript
// Write to buffer
const uniformData = new Float32Array([1.0, 0.0, 0.0, 1.0]);
device.queue.writeBuffer(uniformBuffer, 0, uniformData);

// Write to texture
device.queue.writeTexture(
  { texture: gpuTexture },
  imageData,
  { bytesPerRow: 256 * 4 },
  { width: 256, height: 256 }
);
```

### Synchronization

Commands execute asynchronously. Use `onSubmittedWorkDone()` when you need to wait:

```typescript
await device.queue.onSubmittedWorkDone();
console.log("All GPU work completed");
```

## Initialization Flow

A complete, production-ready initialization:

```typescript
async function initializeWebGPU(): Promise<{
  adapter: GPUAdapter;
  device: GPUDevice;
  format: GPUTextureFormat;
}> {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new Error("No WebGPU adapter found");
  }

  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has("texture-compression-bc")) {
    requiredFeatures.push("texture-compression-bc");
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  device.lost.then((info) => {
    console.error(`Device lost: ${info.message}`);
    if (info.reason !== "destroyed") {
      initializeWebGPU();
    }
  });

  device.addEventListener("uncapturederror", (event) => {
    console.error("Uncaptured error:", event.error);
  });

  const format = navigator.gpu.getPreferredCanvasFormat();

  return { adapter, device, format };
}
```

## Stateless Pipeline Architecture

WebGPU uses an explicit, immutable pipeline object model. Unlike WebGL's state machine where you configure global state incrementally, WebGPU pipelines encapsulate complete GPU state at creation time.

```typescript
const pipeline = device.createRenderPipeline({
  layout: pipelineLayout,
  vertex: {
    module: shaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout],
  },
  fragment: {
    module: shaderModule,
    entryPoint: "fragmentMain",
    targets: [{ format: "bgra8unorm" }],
  },
  primitive: { topology: "triangle-list" },
});

// In render pass
passEncoder.setPipeline(pipeline);
passEncoder.setVertexBuffer(0, vertexBuffer);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.drawIndexed(36);
```

This architecture provides:
- **Upfront validation**: State validated once at creation, not per-draw
- **Parallelism**: Pipelines can be created on multiple threads
- **Predictability**: No hidden state; each pipeline is self-contained
- **Performance**: Drivers optimize aggressively with complete state visibility

## Features and Limits

### Features

Optional capabilities checked via `adapter.features`:

```typescript
const featureChecks = {
  textureCompression: {
    bc: adapter.features.has("texture-compression-bc"),
    etc2: adapter.features.has("texture-compression-etc2"),
    astc: adapter.features.has("texture-compression-astc"),
  },
  shaderFeatures: {
    float16: adapter.features.has("shader-f16"),
  },
};
```

Always request features explicitly in `requestDevice()`:

```typescript
async function createDeviceWithFeatures(adapter: GPUAdapter) {
  const desired: GPUFeatureName[] = ["texture-compression-bc", "shader-f16"];
  const available = desired.filter((f) => adapter.features.has(f));

  return adapter.requestDevice({
    requiredFeatures: available,
  });
}
```

### Limits

Numeric constraints queried via `adapter.limits`:

```typescript
const limits = adapter.limits;
console.log({
  maxTextureDimension2D: limits.maxTextureDimension2D,
  maxBufferSize: limits.maxBufferSize,
  maxBindGroups: limits.maxBindGroups,
  maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
});
```

Request higher limits only up to adapter maximums:

```typescript
const requestedLimit = 1024 * 1024 * 1024;
const actualLimit = Math.min(requestedLimit, adapter.limits.maxStorageBufferBindingSize);

const device = await adapter.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: actualLimit },
});
```

## Resource Lifecycle

### Creation

Resources are created through device methods with descriptor objects:

```typescript
const vertexBuffer = device.createBuffer({
  size: 1024,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  mappedAtCreation: false,
});

const texture = device.createTexture({
  size: { width: 512, height: 512 },
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
```

### Usage Flags

The `usage` parameter declares how resources will be used. WebGPU validates actual usage matches declared usage.

**Buffer Usage Flags**:
- `COPY_SRC` / `COPY_DST`: Copy operations
- `MAP_READ` / `MAP_WRITE`: CPU mapping
- `VERTEX` / `INDEX`: Geometry data
- `UNIFORM`: Read-only shader constants
- `STORAGE`: Read-write shader access

**Texture Usage Flags**:
- `COPY_SRC` / `COPY_DST`: Copy operations
- `TEXTURE_BINDING`: Shader sampling
- `STORAGE_BINDING`: Shader storage access
- `RENDER_ATTACHMENT`: Render target

Combine flags with bitwise OR: `GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST`

### Destruction

Explicitly destroy large resources when done:

```typescript
buffer.destroy();
texture.destroy();
```

For loops creating many resources, explicit destruction prevents GPU memory accumulation:

```typescript
// Explicit cleanup prevents memory buildup
const buffer = device.createBuffer({ size: 1024 * 1024, usage: GPUBufferUsage.STORAGE });
// Use buffer...
buffer.destroy();
```

### Error Scopes

Use error scopes for expected failure points:

```typescript
device.pushErrorScope("validation");
const buffer = device.createBuffer({ /* potentially invalid */ });
const error = await device.popErrorScope();
if (error) {
  console.warn("Buffer creation failed:", error.message);
}
```

## GPU Process Architecture

WebGPU runs in a dedicated GPU process separate from the content process where JavaScript executes. This architecture has important implications:

- **Object Handles**: JavaScript objects like GPUDevice and GPUBuffer are lightweight proxies to objects in the GPU process
- **Asynchronous Operations**: Cross-process communication makes most operations inherently asynchronous
- **Security Boundary**: The GPU process sandbox contains potential exploits
- **Privacy Protection**: Adapter capability binning limits fingerprinting to 32 unique configurations

---

This documentation covers the foundational concepts for working effectively with WebGPU. Understanding adapters, devices, queues, stateless pipelines, and resource lifecycle equips you to build high-performance GPU-accelerated web applications.
