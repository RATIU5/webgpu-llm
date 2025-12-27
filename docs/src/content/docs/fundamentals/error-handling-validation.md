---
title: Error Handling and Validation
sidebar:
  order: 70
---

## Overview

WebGPU uses asynchronous error reporting and a stack-based error scope system, unlike WebGL's synchronous `getError()` approach. All errors are detected asynchronously in a GPU process, avoiding synchronous validation overhead.

When a WebGPU operation fails, the returned object is internally marked as invalid. Operations using that invalid object also become invalid—errors propagate contagiously. No synchronous exceptions are thrown; objects appear valid to JavaScript but fail during command submission. Developers must actively capture errors using error scopes.

## Error Types

### GPUValidationError

Programming mistakes that violate API constraints. Common causes:
- Invalid resource descriptors
- Binding mismatches (bind group layouts not matching pipeline layouts)
- Usage flag violations
- Shader compilation failures
- Using destroyed resources

```javascript
// Validation error: missing COPY_DST flag for writeBuffer
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.UNIFORM,
});
device.queue.writeBuffer(buffer, 0, data); // Fails
```

### GPUOutOfMemoryError

Resource exhaustion despite valid API usage:
- Insufficient GPU memory
- Memory fragmentation
- Exceeding implementation limits

```javascript
// May fail on devices with limited VRAM
const hugeTexture = device.createTexture({
  size: { width: 16384, height: 16384 },
  format: "rgba32float",
  usage: GPUTextureUsage.STORAGE_BINDING,
});
```

### GPUInternalError

Implementation failures despite valid usage—driver crashes, hardware failures, or browser bugs. These are rare and cannot be prevented, but should be handled gracefully.

### Device Lost

Catastrophic failure where the GPU becomes unavailable:
- Hardware disconnection
- Driver crash (TDR)
- Application calls `device.destroy()`
- System resource exhaustion

Device lost requires complete recovery: recreate device and all resources.

## Error Scopes

Error scopes capture errors hermetically, preventing leakage between unrelated code.

### Basic Usage

```javascript
device.pushErrorScope("validation");
const buffer = device.createBuffer(descriptor);
const error = await device.popErrorScope();

if (error) {
  console.error("Buffer creation failed:", error.message);
}
```

Filter options: `'validation'`, `'out-of-memory'`, `'internal'`

### Multiple Error Types

```javascript
async function createBufferWithErrorHandling(device, descriptor) {
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  const buffer = device.createBuffer(descriptor);

  // Pop in reverse order (LIFO)
  const outOfMemoryError = await device.popErrorScope();
  const validationError = await device.popErrorScope();

  if (validationError) {
    console.error("Validation:", validationError.message);
    return null;
  }
  if (outOfMemoryError) {
    console.error("Out of memory:", outOfMemoryError.message);
    return null;
  }

  return buffer;
}
```

### Nested Scopes

Inner scopes capture errors first; uncaptured errors propagate to outer scopes:

```javascript
async function createPipeline(device, shaderCode, pipelineDescriptor) {
  device.pushErrorScope("validation"); // Outer: pipeline

  device.pushErrorScope("validation"); // Inner: shader
  const shaderModule = device.createShaderModule({ code: shaderCode });
  const shaderError = await device.popErrorScope();

  if (shaderError) {
    await device.popErrorScope(); // Clean up outer scope
    throw new Error(`Shader compilation failed: ${shaderError.message}`);
  }

  const pipeline = device.createRenderPipeline({
    ...pipelineDescriptor,
    vertex: { module: shaderModule, entryPoint: "vertex_main" },
    fragment: { module: shaderModule, entryPoint: "fragment_main", targets: [{ format: "bgra8unorm" }] },
  });

  const pipelineError = await device.popErrorScope();
  if (pipelineError) {
    throw new Error(`Pipeline creation failed: ${pipelineError.message}`);
  }

  return pipeline;
}
```

`popErrorScope()` returns only the first error in the scope.

## Uncaptured Errors

Errors not captured by any scope trigger the `uncapturederror` event:

```javascript
device.addEventListener("uncapturederror", (event) => {
  console.error("Uncaptured GPU error:", event.error.message);

  if (event.error instanceof GPUValidationError) {
    console.error("Check API usage");
  } else if (event.error instanceof GPUOutOfMemoryError) {
    console.error("Reduce resource usage");
  } else if (event.error instanceof GPUInternalError) {
    console.error("Possible driver or hardware issue");
  }
});
```

Errors become uncaptured when:
- No error scope is active
- Active scopes don't match the error type
- Scopes are popped before the error occurs

## Device Lost Handling

Monitor `device.lost` immediately after device creation:

```javascript
const device = await adapter.requestDevice();

device.lost.then((info) => {
  console.error(`Device lost: ${info.reason} - ${info.message}`);

  if (info.reason !== "destroyed") {
    attemptRecovery();
  }
});
```

The `reason` is either `'destroyed'` (explicit) or `'unknown'` (system failure).

### Recovery Pattern

```javascript
class WebGPUApp {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.resources = new Map();
  }

  async initialize() {
    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) throw new Error("No adapter");
    await this.initializeDevice();
  }

  async initializeDevice() {
    this.device = await this.adapter.requestDevice();

    this.device.lost.then((info) => {
      console.warn("Device lost:", info);
      if (info.reason !== "destroyed") {
        this.handleDeviceLost();
      }
    });

    this.device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured:", event.error);
    });
  }

  async handleDeviceLost() {
    this.resources.clear();

    try {
      await this.initializeDevice();
      await this.recreateResources();
      this.resume();
    } catch (error) {
      console.error("Recovery failed:", error);
    }
  }

  async recreateResources() {
    // Re-create GPU resources from stored descriptors
  }

  resume() {
    // Resume rendering
  }
}
```

## Debugging with Labels

Labels appear in error messages, making debugging easier:

```javascript
const buffer = device.createBuffer({
  label: "Particle Position Buffer",
  size: 1024,
  usage: GPUBufferUsage.STORAGE,
});

const texture = device.createTexture({
  label: "Shadow Map Texture",
  size: { width: 2048, height: 2048 },
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

// Error messages include labels:
// "Buffer 'Particle Position Buffer': usage must be non-zero"
```

Label all resources during development.

## Common Validation Errors

### Buffer size must be multiple of 4

```javascript
// Wrong
device.createBuffer({ size: 255, usage: GPUBufferUsage.STORAGE });

// Correct
device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
```

### Buffer usage must be non-zero

```javascript
// Wrong
device.createBuffer({ size: 256, usage: 0 });

// Correct
device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
```

### Bind group entry doesn't match layout

```javascript
// Wrong: binding number mismatch
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{ binding: 1, resource: { buffer } }], // Layout expects binding 0
});

// Correct
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{ binding: 0, resource: { buffer } }],
});
```

## TypeGPU Error Prevention

TypeGPU catches many errors at compile time through TypeScript's type system:

```typescript
import * as d from "typegpu/data";

const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  mass: d.f32,
});

// TypeScript error: missing z component
const invalidData = {
  position: [1, 2],  // Error!
  velocity: [0, 0, 0],
  mass: 1.0,
};

// Correct
const validData: (typeof Particle)["~repr"] = {
  position: [1, 2, 3],
  velocity: [0, 0, 0],
  mass: 1.0,
};
```

TypeGPU eliminates:
- Binding type mismatches
- Data layout errors
- Usage flag errors (set automatically)
- Format mismatches

## Complete Error Handling Example

```javascript
class GPUResourceManager {
  constructor(device) {
    this.device = device;
    this.setupErrorHandling();
  }

  setupErrorHandling() {
    this.device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured:", event.error.message);
    });

    this.device.lost.then((info) => {
      console.error("Device lost:", info.reason, info.message);
    });
  }

  async createBuffer(descriptor) {
    this.device.pushErrorScope("validation");
    this.device.pushErrorScope("out-of-memory");

    const buffer = this.device.createBuffer(descriptor);

    const memError = await this.device.popErrorScope();
    const valError = await this.device.popErrorScope();

    if (valError) throw new Error(`Invalid descriptor: ${valError.message}`);
    if (memError) throw new Error(`Out of memory: ${memError.message}`);

    return buffer;
  }

  async createTexture(descriptor) {
    this.device.pushErrorScope("validation");
    this.device.pushErrorScope("out-of-memory");

    const texture = this.device.createTexture(descriptor);

    const memError = await this.device.popErrorScope();
    const valError = await this.device.popErrorScope();

    if (valError) {
      console.error("Texture validation:", valError.message);
      return null;
    }
    if (memError) {
      console.warn("Texture too large, trying half resolution");
      return this.createTexture({
        ...descriptor,
        size: {
          width: Math.floor(descriptor.size.width / 2),
          height: Math.floor(descriptor.size.height / 2),
        },
      });
    }

    return texture;
  }

  async debugShader(shaderCode) {
    this.device.pushErrorScope("validation");

    const module = this.device.createShaderModule({
      label: "Debug Shader",
      code: shaderCode,
    });

    const error = await this.device.popErrorScope();

    if (error) {
      console.error("Shader compilation failed:");
      console.error(error.message);

      const lineMatch = error.message.match(/line (\d+)/);
      if (lineMatch) {
        const lineNum = parseInt(lineMatch[1]);
        const lines = shaderCode.split("\n");
        console.error(`Error at line ${lineNum}:`, lines[lineNum - 1]);
      }
      return null;
    }

    return module;
  }
}
```

---

WebGPU's asynchronous error model trades synchronous simplicity for performance and composability. Use error scopes during development to catch validation and out-of-memory errors, implement a global uncaptured error handler, monitor `device.lost` for catastrophic failures, and label all resources for clear error messages. With proper error handling, WebGPU applications can recover from failures and provide helpful debugging information.
