---
title: Error Handling and Validation
sidebar:
  order: 70
---

## Overview: WebGPU's Error Model

WebGPU introduces a fundamentally different approach to error handling compared to WebGL, driven by the architectural requirements of modern multi-process browsers and the performance demands of GPU-accelerated applications. Understanding this error model is crucial for building robust WebGPU applications.

### Why WebGPU Differs from WebGL

WebGL's traditional error handling relied on synchronous `getError()` calls, which required immediate round-trips to the GPU to check for errors. This approach had several critical problems:

1. **Performance overhead**: Each `getError()` call forced synchronization between CPU and GPU, creating pipeline stalls
2. **Poor composition**: Errors could leak between unrelated code sections, libraries, and browser extensions
3. **Scalability issues**: Modern GPU applications make tens of thousands of API calls per frame—validating each synchronously would be prohibitively expensive

WebGPU solves these problems through **asynchronous error reporting** and a **stack-based error scope system**. All errors are detected asynchronously in a remote GPU process, avoiding the need for synchronous validation in the content process. This architecture enables WebGPU to forward API calls directly to the GPU process without duplicate validation overhead.

### The Contagious Error Model

WebGPU implements what the specification calls an "error monad" or "contagious internal nullability." When a WebGPU operation fails:

- The returned object is internally marked as "invalid" in the GPU process
- Operations using that invalid object also become invalid—the error propagates contagiously
- No synchronous exceptions are thrown to JavaScript
- Objects appear valid to JavaScript code but fail silently during command submission

This design maintains security in sandboxed GPU processes while avoiding performance-killing synchronous round-trips. It means developers must actively capture errors using error scopes rather than relying on exceptions.

## WebGPU Error Types

WebGPU defines three distinct error types that represent different failure categories:

### GPUValidationError

Validation errors occur when your application violates WebGPU API constraints. These represent programming mistakes rather than system failures and should be caught during development. Common causes include:

- **Invalid resource descriptors**: Creating buffers or textures with incompatible parameters
- **Binding mismatches**: Bind group layouts not matching pipeline layouts
- **Usage flag violations**: Attempting operations on resources without appropriate usage flags
- **Shader compilation failures**: WGSL syntax errors or type mismatches
- **Resource compatibility issues**: Using destroyed resources or resources from different devices

Example validation error:

```javascript
// This will generate a validation error - missing required usage flag
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.UNIFORM, // Missing GPUBufferUsage.COPY_DST for writing
});

// Attempting to write will fail validation
device.queue.writeBuffer(buffer, 0, data);
```

### GPUOutOfMemoryError

Out-of-memory errors occur when the GPU cannot allocate sufficient memory for the requested operation. These are resource exhaustion errors that can happen even with valid API usage:

- **Insufficient GPU memory**: Creating textures or buffers larger than available VRAM
- **Memory fragmentation**: Unable to find contiguous memory blocks
- **System resource limits**: Exceeding implementation-defined limits

Example scenario:

```javascript
// May fail with out-of-memory error on devices with limited VRAM
const hugeTexture = device.createTexture({
  size: { width: 16384, height: 16384, depthOrArrayLayers: 1 },
  format: "rgba32float",
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
});
```

### GPUInternalError

Internal errors represent implementation failures that occur despite valid API usage. These indicate problems in the WebGPU implementation, GPU driver, or hardware:

- **Driver crashes**: GPU driver encounters an unexpected condition
- **Hardware failures**: Physical GPU errors or instability
- **Implementation bugs**: Errors in the browser's WebGPU implementation
- **Unrecoverable system errors**: OS-level GPU failures

Internal errors are rare in production but may occur due to driver bugs or hardware issues. Your application should handle them gracefully but cannot prevent them.

### Device Lost

Device lost is a catastrophic error condition where the GPU becomes unavailable. This is not technically a GPUError type but represents the most severe failure mode:

- **Hardware disconnection**: GPU physically removed (external GPUs)
- **Driver crash**: GPU driver stops responding (TDR - Timeout Detection and Recovery)
- **Explicit destruction**: Application calls `device.destroy()`
- **System resource exhaustion**: OS reclaims GPU resources

Device lost requires complete recovery including recreating the device and all resources.

## Error Scope API

Error scopes provide the primary mechanism for capturing and handling errors in WebGPU. Each `GPUDevice` maintains a stack of error scopes that hermetically capture errors, preventing them from leaking between unrelated code.

### Basic Error Scope Usage

The error scope API consists of two methods:

```javascript
device.pushErrorScope(filter); // Push a new scope onto the stack
const error = await device.popErrorScope(); // Pop scope and get captured error
```

The `filter` parameter specifies which error types to capture:

- `'validation'` - Captures `GPUValidationError` only
- `'out-of-memory'` - Captures `GPUOutOfMemoryError` only
- `'internal'` - Captures `GPUInternalError` only

### Complete Error Scope Example

```javascript
async function createBufferWithErrorHandling(device, descriptor) {
  // Push error scope before the operation
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  const buffer = device.createBuffer(descriptor);

  // Pop scopes in reverse order (LIFO)
  const outOfMemoryError = await device.popErrorScope();
  const validationError = await device.popErrorScope();

  if (validationError) {
    console.error("Validation error creating buffer:", validationError.message);
    return null;
  }

  if (outOfMemoryError) {
    console.error("Out of memory creating buffer:", outOfMemoryError.message);
    // Try creating a smaller buffer
    return createBufferWithErrorHandling(device, {
      ...descriptor,
      size: descriptor.size / 2,
    });
  }

  return buffer;
}

// Usage
const buffer = await createBufferWithErrorHandling(device, {
  size: 1024 * 1024 * 100, // 100MB
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
```

### Filtering Error Types

Each error scope captures only the specified error type. This allows fine-grained control over error handling:

```javascript
// Capture only validation errors
device.pushErrorScope("validation");
const texture = device.createTexture(descriptor);
const error = await device.popErrorScope();

if (error) {
  console.error("Invalid texture descriptor:", error.message);
}
```

### Nesting Error Scopes

Error scopes can be nested to handle errors at different granularities. Inner scopes capture errors first; uncaptured errors propagate to outer scopes:

```javascript
async function createRenderPipelineWithResources(device) {
  // Outer scope for the entire operation
  device.pushErrorScope("validation");

  // Inner scope for shader compilation
  device.pushErrorScope("validation");
  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });
  const shaderError = await device.popErrorScope();

  if (shaderError) {
    console.error("Shader compilation failed:", shaderError.message);
    throw new Error("Cannot create pipeline with invalid shader");
  }

  // Continue with pipeline creation
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vertex_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fragment_main",
      targets: [{ format: "bgra8unorm" }],
    },
  });

  const pipelineError = await device.popErrorScope();

  if (pipelineError) {
    console.error("Pipeline creation failed:", pipelineError.message);
    throw new Error("Invalid pipeline configuration");
  }

  return pipeline;
}
```

### Error Scope Stack Behavior

The error scope stack follows Last-In-First-Out (LIFO) semantics:

```javascript
device.pushErrorScope("validation"); // Scope A
device.pushErrorScope("out-of-memory"); // Scope B

// Some operations...

await device.popErrorScope(); // Pops scope B (out-of-memory)
await device.popErrorScope(); // Pops scope A (validation)
```

Errors are captured by the innermost (most recently pushed) matching scope. If no scope matches the error type, it propagates to the `uncapturederror` event.

## Asynchronous Error Handling

WebGPU's asynchronous error model requires understanding how errors propagate through the device timeline and how to capture them with promises.

### Promise-Based Error Capture

Error scopes return promises that resolve to either a `GPUError` or `null`:

```javascript
device.pushErrorScope("validation");

// Multiple operations can occur before popping
const buffer1 = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.UNIFORM,
});
const buffer2 = device.createBuffer({
  size: 512,
  usage: GPUBufferUsage.STORAGE,
});

// Pop returns a promise
const error = await device.popErrorScope();

if (error !== null) {
  // Error is the FIRST error that occurred in the scope
  console.error("First error in scope:", error.message);
}
```

**Important**: `popErrorScope()` returns only the **first** error that occurred within the scope. Subsequent errors are not captured if an earlier error exists.

### Error Handling Patterns

#### Pattern 1: Immediate Error Checking

```javascript
async function createAndValidateBuffer(device, size) {
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  const buffer = device.createBuffer({
    size: size,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  const memoryError = await device.popErrorScope();
  const validationError = await device.popErrorScope();

  if (validationError || memoryError) {
    const errorMsg = validationError?.message || memoryError?.message;
    throw new Error(`Buffer creation failed: ${errorMsg}`);
  }

  return buffer;
}
```

#### Pattern 2: Batch Operation Error Handling

```javascript
async function createResourceBatch(device, resourceDescriptors) {
  device.pushErrorScope("validation");

  const resources = resourceDescriptors.map((desc) => {
    if (desc.type === "buffer") {
      return device.createBuffer(desc.config);
    } else if (desc.type === "texture") {
      return device.createTexture(desc.config);
    }
  });

  const error = await device.popErrorScope();

  if (error) {
    console.error("Batch resource creation failed:", error.message);
    // First error in batch - subsequent resources may be invalid
    return null;
  }

  return resources;
}
```

#### Pattern 3: Try-Catch Style with Error Scopes

```javascript
async function withErrorScope(device, filter, operation) {
  device.pushErrorScope(filter);
  const result = await operation();
  const error = await device.popErrorScope();

  if (error) {
    throw error;
  }

  return result;
}

// Usage
try {
  const pipeline = await withErrorScope(device, "validation", async () => {
    return device.createRenderPipeline(pipelineDescriptor);
  });
} catch (error) {
  console.error("Pipeline creation failed:", error.message);
}
```

### Complete Async Error Handling Example

```javascript
class GPUResourceManager {
  constructor(device) {
    this.device = device;
  }

  async createBuffer(descriptor) {
    this.device.pushErrorScope("validation");
    this.device.pushErrorScope("out-of-memory");

    const buffer = this.device.createBuffer(descriptor);

    const outOfMemoryError = await this.device.popErrorScope();
    const validationError = await this.device.popErrorScope();

    if (validationError) {
      throw new Error(`Invalid buffer descriptor: ${validationError.message}`);
    }

    if (outOfMemoryError) {
      throw new Error(`Insufficient GPU memory: ${outOfMemoryError.message}`);
    }

    return buffer;
  }

  async createTexture(descriptor) {
    this.device.pushErrorScope("validation");
    this.device.pushErrorScope("out-of-memory");

    const texture = this.device.createTexture(descriptor);

    const outOfMemoryError = await this.device.popErrorScope();
    const validationError = await this.device.popErrorScope();

    if (validationError) {
      console.error("Texture validation error:", validationError.message);
      return null;
    }

    if (outOfMemoryError) {
      console.warn("Texture too large, attempting lower resolution...");
      // Fallback to half resolution
      return this.createTexture({
        ...descriptor,
        size: {
          width: descriptor.size.width / 2,
          height: descriptor.size.height / 2,
          depthOrArrayLayers: descriptor.size.depthOrArrayLayers || 1,
        },
      });
    }

    return texture;
  }

  async executeCommandsWithErrorHandling(commands) {
    this.device.pushErrorScope("validation");

    const commandEncoder = this.device.createCommandEncoder();
    commands(commandEncoder);
    const commandBuffer = commandEncoder.finish();

    const error = await this.device.popErrorScope();

    if (error) {
      console.error("Command encoding failed:", error.message);
      return false;
    }

    this.device.queue.submit([commandBuffer]);
    return true;
  }
}
```

## Uncaptured Errors

When errors are not captured by any error scope, they trigger the `uncapturederror` event on the device. This provides a global error handler for unexpected errors.

### GPUUncapturedErrorEvent

```javascript
device.addEventListener("uncapturederror", (event) => {
  // event.error is a GPUError (GPUValidationError, GPUOutOfMemoryError, or GPUInternalError)
  console.error("Uncaptured GPU error:", event.error.message);

  if (event.error instanceof GPUValidationError) {
    console.error("Validation error - check API usage");
  } else if (event.error instanceof GPUOutOfMemoryError) {
    console.error("Out of memory - reduce resource usage");
  } else if (event.error instanceof GPUInternalError) {
    console.error("Internal error - may be driver or hardware issue");
  }
});
```

### Global Error Handling Strategy

Implement a comprehensive error handling strategy combining error scopes and uncaptured error handlers:

```javascript
class WebGPUApplication {
  constructor() {
    this.device = null;
    this.errorLog = [];
  }

  async initialize(adapter) {
    this.device = await adapter.requestDevice();

    // Global uncaptured error handler
    this.device.addEventListener("uncapturederror", (event) => {
      this.handleUncapturedError(event.error);
    });

    // Monitor device lost
    this.device.lost.then((info) => {
      this.handleDeviceLost(info);
    });
  }

  handleUncapturedError(error) {
    const errorInfo = {
      timestamp: Date.now(),
      type: error.constructor.name,
      message: error.message,
    };

    this.errorLog.push(errorInfo);

    console.error("Uncaptured GPU error:", errorInfo);

    // Show user-friendly error message
    if (error instanceof GPUValidationError) {
      this.showUserMessage(
        "A graphics error occurred. Please refresh the page.",
      );
    } else if (error instanceof GPUOutOfMemoryError) {
      this.showUserMessage(
        "Insufficient GPU memory. Try closing other applications.",
      );
    } else if (error instanceof GPUInternalError) {
      this.showUserMessage(
        "A GPU error occurred. Please update your graphics drivers.",
      );
    }

    // Send to error tracking service
    this.reportErrorToService(errorInfo);
  }

  showUserMessage(message) {
    // Display user-friendly error message
    console.log("User message:", message);
  }

  reportErrorToService(errorInfo) {
    // Send to analytics/error tracking
    console.log("Reporting error:", errorInfo);
  }

  handleDeviceLost(info) {
    console.error("Device lost:", info.reason, info.message);
  }
}
```

### When Errors Become Uncaptured

Errors become uncaptured when:

1. No error scope is active when the error occurs
2. The active error scope filters don't match the error type
3. An error occurs after all scopes have been popped

Example:

```javascript
// No error scope - this will be uncaptured
const buffer = device.createBuffer({
  size: 256,
  usage: 0, // Invalid: no usage flags
});

// Mismatched filter - validation error won't be captured
device.pushErrorScope("out-of-memory");
const texture = device.createTexture({
  size: { width: 0, height: 0 }, // Invalid: zero size
  format: "rgba8unorm",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const error = await device.popErrorScope(); // Will be null, error is uncaptured
```

## Device Lost Handling

Device lost represents the most severe failure condition in WebGPU. The `device.lost` promise resolves when the device becomes permanently unavailable.

### GPUDeviceLostInfo

```javascript
device.lost.then((info) => {
  console.error("Device lost!");
  console.error("Reason:", info.reason); // 'unknown' or 'destroyed'
  console.error("Message:", info.message);
});
```

The `reason` field has two possible values:

- `'destroyed'`: Application explicitly called `device.destroy()`
- `'unknown'`: Device lost due to hardware failure, driver crash, or other system issue

### Recovery Strategies

#### Strategy 1: Automatic Device Recovery

```javascript
class ResilientWebGPUApp {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.resources = new Map();
  }

  async initialize() {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported");
    }

    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error("No GPU adapter found");
    }

    await this.initializeDevice();
  }

  async initializeDevice() {
    this.device = await this.adapter.requestDevice();

    // Monitor device lost
    this.device.lost.then((info) => {
      console.warn("Device lost, attempting recovery...", info);
      this.handleDeviceLost(info);
    });

    // Global error handler
    this.device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured error:", event.error);
    });
  }

  async handleDeviceLost(info) {
    if (info.reason === "destroyed") {
      console.log("Device was explicitly destroyed");
      return;
    }

    // Clear all resource references
    this.resources.clear();

    // Attempt to recreate device
    try {
      await this.initializeDevice();
      console.log("Device recovered successfully");

      // Recreate all resources
      await this.recreateResources();

      // Resume rendering
      this.resume();
    } catch (error) {
      console.error("Failed to recover device:", error);
      this.showFatalError(
        "GPU device could not be recovered. Please refresh the page.",
      );
    }
  }

  async recreateResources() {
    // Recreate all GPU resources
    // This is application-specific
    console.log("Recreating GPU resources...");
  }

  resume() {
    // Resume application rendering
    console.log("Resuming application...");
  }

  showFatalError(message) {
    // Display error to user
    console.error("Fatal error:", message);
  }
}
```

#### Strategy 2: Graceful Degradation

```javascript
class WebGPURenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.fallbackMode = false;
  }

  async initialize(adapter) {
    this.device = await adapter.requestDevice();

    this.device.lost.then((info) => {
      console.warn("Device lost:", info.reason);
      this.enableFallbackMode();
    });
  }

  enableFallbackMode() {
    this.fallbackMode = true;
    console.log("Switching to fallback rendering mode");

    // Switch to Canvas 2D or WebGL fallback
    const ctx = this.canvas.getContext("2d");
    if (ctx) {
      this.renderWithCanvas2D(ctx);
    } else {
      this.showStaticMessage();
    }
  }

  renderWithCanvas2D(ctx) {
    // Simplified rendering using Canvas 2D API
    ctx.fillStyle = "#333";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = "#fff";
    ctx.font = "24px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      "GPU rendering unavailable",
      this.canvas.width / 2,
      this.canvas.height / 2,
    );
  }

  showStaticMessage() {
    // Display static error message
    console.log("No fallback rendering available");
  }
}
```

### Recreating Device and Resources

After device recovery, you must recreate all GPU resources:

```javascript
class GPUResourceCache {
  constructor() {
    this.descriptors = new Map();
    this.resources = new Map();
  }

  createBuffer(device, id, descriptor) {
    // Store descriptor for recreation
    this.descriptors.set(id, { type: "buffer", descriptor });

    const buffer = device.createBuffer(descriptor);
    this.resources.set(id, buffer);

    return buffer;
  }

  createTexture(device, id, descriptor) {
    this.descriptors.set(id, { type: "texture", descriptor });

    const texture = device.createTexture(descriptor);
    this.resources.set(id, texture);

    return texture;
  }

  async recreateAll(device) {
    // Clear old resources
    this.resources.clear();

    // Recreate from descriptors
    for (const [id, { type, descriptor }] of this.descriptors) {
      if (type === "buffer") {
        const buffer = device.createBuffer(descriptor);
        this.resources.set(id, buffer);
      } else if (type === "texture") {
        const texture = device.createTexture(descriptor);
        this.resources.set(id, texture);
      }
    }

    console.log(`Recreated ${this.resources.size} resources`);
  }

  getResource(id) {
    return this.resources.get(id);
  }
}
```

## Validation Layers

Modern browsers provide powerful developer tools for debugging WebGPU validation errors.

### Browser Developer Tools

Enable WebGPU validation in your browser:

**Chrome/Edge:**

1. Open DevTools (F12)
2. Navigate to Console
3. Validation errors appear with detailed messages

**Firefox:**

1. Open Web Console (Ctrl+Shift+K)
2. WebGPU errors show with stack traces

### Validation Messages and Stack Traces

Validation errors include detailed messages indicating the problem:

```javascript
// Example validation error output:
// GPUValidationError: createBuffer: usage (0x0) must be non-zero
//   at GPUDevice.createBuffer
//   at createResources (app.js:42)
//   at initialize (app.js:15)
```

### Common Validation Errors and Solutions

#### Error: "Buffer usage must be non-zero"

```javascript
// WRONG
const buffer = device.createBuffer({
  size: 256,
  usage: 0,
});

// CORRECT
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
```

#### Error: "Buffer size must be multiple of 4"

```javascript
// WRONG
const buffer = device.createBuffer({
  size: 255, // Not aligned
  usage: GPUBufferUsage.STORAGE,
});

// CORRECT
const buffer = device.createBuffer({
  size: 256, // Aligned to 4 bytes
  usage: GPUBufferUsage.STORAGE,
});
```

#### Error: "Bind group layout doesn't match pipeline"

```javascript
// WRONG - mismatched binding
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    },
  ],
});

const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {
      binding: 1, // Wrong binding number!
      resource: { buffer },
    },
  ],
});

// CORRECT
const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0, // Matches layout
      resource: { buffer },
    },
  ],
});
```

#### Error: "Texture format not renderable"

```javascript
// WRONG
const texture = device.createTexture({
  size: { width: 512, height: 512 },
  format: "r32float", // Not renderable
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

// CORRECT
const texture = device.createTexture({
  size: { width: 512, height: 512 },
  format: "rgba8unorm", // Renderable format
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

## TypeGPU Error Patterns

TypeGPU provides type-safe abstractions that prevent many runtime errors at compile time through TypeScript's type system.

### Type-Safe Error Prevention

TypeGPU catches errors during development rather than at runtime:

```typescript
import tgpu from "typegpu";

// TypeScript will catch type mismatches
const root = await tgpu.init();

// WRONG - TypeScript error
const buffer = root.createBuffer(
  tgpu.arrayOf(tgpu.f32, 100),
  new Float32Array(50), // Error: Array size mismatch!
);

// CORRECT
const buffer = root.createBuffer(
  tgpu.arrayOf(tgpu.f32, 100),
  new Float32Array(100),
);
```

### How TypeGPU Reduces Runtime Errors

TypeGPU eliminates entire categories of WebGPU errors:

1. **Binding type mismatches**: TypeScript ensures shader bindings match bind groups
2. **Data layout errors**: Schema validation prevents struct layout mistakes
3. **Usage flag errors**: TypeGPU automatically sets appropriate usage flags
4. **Format mismatches**: Type system enforces compatible texture formats

Example - Automatic Usage Flag Management:

```typescript
// TypeGPU automatically adds required usage flags
const buffer = root.createBuffer(
  tgpu.arrayOf(tgpu.vec3f, 1000),
  // Usage flags automatically set based on how buffer is used
);

// WebGPU equivalent - manual usage management
const rawBuffer = device.createBuffer({
  size: 1000 * 3 * 4,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, // Must manually specify all needed flags
});
```

### Schema Validation

TypeGPU validates data schemas at compile time:

```typescript
import * as d from "typegpu/data";

// Define schema
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  mass: d.f32,
});

// WRONG - TypeScript error
const invalidData = {
  position: [1, 2], // Missing z component!
  velocity: [0, 0, 0],
  mass: 1.0,
};

// CORRECT - type-checked
const validData: (typeof Particle)["~repr"] = {
  position: [1, 2, 3],
  velocity: [0, 0, 0],
  mass: 1.0,
};

const buffer = root.createBuffer(Particle, validData);
```

### Runtime Validation with TypeGPU

Even with TypeGPU, add runtime validation for dynamic data:

```typescript
function createParticleBuffer(root: TgpuRoot, particleCount: number) {
  // Validate dynamic parameters
  if (particleCount <= 0) {
    throw new Error("Particle count must be positive");
  }

  if (particleCount > 1000000) {
    throw new Error("Particle count too large for GPU memory");
  }

  const ParticleArray = d.arrayOf(Particle, particleCount);

  try {
    return root.createBuffer(ParticleArray);
  } catch (error) {
    console.error("Failed to create particle buffer:", error);
    throw error;
  }
}
```

## Debugging Strategies

Effective debugging requires a systematic approach to identifying and resolving GPU errors.

### Systematic Debugging Approach

1. **Enable all error scopes during development**
2. **Add descriptive labels to resources**
3. **Use small test cases to isolate problems**
4. **Check browser console for validation messages**
5. **Verify resource creation succeeded before using**

### Labeling Resources for Better Errors

```javascript
const buffer = device.createBuffer({
  label: "Particle Position Buffer", // Appears in error messages!
  size: 1024,
  usage: GPUBufferUsage.STORAGE,
});

const texture = device.createTexture({
  label: "Shadow Map Texture",
  size: { width: 2048, height: 2048 },
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

// Validation errors now include labels:
// "Texture 'Shadow Map Texture': format 'depth24plus' requires DEPTH_STENCIL_ATTACHMENT usage"
```

### Common Error Scenarios

#### Scenario 1: Shader Compilation Errors

```javascript
async function debugShaderCompilation(device, shaderCode) {
  device.pushErrorScope("validation");

  const shaderModule = device.createShaderModule({
    label: "Debug Shader",
    code: shaderCode,
  });

  const error = await device.popErrorScope();

  if (error) {
    console.error("Shader compilation failed:");
    console.error(error.message);

    // Parse error message for line numbers
    const lineMatch = error.message.match(/line (\d+)/);
    if (lineMatch) {
      const lineNum = parseInt(lineMatch[1]);
      const lines = shaderCode.split("\n");
      console.error(`Error at line ${lineNum}:`, lines[lineNum - 1]);
    }

    return null;
  }

  return shaderModule;
}
```

#### Scenario 2: Buffer Mapping Failures

```javascript
async function safeMapBuffer(buffer, mode) {
  try {
    await buffer.mapAsync(mode);
    return buffer.getMappedRange();
  } catch (error) {
    console.error("Buffer mapping failed:", error);
    console.error("Buffer label:", buffer.label);
    console.error("Map mode:", mode);

    // Check if buffer was destroyed
    if (error.message.includes("destroyed")) {
      console.error("Buffer was destroyed before mapping");
    }

    // Check if already mapped
    if (error.message.includes("already mapped")) {
      console.error("Buffer is already mapped");
    }

    throw error;
  }
}
```

### Console Logging Patterns

```javascript
class DebugGPUDevice {
  constructor(device) {
    this.device = device;
    this.operationCount = 0;
  }

  createBuffer(descriptor) {
    const opId = ++this.operationCount;
    console.log(`[${opId}] Creating buffer:`, descriptor);

    this.device.pushErrorScope("validation");
    this.device.pushErrorScope("out-of-memory");

    const buffer = this.device.createBuffer(descriptor);

    this.device.popErrorScope().then((error) => {
      if (error) console.error(`[${opId}] OOM error:`, error.message);
    });

    this.device.popErrorScope().then((error) => {
      if (error) {
        console.error(`[${opId}] Validation error:`, error.message);
      } else {
        console.log(`[${opId}] Buffer created successfully`);
      }
    });

    return buffer;
  }

  submit(commandBuffers) {
    const opId = ++this.operationCount;
    console.log(
      `[${opId}] Submitting ${commandBuffers.length} command buffers`,
    );

    this.device.pushErrorScope("validation");
    this.device.queue.submit(commandBuffers);

    this.device.popErrorScope().then((error) => {
      if (error) {
        console.error(`[${opId}] Submit error:`, error.message);
      } else {
        console.log(`[${opId}] Submit successful`);
      }
    });
  }
}
```

## Best Practices

### Error Handling Architecture

1. **Use error scopes for all resource creation during development**
2. **Implement global uncaptured error handler**
3. **Monitor device.lost for catastrophic failures**
4. **Log errors to analytics in production**
5. **Provide user-friendly error messages**

```javascript
class ProductionGPUApp {
  async initialize() {
    this.device = await this.adapter.requestDevice();

    // Production error handling
    this.device.addEventListener("uncapturederror", (event) => {
      // Log to error tracking service
      this.logToAnalytics({
        type: "uncaptured_gpu_error",
        errorType: event.error.constructor.name,
        message: event.error.message,
        userAgent: navigator.userAgent,
        timestamp: Date.now(),
      });

      // Show user message
      this.showErrorNotification(
        "A graphics error occurred. Please try refreshing.",
      );
    });

    this.device.lost.then((info) => {
      this.logToAnalytics({
        type: "device_lost",
        reason: info.reason,
        message: info.message,
      });

      if (info.reason !== "destroyed") {
        this.attemptRecovery();
      }
    });
  }

  logToAnalytics(data) {
    // Send to analytics service
    console.log("Analytics:", data);
  }

  showErrorNotification(message) {
    // Display to user
    console.log("User notification:", message);
  }

  attemptRecovery() {
    // Recovery logic
    console.log("Attempting device recovery...");
  }
}
```

### Logging Strategy

```javascript
class GPUErrorLogger {
  constructor(enableConsole = true, enableRemote = false) {
    this.enableConsole = enableConsole;
    this.enableRemote = enableRemote;
    this.errors = [];
  }

  logError(context, error) {
    const entry = {
      timestamp: Date.now(),
      context,
      type: error.constructor.name,
      message: error.message,
      stack: error.stack,
    };

    this.errors.push(entry);

    if (this.enableConsole) {
      console.error(`[${context}]`, error.message);
    }

    if (this.enableRemote) {
      this.sendToRemote(entry);
    }
  }

  sendToRemote(entry) {
    // Send to remote logging service
    fetch("/api/gpu-errors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    }).catch((err) => console.error("Failed to log remotely:", err));
  }

  getErrorSummary() {
    const summary = {};
    for (const error of this.errors) {
      summary[error.type] = (summary[error.type] || 0) + 1;
    }
    return summary;
  }
}
```

### User Communication

```javascript
function showGPUError(error) {
  let userMessage = "An error occurred with GPU rendering.";
  let technicalMessage = error.message;

  if (error instanceof GPUValidationError) {
    userMessage =
      "The application encountered a graphics configuration error. Please refresh the page.";
  } else if (error instanceof GPUOutOfMemoryError) {
    userMessage =
      "Insufficient GPU memory. Try closing other applications or reducing graphics quality.";
  } else if (error instanceof GPUInternalError) {
    userMessage =
      "A GPU driver error occurred. Please update your graphics drivers and try again.";
  }

  // Display user-friendly message
  displayNotification(userMessage);

  // Log technical details
  console.error("Technical error:", technicalMessage);
}
```

## Common Pitfalls

### Missing Error Handling

```javascript
// WRONG - No error handling
const buffer = device.createBuffer({
  size: 1000000000, // May fail with OOM
  usage: GPUBufferUsage.STORAGE,
});
device.queue.writeBuffer(buffer, 0, data); // May fail silently if buffer is invalid

// CORRECT - Proper error handling
async function createBufferSafely(device, size, data) {
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");

  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const memError = await device.popErrorScope();
  const valError = await device.popErrorScope();

  if (memError || valError) {
    throw new Error(
      `Buffer creation failed: ${memError?.message || valError?.message}`,
    );
  }

  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
```

### Async Timing Issues

```javascript
// WRONG - Race condition
device.pushErrorScope("validation");
const buffer1 = device.createBuffer(descriptor1);
const error1 = await device.popErrorScope(); // May capture buffer2 error!
const buffer2 = device.createBuffer(descriptor2);

// CORRECT - Proper scope management
device.pushErrorScope("validation");
const buffer1 = device.createBuffer(descriptor1);
const error1 = await device.popErrorScope();

device.pushErrorScope("validation");
const buffer2 = device.createBuffer(descriptor2);
const error2 = await device.popErrorScope();
```

### Silent Failures

```javascript
// WRONG - Errors go uncaptured
function createPipeline(device, descriptor) {
  return device.createRenderPipeline(descriptor);
  // Validation errors will be uncaptured!
}

// CORRECT - Capture and handle errors
async function createPipeline(device, descriptor) {
  device.pushErrorScope("validation");
  const pipeline = device.createRenderPipeline(descriptor);
  const error = await device.popErrorScope();

  if (error) {
    console.error("Pipeline creation failed:", error.message);
    throw error;
  }

  return pipeline;
}
```

### Forgetting Device Lost Monitoring

```javascript
// WRONG - No device lost handling
const device = await adapter.requestDevice();
// Device may become lost without warning!

// CORRECT - Monitor device lost
const device = await adapter.requestDevice();

device.lost.then((info) => {
  console.error("Device lost:", info.reason, info.message);

  if (info.reason !== "destroyed") {
    // Attempt recovery or show error to user
    handleDeviceLost(info);
  }
});
```

### Not Labeling Resources

```javascript
// WRONG - Generic error messages
const buffer = device.createBuffer({
  size: 256,
  usage: 0, // Error: "Buffer usage must be non-zero"
});

// CORRECT - Labeled resources give better errors
const buffer = device.createBuffer({
  label: "Particle Velocity Buffer",
  size: 256,
  usage: 0, // Error: "Buffer 'Particle Velocity Buffer': usage must be non-zero"
});
```

---

## Summary

WebGPU's error handling model represents a significant evolution from WebGL, trading synchronous simplicity for asynchronous performance and composability. By understanding error types, mastering error scopes, handling device loss, and following best practices, you can build robust WebGPU applications that gracefully handle errors and provide excellent user experiences.

Key takeaways:

1. Always use error scopes during development to catch validation and out-of-memory errors
2. Implement a global uncaptured error handler for unexpected errors
3. Monitor `device.lost` and implement recovery strategies
4. Label all resources for better error messages
5. Use TypeGPU to prevent many errors at compile time
6. Provide user-friendly error messages in production
7. Log errors to analytics for monitoring
8. Test error handling paths as thoroughly as success paths

With proper error handling, your WebGPU applications can recover from failures, provide helpful debugging information during development, and maintain stability in production environments.
