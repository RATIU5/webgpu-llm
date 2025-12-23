---
title: Debugging GPU Code
sidebar:
  order: 30
---

## Overview

Debugging GPU code presents unique challenges that differ fundamentally from traditional CPU debugging. GPU programs execute in a massively parallel environment on specialized hardware, often in a separate process from your application code, making traditional debugging approaches ineffective or impossible. Understanding these challenges and mastering the available debugging tools is essential for productive WebGPU and TypeGPU development.

Modern GPU debugging combines multiple strategies: runtime validation layers that catch common errors, specialized console logging that bridges the CPU-GPU divide, CPU simulation modes for traditional step-through debugging, visual debugging techniques that encode program state as colors, and browser developer tools designed specifically for GPU workloads. This guide explores each approach in depth, providing practical techniques you can apply immediately to diagnose and fix GPU code issues.

Whether you're tracking down shader compilation errors, investigating unexpected rendering results, or optimizing performance, effective debugging workflows can dramatically reduce development time and frustration. The key is understanding when to use each tool and how to combine them effectively.

## Why GPU Debugging is Hard

### Parallel Execution Model

The fundamental challenge of GPU debugging stems from the massive parallelism of GPU execution. A single GPU compute dispatch might launch thousands or millions of shader invocations simultaneously, each processing different data. Unlike CPU code where execution follows a predictable sequential path, GPU shaders execute in lockstep groups (workgroups) with complex synchronization semantics.

This parallelism makes traditional debugging approaches impractical:

- **No single execution path**: Setting a breakpoint in shader code doesn't stop one thread—it would need to stop thousands of parallel invocations simultaneously
- **Non-deterministic ordering**: The order in which workgroups execute is implementation-defined and can vary between runs
- **Race conditions**: Parallel threads accessing shared resources can produce different results on different hardware or runs
- **Massive output**: Logging from thousands of invocations generates overwhelming amounts of data

Consider a simple compute shader processing a 1024x1024 texture. With a workgroup size of 8x8, this launches 16,384 workgroups containing 1,048,576 total shader invocations. Traditional step-through debugging of each invocation would be completely infeasible.

### Separate Execution Context

GPU code executes in a fundamentally different environment from your application:

- **Different process**: Modern browsers run GPU code in a separate GPU process for security and stability, making cross-process debugging complex
- **Different language**: Shaders are written in WGSL (or transpiled from TGSL), not JavaScript, requiring different tooling
- **Different memory space**: GPU memory is separate from CPU memory, with explicit transfer operations required to inspect data
- **Asynchronous execution**: GPU commands are queued and executed asynchronously, making it difficult to correlate GPU errors with specific application code

This separation means you can't simply attach a debugger to your JavaScript code and step into shader execution. The CPU schedules work for the GPU, but actual shader execution happens independently on the GPU timeline.

### Limited Introspection Capabilities

Traditional debuggers provide rich introspection: examining local variables, evaluating expressions, modifying values at runtime, and stepping through code line by line. GPU execution offers limited equivalents:

- **No breakpoints**: Hardware doesn't support pausing individual shader invocations
- **No interactive inspection**: You can't examine shader variables during execution
- **No runtime modification**: Shader code is compiled and immutable during execution
- **Limited error information**: Errors often manifest as incorrect results rather than exceptions

These limitations require alternative debugging strategies focused on validation, logging, simulation, and visualization rather than interactive debugging.

### Compilation to Native Shaders

TypeGPU adds another layer of complexity: TGSL code is transpiled to WGSL at build time, which is then compiled to native GPU assembly at runtime. This multi-stage compilation process means:

- **Source mapping**: Errors in generated WGSL must be traced back to original TGSL code
- **Transpilation bugs**: Issues might originate in the TypeGPU transpiler rather than your code
- **Optimization interference**: Shader compilers aggressively optimize, potentially obscuring the relationship between source and execution
- **Platform differences**: The same WGSL may compile to different assembly on different GPUs

Understanding the entire compilation pipeline—from TypeScript to TGSL to WGSL to native assembly—is essential for effective debugging.

## TypeGPU Console.log

One of TypeGPU's most developer-friendly features is support for `console.log` within TGSL functions, bridging the CPU-GPU debugging gap by allowing familiar logging syntax in shader code.

### How It Works

TypeGPU's console.log implementation works by automatically generating infrastructure to capture log messages from GPU shaders and report them back to the CPU. When you write `console.log` in TGSL code:

1. **Build-time transformation**: The TypeGPU compiler detects console.log calls and transforms them into buffer write operations
2. **Buffer allocation**: TypeGPU automatically allocates staging buffers to receive log data from the GPU
3. **Data capture**: During shader execution, log messages are serialized and written to these buffers
4. **Readback**: After GPU execution completes, TypeGPU reads the staging buffers back to CPU memory
5. **Console output**: Captured messages are formatted and output to the browser's JavaScript console

This process is largely transparent to developers. You write standard console.log calls, and TypeGPU handles all the buffer management and data transfer behind the scenes.

The implementation must respect GPU execution constraints:

- **Fixed-size buffers**: Log buffers have predetermined sizes, limiting total output
- **No dynamic strings**: GPU shaders can't allocate memory dynamically, so string formatting is limited
- **Deferred output**: Messages only appear after the GPU command completes, not during execution
- **Performance overhead**: Buffer writes consume GPU time and memory bandwidth

Despite these limitations, console.log provides invaluable insight into shader execution, especially during initial development and algorithm verification.

### Usage Examples

Using console.log in TypeGPU shaders is straightforward and follows familiar JavaScript conventions:

```typescript
import { tgpu } from "typegpu";
import * as d from "typegpu/data";

// Basic logging in a compute shader
const processData = tgpu.fn([d.f32], d.f32).does((input) => {
  console.log("Processing input:", input);

  const result = input * 2.0;
  console.log("Computed result:", result);

  return result;
});

// Conditional logging for debugging edge cases
const clampValue = tgpu
  .fn([d.f32, d.f32, d.f32], d.f32)
  .does((value, min, max) => {
    if (value < min) {
      console.log("Value clamped to min:", value, "->", min);
      return min;
    }
    if (value > max) {
      console.log("Value clamped to max:", value, "->", max);
      return max;
    }
    return value;
  });

// Logging in loops (use sparingly!)
const sumArray = tgpu.fn([d.arrayOf(d.f32, 10)], d.f32).does((values) => {
  let sum = 0.0;
  for (let i = 0; i < 10; i++) {
    const val = values[i];
    // Only log first few iterations to avoid overwhelming output
    if (i < 3) {
      console.log("Index", i, "value:", val);
    }
    sum += val;
  }
  console.log("Total sum:", sum);
  return sum;
});

// Debugging vector operations
const normalizeVector = tgpu.fn([d.vec3f], d.vec3f).does((v) => {
  const length = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  console.log("Vector:", v.x, v.y, v.z, "Length:", length);

  if (length < 0.0001) {
    console.log("WARNING: Near-zero length vector!");
    return tgpu.vec3f(0, 0, 0);
  }

  return tgpu.vec3f(v.x / length, v.y / length, v.z / length);
});
```

### Best Practices and Limitations

While console.log is powerful, it must be used judiciously due to inherent constraints:

**Performance Impact**: Every console.log call adds buffer write operations to your shader. For shaders executing millions of times per frame, even lightweight logging can severely impact performance. Use console.log during development and remove it from performance-critical code paths.

**Output Volume Limits**: Log buffers have fixed sizes. If your shader logs too much data, later messages may be truncated or lost. This is especially problematic in parallel execution where thousands of invocations might log simultaneously:

```typescript
// PROBLEMATIC: Logging from every invocation
const processPixel = tgpu.fn([d.u32], d.vec4f).does((pixelIndex) => {
  // In a 1920x1080 image, this logs 2,073,600 times!
  console.log("Processing pixel:", pixelIndex);
  // ...
});

// BETTER: Conditional logging for specific cases
const processPixel = tgpu.fn([d.u32], d.vec4f).does((pixelIndex) => {
  // Only log the first few pixels or specific problem areas
  if (pixelIndex < 10 || pixelIndex === 1000000) {
    console.log("Processing pixel:", pixelIndex);
  }
  // ...
});
```

**Limited Context Availability**: Console.log may not be available in all shader types or execution contexts. Certain GPU operations, particularly fragment shaders with complex derivatives or early-depth testing, may have restrictions on buffer writes.

**Asynchronous Nature**: Log messages appear only after GPU execution completes and data is read back. You won't see real-time output during shader execution, making it less useful for timing-sensitive debugging.

**String Formatting Limitations**: Unlike JavaScript's console.log, TGSL console.log has limited string formatting capabilities due to GPU memory constraints. Complex object serialization or dynamic string construction may not work as expected.

## Validation Layers

WebGPU's validation layers provide comprehensive automatic error detection, catching common mistakes before they reach the GPU driver. Unlike WebGL's synchronous `getError()` approach, WebGPU validates operations asynchronously and reports detailed error messages with stack traces.

### Browser Validation

Modern browsers implement robust validation of all WebGPU operations. When validation is enabled (typically by default in development builds), the browser checks every operation against the WebGPU specification:

- **Resource creation**: Validates buffer sizes, texture formats, usage flags, and descriptor parameters
- **Pipeline construction**: Ensures shader modules are valid and pipeline configurations are compatible
- **Command encoding**: Validates render pass attachments, compute dispatches, and resource bindings
- **Resource usage**: Detects conflicts like simultaneous read-write access to the same resource

Validation errors are reported through the browser's developer console with detailed messages explaining what went wrong and often including stack traces pointing to the offending JavaScript code.

The browser validation layer is your first line of defense against bugs. It catches most common mistakes immediately with clear error messages, making it far more effective than hunting for mysterious rendering glitches or crashes.

### Common Validation Errors

Understanding common validation error patterns helps you quickly diagnose and fix issues:

**Resource Usage Conflicts**:

```typescript
// ERROR: Buffer created without COPY_DST but used for writing
const buffer = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.UNIFORM, // Missing COPY_DST flag
});

device.queue.writeBuffer(buffer, 0, data);
// Validation error: "Buffer is missing COPY_DST usage flag"
```

**Binding Mismatches**:

```typescript
// ERROR: Bind group layout doesn't match pipeline layout
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    },
  ],
});

// Shader expects storage buffer but layout specifies uniform buffer
const shaderCode = `
  @group(0) @binding(0) var<storage, read> data: array<f32>;
  // ...
`;
// Validation error: "Bind group layout entry 0 type mismatch"
```

**Invalid State Transitions**:

```typescript
// ERROR: Texture view format incompatible with texture format
const texture = device.createTexture({
  size: { width: 512, height: 512 },
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING,
});

const view = texture.createView({
  format: "rgba16float", // Incompatible with texture's rgba8unorm
});
// Validation error: "Texture view format incompatible with texture format"
```

### Reading and Understanding Error Messages

WebGPU validation errors are designed to be informative. A typical error message includes:

1. **Error type**: "ValidationError", "OperationError", etc.
2. **Operation context**: Which WebGPU method failed
3. **Specific violation**: Exactly what constraint was violated
4. **Stack trace**: JavaScript call stack leading to the error

Example comprehensive error:

```
WebGPU ValidationError: createBindGroup():
  Bind group entry 1 buffer binding size (128) is smaller than
  minimum binding size (256) specified in bind group layout.

  at GPUDevice.createBindGroup (webgpu-runtime.js:423)
  at setupBindings (app.js:156)
  at initializePipeline (app.js:89)
  at main (app.js:15)
```

This error clearly indicates: the bind group at entry 1 has a buffer that's too small (128 bytes instead of required 256 bytes), and the stack trace shows exactly where in your code this happened.

## tgpu.simulate()

TypeGPU's `tgpu.simulate()` API enables running TGSL code directly on the CPU using standard JavaScript execution, unlocking traditional debugging capabilities for shader code.

### CPU Debugging with Traditional Tools

When you call `tgpu.simulate()`, TypeGPU executes your TGSL function as regular JavaScript instead of compiling it to WGSL and running it on the GPU. This means you can use all familiar debugging tools:

```typescript
import { tgpu } from "typegpu";
import * as d from "typegpu/data";

const computeGradient = tgpu.fn([d.f32, d.f32], d.f32).does((x, y) => {
  const dx = x * 2.0;
  const dy = y * 3.0;
  return Math.sqrt(dx * dx + dy * dy);
});

// Simulate CPU execution - full debugging available
const result = tgpu.simulate(computeGradient)(5.0, 3.0);
console.log("Simulated result:", result);

// Set breakpoints in the shader code and step through line-by-line
// Inspect variables: dx, dy, intermediate calculations
// Evaluate expressions in the debugger console
```

During simulation:

- **Breakpoints work**: Set breakpoints anywhere in the TGSL function and execution pauses
- **Variable inspection**: Examine all local variables and their values
- **Step execution**: Step through code line by line or expression by expression
- **Call stack**: See the full call stack including nested function calls
- **Expression evaluation**: Use the debugger console to evaluate arbitrary expressions

This is invaluable for complex algorithm development where you need to understand exactly what's happening at each step.

### When to Use Simulation

CPU simulation is most effective in specific scenarios:

**Algorithm Verification**: When developing new shader algorithms, simulate with known inputs and verify outputs match expectations. This is much faster than running on GPU and inspecting results.

```typescript
// Develop and verify a complex physics calculation
const calculateForce = tgpu
  .fn([d.vec3f, d.vec3f, d.f32], d.vec3f)
  .does((position, velocity, mass) => {
    // Complex force calculation
    const gravity = tgpu.vec3f(0, -9.8, 0);
    const drag = velocity.mul(-0.1);
    const total = gravity.add(drag).mul(mass);
    return total;
  });

// Simulate to verify physics are correct
const force = tgpu.simulate(calculateForce)(
  tgpu.vec3f(0, 10, 0), // position
  tgpu.vec3f(5, -2, 3), // velocity
  2.5, // mass
);

// Use debugger to step through and verify each calculation step
console.log("Computed force:", force);
```

**Edge Case Testing**: Test boundary conditions that might be rare in GPU execution but critical to handle correctly:

```typescript
const safeDivide = tgpu
  .fn([d.f32, d.f32], d.f32)
  .does((numerator, denominator) => {
    if (Math.abs(denominator) < 0.0001) {
      return 0.0; // Avoid division by near-zero
    }
    return numerator / denominator;
  });

// Simulate edge cases
console.log(tgpu.simulate(safeDivide)(10, 0)); // 0.0
console.log(tgpu.simulate(safeDivide)(10, 0.00001)); // 0.0
console.log(tgpu.simulate(safeDivide)(10, 2)); // 5.0
```

**Initial Development**: Start shader development using simulation to rapidly iterate on algorithm logic before dealing with GPU-specific concerns like memory layout and parallelism.

**Limitations of Simulation**: Remember that simulation runs sequential JavaScript, not parallel GPU code. It won't catch:

- Race conditions in parallel execution
- Memory access pattern issues
- Performance problems
- GPU-specific numerical precision differences

Always test on actual GPU after verifying logic with simulation.

## Visual Debugging

Visual debugging encodes program state as colors or patterns in rendered images, leveraging the human visual system to identify anomalies that would be hard to spot in numerical output.

### Rendering Debug Information as Colors

The core technique of visual debugging is mapping shader values to color channels for visualization:

```typescript
// Debug shader: visualize normal vectors as RGB colors
const debugNormals = tgpu.fn([d.vec3f], d.vec4f).does((normal) => {
  // Transform normals from [-1, 1] to [0, 1] color range
  const r = normal.x * 0.5 + 0.5;
  const g = normal.y * 0.5 + 0.5;
  const b = normal.z * 0.5 + 0.5;

  return tgpu.vec4f(r, g, b, 1.0);
});

// Debug shader: visualize depth as grayscale
const debugDepth = tgpu
  .fn([d.f32, d.f32, d.f32], d.vec4f)
  .does((depth, nearPlane, farPlane) => {
    // Linearize and normalize depth
    const linear = (depth - nearPlane) / (farPlane - nearPlane);
    const clamped = Math.max(0.0, Math.min(1.0, linear));

    return tgpu.vec4f(clamped, clamped, clamped, 1.0);
  });

// Debug shader: color-code value ranges
const debugValueRange = tgpu.fn([d.f32], d.vec4f).does((value) => {
  // Red for negative, green for 0-1, blue for >1
  if (value < 0.0) {
    return tgpu.vec4f(1.0, 0.0, 0.0, 1.0); // Red
  } else if (value <= 1.0) {
    return tgpu.vec4f(0.0, value, 0.0, 1.0); // Green gradient
  } else {
    return tgpu.vec4f(0.0, 0.0, 1.0, 1.0); // Blue
  }
});
```

### Common Visual Debugging Techniques

**Normals as Colors**: Surface normals encoded as RGB reveal geometry and normal calculation issues. Smooth color gradients indicate correct normals; sharp transitions or unexpected colors indicate problems.

**Depth Visualization**: Rendering depth as grayscale helps debug depth buffer issues, clipping plane problems, and z-fighting artifacts. Near surfaces appear dark, far surfaces light (or vice versa).

**UV Coordinate Debugging**: Visualize texture coordinates by assigning U to red channel and V to green channel:

```typescript
const debugUVs = tgpu.fn([d.vec2f], d.vec4f).does((uv) => {
  return tgpu.vec4f(uv.x, uv.y, 0.0, 1.0);
});
```

Correct UVs show smooth red-green gradients. Discontinuities indicate UV mapping problems.

**Heat Maps**: Visualize scalar values (density, energy, iteration counts) using color gradients from cool to hot colors:

```typescript
const heatMap = tgpu.fn([d.f32], d.vec4f).does((value) => {
  // Value from 0 to 1 maps to blue -> cyan -> green -> yellow -> red
  const clamped = Math.max(0.0, Math.min(1.0, value));

  let r = 0.0,
    g = 0.0,
    b = 0.0;

  if (clamped < 0.25) {
    // Blue to cyan
    b = 1.0;
    g = clamped * 4.0;
  } else if (clamped < 0.5) {
    // Cyan to green
    g = 1.0;
    b = (0.5 - clamped) * 4.0;
  } else if (clamped < 0.75) {
    // Green to yellow
    g = 1.0;
    r = (clamped - 0.5) * 4.0;
  } else {
    // Yellow to red
    r = 1.0;
    g = (1.0 - clamped) * 4.0;
  }

  return tgpu.vec4f(r, g, b, 1.0);
});
```

**Binary Flags**: Use distinct colors to visualize boolean conditions or categorical data:

```typescript
const debugConditions = tgpu
  .fn([d.bool, d.bool], d.vec4f)
  .does((conditionA, conditionB) => {
    if (conditionA && conditionB) {
      return tgpu.vec4f(1, 1, 0, 1); // Yellow: both true
    } else if (conditionA) {
      return tgpu.vec4f(1, 0, 0, 1); // Red: only A true
    } else if (conditionB) {
      return tgpu.vec4f(0, 1, 0, 1); // Green: only B true
    } else {
      return tgpu.vec4f(0, 0, 1, 1); // Blue: both false
    }
  });
```

Visual debugging is especially powerful for spatial algorithms (rendering, physics simulation, fluid dynamics) where seeing patterns immediately reveals problems that would be obscure in numerical logs.

## Shader Compilation Errors

WGSL compilation errors occur when shader code violates WGSL language rules. Understanding error types and debugging strategies accelerates fixing these issues.

### Types of WGSL Errors

**Parser Errors**: Syntax violations that prevent WGSL from being parsed:

```wgsl
// ERROR: Missing semicolon
fn compute(x: f32) -> f32 {
  let result = x * 2.0  // Parser error: expected ';'
  return result;
}

// ERROR: Invalid token
fn compute(x: f32) -> f32 {
  return x ** 2.0;  // Parser error: '**' is not valid (use pow)
}
```

**Type Errors**: Type mismatches or invalid type usage:

```wgsl
// ERROR: Type mismatch
fn compute(x: f32) -> i32 {
  return x;  // Type error: cannot return f32 from i32 function
}

// ERROR: Invalid type conversion
fn compute(x: vec3f) -> f32 {
  return x;  // Type error: cannot implicitly convert vec3f to f32
}
```

**Semantic Errors**: Code that parses correctly but violates WGSL semantic rules:

```wgsl
// ERROR: Binding number used twice
@group(0) @binding(0) var<uniform> data1: f32;
@group(0) @binding(0) var<uniform> data2: f32;
// Semantic error: binding 0 in group 0 used multiple times

// ERROR: Invalid address space for variable
fn compute() -> f32 {
  var<workgroup> x: f32 = 1.0;
  // Semantic error: workgroup variables must be module scope
  return x;
}
```

### Debugging Compilation Errors

When faced with compilation errors, systematic approaches help quickly identify root causes:

**Isolate the Problem**: Comment out shader code sections until the error disappears, then progressively re-enable code to pinpoint the exact line causing the issue.

**Minimal Reproduction**: Create the smallest possible shader that reproduces the error:

```typescript
// Start with minimal working shader
const minimalShader = `
  @compute @workgroup_size(1)
  fn main() {
    // Add code incrementally until error appears
  }
`;
```

**Check Generated WGSL**: For TypeGPU/TGSL code, inspect the generated WGSL to determine if the error is in your TGSL or in the generated output. TypeGPU provides ways to view generated shader code.

**Verify Types**: Many errors stem from type mismatches. Explicitly annotate types to clarify intent:

```typescript
// Implicit - harder to debug
const value = computeSomething(x);

// Explicit - compiler helps verify correctness
const value: f32 = computeSomething(x);
```

**Consult WGSL Specification**: The [official WGSL specification](https://www.w3.org/TR/WGSL/) authoritatively defines valid syntax and semantics. When error messages are unclear, checking the spec clarifies what's allowed.

## TypeGPU Translator Tool

TypeGPU provides mechanisms to inspect the WGSL code generated from TGSL functions, essential for understanding what actually runs on the GPU and debugging transpilation issues.

### Inspecting Generated WGSL

While TypeGPU aims for semantic equivalence between TGSL and generated WGSL, seeing the actual GPU code is valuable for:

- **Verifying correctness**: Ensure transpilation produced expected WGSL
- **Performance analysis**: Identify optimization opportunities in generated code
- **Debugging transpilation bugs**: Determine if issues stem from your code or TypeGPU
- **Learning WGSL**: Compare TGSL input with WGSL output to understand WGSL patterns

TypeGPU's build tooling and runtime can expose generated WGSL through various mechanisms. For example, you can log the generated shader module code or use browser developer tools to inspect shader sources.

Understanding the TGSL-to-WGSL mapping helps when debugging:

```typescript
// TGSL code
const multiply = tgpu.fn([d.f32, d.f32], d.f32).does((a, b) => {
  return a * b;
});

// Generates WGSL similar to:
// fn multiply(a: f32, b: f32) -> f32 {
//   return a * b;
// }
```

For more complex code involving vectors and matrices, the generated WGSL uses WGSL-specific syntax that may differ from TypeScript:

```typescript
// TGSL with vector operations
const dotProduct = tgpu.fn([d.vec3f, d.vec3f], d.f32).does((a, b) => {
  return a.x * b.x + a.y * b.y + a.z * b.z;
});

// Generated WGSL might use built-in:
// fn dotProduct(a: vec3f, b: vec3f) -> f32 {
//   return dot(a, b);
// }
```

## Runtime Errors

Runtime GPU errors occur during command execution rather than during setup or compilation. These errors require different debugging approaches.

### Device Lost

Device lost is the most catastrophic runtime error, indicating the GPU is no longer available. Common causes include:

- **Driver crashes**: GPU driver encountered an unrecoverable error (TDR - Timeout Detection and Recovery)
- **Hardware issues**: GPU overheating, instability, or physical disconnection
- **Resource exhaustion**: System ran out of GPU memory or other critical resources
- **Explicit destruction**: Application called `device.destroy()`

**Detection**:

```typescript
device.lost.then((info) => {
  console.error("Device lost:", info.reason);
  console.error("Message:", info.message);

  if (info.reason === "destroyed") {
    // Device was explicitly destroyed
  } else {
    // Unexpected device loss - likely driver crash or hardware issue
  }
});
```

**Recovery Strategies**:

```typescript
async function handleDeviceLost(device) {
  // Wait for device loss
  const info = await device.lost;

  console.warn("Device lost, attempting recovery...");

  // Request new device
  const newDevice = await adapter.requestDevice();

  // Recreate all resources (buffers, textures, pipelines, etc.)
  await recreateResources(newDevice);

  // Resume rendering with new device
  startRenderLoop(newDevice);
}
```

**Prevention**: While you can't prevent all device losses (hardware failures, driver bugs), you can minimize risk:

- Avoid infinite loops in shaders
- Validate shader execution time doesn't exceed driver watchdog limits
- Monitor GPU memory usage to prevent exhaustion
- Test on multiple devices to catch driver-specific issues

### Out of Memory Errors

GPU out-of-memory errors occur when resource allocation fails due to insufficient VRAM:

**Symptoms**:

- Texture or buffer creation fails
- Pipeline creation succeeds but execution fails
- Gradual performance degradation as system swaps memory

**Detection and Handling**:

```typescript
device.pushErrorScope("out-of-memory");

const largeTexture = device.createTexture({
  size: { width: 8192, height: 8192 },
  format: "rgba16float",
  usage: GPUTextureUsage.STORAGE_BINDING,
});

const error = await device.popErrorScope();
if (error) {
  console.error("Out of memory creating texture:", error.message);
  // Fallback: use smaller texture or alternative approach
  const smallerTexture = device.createTexture({
    size: { width: 4096, height: 4096 },
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING,
  });
}
```

**Prevention Strategies**:

- Monitor VRAM usage and implement resource limits
- Use appropriate texture formats (e.g., `rgba8unorm` instead of `rgba32float` when precision allows)
- Implement texture streaming for large datasets
- Destroy unused resources promptly
- Query device limits and respect them

## Browser Developer Tools

Modern browsers provide specialized tools for WebGPU debugging and inspection.

### Chrome DevTools and Extensions

**WebGPU Inspector** (available for Chrome, Firefox, and Safari) provides comprehensive GPU debugging:

- **Inspection Mode**: Records all GPU objects live on the page for inspection
- **Capture Mode**: Records GPU commands used to render specific frames
- **Shader Editing**: Edit shaders live and see results immediately
- **Performance Profiling**: Plot frame times and GPU object counts over time
- **Command History**: View sequence of WebGPU calls with parameters

**Installation**:

- Chrome: Install from [Chrome Web Store](https://chromewebstore.google.com/detail/webgpu-inspector/holcbbnljhkpkjkhgkagjkhhpeochfal)
- Firefox: Install from [Firefox Add-ons](https://addons.mozilla.org/en-US/firefox/addon/webgpu-inspector/)
- Safari: Build from source via Xcode

**Usage**: Open DevTools, navigate to the "WebGPU" tab. Click "Start Capture" to record a frame, then inspect individual draw calls, resource bindings, and shader code.

### Firefox Developer Tools

Firefox includes WebGPU debugging capabilities in its standard developer tools:

- **Graphics inspector**: View render passes and pipeline states
- **Shader viewer**: Inspect WGSL shader source and compilation info
- **Performance profiler**: Measure GPU frame time contributions

**Enabling**: Firefox Nightly includes the most recent WebGPU debugging features. Open about:config and ensure `dom.webgpu.enabled` is true.

### Safari Web Inspector

Safari Technology Preview includes WebGPU debugging tools:

- **WebGPU inspector**: View resources, pipelines, and command buffers
- **Shader debugger**: Step through shader execution on supported hardware
- **Metal frame capture**: Capture and analyze Metal commands underlying WebGPU

**Usage**: Enable the Develop menu in Safari Preferences > Advanced, then use Develop > Show Web Inspector when viewing a WebGPU page.

## Best Practices

Effective GPU debugging combines multiple techniques and requires discipline to avoid common pitfalls.

### Debug vs. Release Builds

Maintain separate debug and release configurations:

**Debug builds**:

- Enable all validation layers
- Include console.log statements and visual debugging
- Use verbose error logging
- Disable optimizations for more readable generated WGSL

**Release builds**:

- Disable development-only logging
- Remove visual debugging shaders
- Enable shader optimizations
- Use minimal error handling for performance

Use environment variables or build flags to toggle configurations:

```typescript
const DEBUG = import.meta.env.MODE === "development";

if (DEBUG) {
  device.pushErrorScope("validation");
  // ... development-only validation
}
```

### Logging Strategies

Implement structured logging for GPU operations:

```typescript
class GPULogger {
  private enabled: boolean;

  constructor(enabled: boolean) {
    this.enabled = enabled;
  }

  logResourceCreation(type: string, descriptor: any) {
    if (!this.enabled) return;
    console.log(`[GPU] Creating ${type}:`, descriptor);
  }

  logCommandSubmission(label: string, commandCount: number) {
    if (!this.enabled) return;
    console.log(`[GPU] Submitting ${commandCount} commands for ${label}`);
  }

  logError(error: GPUError) {
    // Always log errors, even in production
    console.error("[GPU Error]", error.constructor.name, error.message);
  }
}

const logger = new GPULogger(DEBUG);
```

### Incremental Development

Build GPU code incrementally, verifying each step:

1. Start with minimal working shader
2. Add one feature at a time
3. Test after each addition
4. Use simulation for algorithm verification
5. Validate on GPU with visual debugging
6. Optimize only after confirming correctness

This approach isolates issues to recent changes, making debugging much easier than trying to debug a complete complex shader all at once.

### Label Everything

Use WebGPU's labeling feature extensively:

```typescript
const buffer = device.createBuffer({
  label: "Particle Position Buffer", // Always provide descriptive labels
  size: particleCount * 16,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const pipeline = device.createComputePipeline({
  label: "Particle Physics Update Pipeline",
  // ...
});

const commandEncoder = device.createCommandEncoder({
  label: "Frame 123 Command Encoder",
});
```

Labels appear in error messages and developer tools, making it immediately clear which resource or operation failed.

## Common Pitfalls

Avoid these frequent debugging mistakes:

### Over-Relying on console.log

While console.log is useful, excessive logging:

- Severely degrades performance
- Generates overwhelming output in parallel execution
- May not be available in all contexts
- Creates dependencies on TypeGPU-specific features

Use console.log for initial debugging, then migrate to visual debugging or validation for production code.

### Ignoring Asynchronous Errors

GPU operations execute asynchronously. Errors may appear later than the causative code:

```typescript
// WRONG: Error handling too late
device.createBuffer(/* invalid descriptor */);
const error = await device.popErrorScope(); // Scope wasn't pushed first!

// CORRECT: Proper error scoping
device.pushErrorScope("validation");
device.createBuffer(/* descriptor */);
const error = await device.popErrorScope();
if (error) {
  console.error("Buffer creation failed:", error);
}
```

Always use error scopes around operations that might fail.

### Missing Error Handling

Production code must handle device loss and out-of-memory errors gracefully:

```typescript
// WRONG: No error handling
const texture = device.createTexture(descriptor);

// CORRECT: Handle potential failure
device.pushErrorScope("out-of-memory");
const texture = device.createTexture(descriptor);
const error = await device.popErrorScope();

if (error) {
  // Fallback strategy
  const reducedTexture = device.createTexture(fallbackDescriptor);
}

// Also handle device loss
device.lost.then((info) => {
  console.error("Device lost, cannot continue");
  displayErrorMessageToUser();
});
```

### Not Testing on Multiple Devices

GPU behavior varies across hardware and drivers. Always test on:

- Different GPU vendors (NVIDIA, AMD, Intel)
- Different device tiers (integrated vs. discrete GPUs)
- Mobile devices (if targeting mobile)
- Different browsers and OS combinations

Issues that don't appear on your development GPU may affect users on different hardware.

### Debugging Optimized Shaders

GPU shader compilers aggressively optimize, potentially transforming code in ways that obscure bugs:

```typescript
// Your code
const result = expensiveCalculation(x) * 0.0;

// Compiler optimizes to
const result = 0.0;

// Bug in expensiveCalculation() never triggers because it's eliminated
```

When debugging, temporarily disable optimizations or add code that prevents optimization:

```typescript
// Force calculation to execute
const temp = expensiveCalculation(x);
console.log("Debug:", temp); // Prevents optimization
const result = temp * 0.0;
```

## Conclusion

Effective GPU debugging requires understanding the unique constraints of GPU execution and mastering a diverse toolkit: validation layers catch errors early, console.log and simulation enable traditional debugging workflows, visual techniques reveal spatial patterns, and browser developer tools provide deep inspection capabilities.

The key is selecting the right tool for each situation. Use validation layers and error scopes for catching API usage mistakes. Use console.log and tgpu.simulate() for algorithm development and edge case testing. Use visual debugging for rendering and spatial algorithms. Use browser developer tools for performance analysis and command inspection.

Most importantly, build defensively: label resources, implement proper error handling, test incrementally, and validate on multiple devices. Debugging GPU code is challenging, but with systematic approaches and the right tools, you can efficiently diagnose and fix even complex issues.

## Further Resources

- [WebGPU Inspector Extension](https://github.com/brendan-duncan/webgpu_inspector) - Cross-browser debugging extension
- [WebGPU Error Handling Best Practices](https://toji.dev/webgpu-best-practices/error-handling.html) - Comprehensive error handling guide
- [WebGPU Debugging Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-debugging.html) - Practical debugging techniques
- [WGSL Specification](https://www.w3.org/TR/WGSL/) - Official WebGPU Shading Language specification
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/) - Official TypeGPU documentation including TGSL guide
- [Chrome GPU Crash Testing](chrome://gpucrash) - Manually test device loss recovery
- [WebGPU Device Loss Best Practices](https://toji.dev/webgpu-best-practices/device-loss) - Handling catastrophic GPU failures
