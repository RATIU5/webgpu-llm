---
title: WebGPU vs WebGL Migration Guide
sidebar:
  order: 20
---

## Overview

WebGPU represents a fundamental shift in how web applications interact with graphics hardware, moving away from WebGL's OpenGL-based design toward a modern, low-level graphics API inspired by Vulkan, Metal, and Direct3D 12. While WebGL has served the web platform well since its introduction, it was designed around OpenGL ES 2.0's paradigms from 2011, which no longer align with how modern GPUs operate.

The transition from WebGL to WebGPU is not merely an incremental upgrade—it's a complete architectural reimagining. WebGPU offers explicit control over GPU resources, better performance through reduced CPU overhead, support for compute shaders enabling GPGPU (General-Purpose GPU) computations, and a more predictable cross-platform behavior. However, this power comes with increased complexity and a steeper learning curve.

Understanding when and how to migrate requires careful consideration of your project's requirements, timeline, and target audience. This guide explores the fundamental differences between these two APIs, helping you make informed decisions about adopting WebGPU.

## Architectural Differences

### State Management

**WebGL: Global State Machine**

WebGL operates as a global state machine, where settings persist across rendering operations until explicitly changed. When you bind a texture, configure blending modes, or set the active shader program, these states remain in effect for all subsequent draw calls. This approach, inherited from OpenGL, makes getting started relatively straightforward—you can issue commands sequentially and the GPU remembers your configuration.

However, global state management introduces significant challenges. It becomes easy to forget to reset a setting, leading to subtle bugs where one part of your code inadvertently affects another. State leakage is a common problem in larger WebGL applications, requiring developers to carefully manage and restore state when sharing code between different rendering systems. Additionally, the global state model makes multithreading impossible, as thread-safe access to shared global state would introduce unacceptable performance overhead.

**WebGPU: Stateless Pipeline Objects**

WebGPU eliminates global state in favor of immutable pipeline objects. A pipeline encapsulates all rendering state—vertex layout, blending configuration, depth testing, topology, and shader programs—into a single, immutable object created upfront. When you want different rendering behavior, you create a different pipeline and switch between them explicitly.

This stateless design offers several advantages:

- **Predictability**: Each pipeline is self-contained, eliminating unexpected interactions between different rendering operations
- **Performance**: GPU drivers can optimize pipelines more aggressively when they know the state is immutable
- **Parallelism**: Multiple threads can record commands to separate command buffers without worrying about state conflicts
- **Explicitness**: All configuration is visible at the point where the pipeline is created, making code more maintainable

The tradeoff is increased verbosity during setup. Where WebGL might require a few state-setting calls before drawing, WebGPU requires you to think ahead and create pipelines for each rendering configuration you'll need.

### Resource Management

**WebGL: Implicit, Driver-Managed**

WebGL handles many resource management tasks automatically. When you create a framebuffer, the driver automatically creates and attaches depth and stencil buffers if needed. Texture mipmaps can be generated with a single function call. The driver decides when to allocate GPU memory, when to synchronize between CPU and GPU, and how to batch operations for efficiency.

This implicit management reduces cognitive load for developers but sacrifices control and predictability. You cannot reliably predict when memory allocations occur or how much overhead various operations will incur. The driver makes decisions that might not align with your application's specific needs, and debugging performance issues often requires diving into driver-specific behaviors.

**WebGPU: Explicit, Application-Managed**

WebGPU requires explicit management of all GPU resources. There are no automatic depth buffers—you must create them yourself, specifying exact formats, sizes, and usage patterns. Buffer layouts must be manually calculated with proper alignment. Memory barriers and synchronization are your responsibility.

This explicit model provides several benefits:

- **Performance Control**: You decide exactly when allocations happen, enabling strategic pre-allocation to avoid runtime hitches
- **Memory Efficiency**: Precise control over resource lifetimes and reuse patterns
- **Predictability**: Behavior is consistent across different GPU vendors and driver versions
- **Debugging**: Clearer understanding of what's happening at the GPU level

The learning curve is steeper, requiring understanding of GPU memory models, alignment requirements, and synchronization primitives. However, this knowledge translates directly to better performance and more maintainable code.

### Command Model

**WebGL: Immediate Mode**

WebGL uses an immediate mode rendering model. When you call `gl.drawArrays()`, the driver processes that draw call immediately (or as soon as possible), executing it on the GPU. Each WebGL call potentially requires driver validation, state checking, and immediate submission to the GPU command queue.

This immediate execution model is intuitive and easy to debug—you can step through rendering code and see immediate results. However, it introduces significant CPU overhead. The driver must validate every call immediately, translate it to the GPU's native command format, and manage the command queue in real-time. This overhead becomes particularly problematic when issuing many draw calls.

**WebGPU: Command Buffers**

WebGPU adopts a deferred command buffer model used by modern graphics APIs. Instead of executing commands immediately, you record them into command buffers using command encoders. These encoders batch multiple operations together, and nothing is executed until you explicitly submit the command buffer to the device queue.

```javascript
// WebGPU command buffer approach
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass(renderPassDescriptor);

pass.setPipeline(pipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.setBindGroup(0, bindGroup);
pass.draw(vertexCount);
pass.end();

const commandBuffer = encoder.finish();
device.queue.submit([commandBuffer]);
```

This model offers substantial advantages:

- **Reduced CPU Overhead**: Commands are validated once during encoding, not during execution
- **Parallelism**: Multiple threads can record independent command buffers simultaneously
- **Optimization**: The driver can analyze entire command buffers and optimize execution
- **Reusability**: Command buffers can be recorded once and submitted multiple times (in some scenarios)

It's important to understand that functions like `setPipeline()` and `draw()` only add commands to the buffer—they don't execute anything. Actual GPU execution happens when you submit the command buffer to the device queue, providing a clear separation between command recording and execution.

## Feature Comparison

### Compute Shaders

**Only in WebGPU**

Perhaps the most significant feature gap between WebGL and WebGPU is compute shader support. WebGL offers only vertex and fragment shaders, limiting you to rendering-centric workloads. While clever developers have found ways to abuse fragment shaders for general computation (rendering to texture, reading back results), these workarounds are cumbersome, inefficient, and limited.

WebGPU provides first-class compute shader support, enabling true GPGPU (General-Purpose GPU) computations. Compute shaders are designed to handle arbitrary data processing, not just vertices or fragments, and can be executed in parallel by hundreds or thousands of threads.

**GPGPU Capabilities**

Compute shaders unlock entirely new categories of web applications:

- **Machine Learning**: Running neural network inference directly on the GPU
- **Physics Simulations**: Particle systems, fluid dynamics, cloth simulation
- **Data Processing**: Image processing, video encoding, large dataset analysis
- **Ray Tracing**: Custom rendering algorithms beyond rasterization
- **Cryptography**: Parallel hash computation, blockchain applications

Example WebGPU compute shader:

```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn computeMain(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * 2.0;
    }
}
```

The `@workgroup_size` attribute specifies how many threads execute together in a workgroup, enabling sophisticated parallel algorithms with shared memory and synchronization.

### Shading Languages

**WebGL: GLSL ES**

WebGL uses GLSL ES (OpenGL ES Shading Language), which has been stable and well-documented for over a decade. GLSL ES is based on C syntax with vector and matrix types built-in:

```glsl
// WebGL vertex shader (GLSL ES)
attribute vec3 position;
attribute vec2 texCoord;

uniform mat4 modelViewProjection;

varying vec2 vTexCoord;

void main() {
    gl_Position = modelViewProjection * vec4(position, 1.0);
    vTexCoord = texCoord;
}
```

Key characteristics:

- Entry point is always `main()`
- Attributes, uniforms, and varyings declared as globals
- Implicit location assignment by the compiler
- Runtime compilation by graphics driver (can vary across hardware)

**WebGPU: WGSL**

WebGPU introduces WGSL (WebGPU Shading Language), a new language designed specifically for WebGPU with syntax inspired by Rust and Swift:

```wgsl
// WebGPU vertex shader (WGSL)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) texCoord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
}

@group(0) @binding(0) var<uniform> modelViewProjection: mat4x4<f32>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = modelViewProjection * vec4<f32>(input.position, 1.0);
    output.texCoord = input.texCoord;
    return output;
}
```

**Syntax Comparison**

Key differences between GLSL and WGSL:

1. **Type Specificity**: WGSL requires explicit type parameters (e.g., `vec4<f32>` instead of `vec4`)
2. **Entry Points**: Custom function names with `@vertex`, `@fragment`, or `@compute` attributes
3. **Explicit Locations**: All bindings and locations must be explicitly specified
4. **Struct-Based I/O**: Input and output use structs with explicit layout
5. **Stricter Syntax**: WGSL is more rigid but produces more predictable results

**Shader Compilation**

GLSL shaders are compiled at runtime by graphics drivers, which can lead to inconsistent behavior across different hardware and vendors. WGSL is designed for ahead-of-time validation and compilation, ensuring consistent behavior regardless of the underlying GPU. The WebGPU implementation translates WGSL to the backend's native shader language (SPIR-V for Vulkan, MSL for Metal, HLSL for Direct3D 12).

### Modern GPU Features

**WebGPU: Modern Extensions**

WebGPU is designed around capabilities available in modern GPUs (2015+), including:

- **Compute Shaders**: Full GPGPU support
- **Storage Buffers**: Large read-write buffers in shaders
- **Storage Textures**: Direct texture writes from shaders
- **Multiple Render Targets**: Efficient deferred rendering
- **Depth Clipping Control**: Better control over depth range
- **Advanced Texture Formats**: Including compressed formats, depth-stencil combinations
- **Query Sets**: Timestamp queries, occlusion queries
- **Indirect Drawing**: GPU-driven rendering

**WebGL: Frozen at 2011**

WebGL's feature set is fundamentally limited to OpenGL ES 2.0/3.0 capabilities. While extensions provide access to some newer features, they're optional, inconsistently supported, and require fallback code. The WebGL specification is effectively frozen—major architectural improvements aren't possible without breaking compatibility.

## Coordinate System Differences

Understanding coordinate system differences is critical for successful migration, as these subtle changes can cause rendering artifacts if not properly addressed.

### Depth Range

**WebGL: -1 to 1**

WebGL inherits OpenGL's normalized device coordinate (NDC) system where the depth range spans from -1 (near plane) to 1 (far plane). This symmetric range matches OpenGL's historical design.

When creating projection matrices for WebGL, you typically use functions that assume this range:

```javascript
// WebGL perspective matrix (depth -1 to 1)
function perspective(fov, aspect, near, far) {
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
    near * far * rangeInv * 2,
    0,
  ];
}
```

**WebGPU: 0 to 1**

WebGPU adopts the depth range convention used by Direct3D, Metal, and Vulkan, where depth spans from 0 (near plane) to 1 (far plane). This asymmetric range offers several technical advantages identified in modern graphics APIs, including better depth precision distribution.

**Matrix Adjustments (perspectiveZO)**

When migrating to WebGPU, you must adjust projection matrices to output the correct depth range. Many math libraries provide "ZO" (Zero to One) variants:

```javascript
// WebGPU perspective matrix (depth 0 to 1)
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

Failing to adjust the projection matrix will cause objects to be clipped incorrectly or not appear at all.

### Y-Axis Direction

**Framebuffer Coordinates**

In WebGL (following OpenGL), the origin of framebuffer coordinates is in the bottom-left corner, with Y increasing upward. WebGPU follows the Metal convention where the origin is in the top-left corner, with Y increasing downward.

This affects:

- **Viewport Configuration**: Y coordinate interpretation
- **Scissor Rectangles**: Rectangle positioning
- **ReadPixels Operations**: Where (0, 0) is located

**Texture Coordinates**

Texture coordinate systems also differ. WebGL/OpenGL traditionally loads images with the first pixel in the bottom-left corner. However, many developers load images from the top-left (matching image file formats), resulting in flipped textures that require workarounds.

WebGPU standardizes on the top-left origin matching Direct3D and Metal, aligning with how most image formats and developers naturally think about image data. This means:

- Images load more intuitively without flipping
- Texture sampling coordinates match framebuffer coordinates
- Render-to-texture operations are more straightforward

**Migration Consideration**

If your WebGL code samples from framebuffers or performs pixel-based operations, you may need coordinate adjustments:

```javascript
// Flipping Y coordinate when migrating
const webgpuY = 1.0 - webglY;
```

### Clip Space

**NDC Differences**

While both APIs use normalized device coordinates (NDC), the differences are:

| Coordinate                | WebGL   | WebGPU  |
| ------------------------- | ------- | ------- |
| X Range                   | -1 to 1 | -1 to 1 |
| Y Range                   | -1 to 1 | -1 to 1 |
| Z Range                   | -1 to 1 | 0 to 1  |
| Y Direction (NDC)         | Up      | Up      |
| Y Direction (Framebuffer) | Up      | Down    |

The Z range difference is the most significant, requiring projection matrix changes as discussed above.

**Viewport Transformation**

The transformation from NDC to window coordinates differs due to the Y-axis direction change. In WebGL, NDC Y of -1 maps to the bottom of the viewport. In WebGPU, NDC Y of -1 maps to the top of the viewport after accounting for the flipped framebuffer coordinate system.

## Performance Comparison

### CPU Overhead Reduction

WebGPU dramatically reduces CPU overhead compared to WebGL:

- **Validation**: Command recording validates once, not per draw call
- **State Management**: No state tracking overhead
- **Driver Translation**: More direct mapping to native GPU commands
- **Batch Submission**: Commands submitted in efficient batches

Benchmarks show **50-70% reduction in CPU time** for draw-call-heavy applications when migrating from WebGL to WebGPU.

### Batching Efficiency

WebGL encourages batching because each draw call has significant overhead. WebGPU's lower per-call overhead makes instancing and indirect drawing more practical:

```javascript
// WebGPU efficient multi-draw
pass.setPipeline(pipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.setBindGroup(0, bindGroup);

// Multiple draws with minimal overhead
for (let i = 0; i < objectCount; i++) {
  pass.setBindGroup(1, objectBindGroups[i]);
  pass.draw(vertexCount, 1, 0, 0);
}
```

Rather than constantly recreating and rebinding small buffers, WebGPU encourages creating one large buffer and using different offsets for different draw calls, significantly increasing performance.

### Multi-threading Potential

WebGPU's command buffer model enables true multi-threaded rendering. Multiple threads can record independent command buffers in parallel:

```javascript
// Parallel command recording (Web Workers)
const workers = createWorkerPool();
const commandBuffers = await Promise.all(
  chunks.map((chunk) => workers.execute(recordRenderCommands, chunk)),
);

device.queue.submit(commandBuffers);
```

This parallelism is impossible in WebGL due to its global state model.

## When to Migrate

### Good Candidates

**Compute-Heavy Applications**

If your application needs GPGPU capabilities—machine learning inference, physics simulations, data processing—WebGPU is the clear choice. These workloads are impossible or severely limited in WebGL.

**Performance-Critical Projects**

Applications with many draw calls, complex scenes, or tight performance budgets benefit significantly from WebGPU's reduced overhead. Examples include:

- Advanced 3D visualization tools
- Real-time multiplayer games
- CAD/modeling applications
- Scientific simulations

**New Projects**

Starting fresh with WebGPU avoids migration pain and positions your application for the future. The learning investment pays off in better architecture and performance.

**Control Requirements**

If you need explicit control over GPU resources, memory management, or command execution, WebGPU provides the necessary tools.

### Stay with WebGL

**Broad Browser Support Needed**

WebGL works on virtually every device with a web browser, including older hardware. WebGPU requires modern browsers (Chrome 113+, Edge 113+, Safari 18+) and relatively recent GPUs. If you need maximum compatibility, WebGL remains the safer choice.

**Simple 2D Graphics**

For straightforward 2D rendering, sprite-based games, or simple visualizations, WebGL's immediate mode is often simpler and adequate. WebGPU's complexity isn't justified unless you need its advanced features.

**Existing Stable Codebases**

Mature, working WebGL applications don't necessarily benefit from migration. If performance is adequate and features are sufficient, the migration cost may outweigh benefits. Focus migration efforts on bottlenecks rather than wholesale rewrites.

**Limited Development Resources**

WebGPU's steeper learning curve requires more expertise and development time. Small teams or projects with tight deadlines might not justify the investment.

## Migration Strategies

### Gradual Migration

**Hybrid Approach**

You can run WebGL and WebGPU side-by-side in the same application, migrating incrementally:

```javascript
// Feature detection and progressive enhancement
if (navigator.gpu) {
  renderer = await createWebGPURenderer();
} else {
  renderer = createWebGLRenderer();
}
```

This approach allows:

- Migrating performance-critical components first
- Maintaining WebGL fallback for compatibility
- Learning WebGPU incrementally
- Reducing risk through phased rollout

**Feature-by-Feature**

Migrate individual rendering systems independently:

1. Start with compute shaders (new functionality)
2. Move particle systems (performance benefit)
3. Convert main rendering pipeline (largest impact)
4. Migrate post-processing effects (moderate complexity)
5. Update UI rendering last (lowest priority)

### Complete Rewrite

**When It Makes Sense**

Full rewrites are appropriate when:

- The existing codebase is legacy or poorly structured
- You're making significant architectural changes anyway
- The team has strong WebGPU expertise
- Performance requirements demand optimization throughout

**Planning Considerations**

- **Timeline**: Budget 2-3x initial estimates for first WebGPU project
- **Learning Curve**: Invest in training and experimentation
- **Testing**: Comprehensive testing across GPU vendors and platforms
- **Fallback**: Maintain WebGL version during transition
- **Documentation**: Document architecture decisions for team knowledge

## Common Migration Issues

### Projection Matrix Changes

**Problem**: Objects don't render or appear clipped incorrectly.

**Cause**: WebGL's -1 to 1 depth range vs WebGPU's 0 to 1 range.

**Solution**: Use `perspectiveZO` matrix functions or adjust your projection matrix calculation:

```javascript
// Adjustment factor for existing WebGL projection matrices
const webglToWebGPUMatrix = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1,
];

const webgpuProjection = multiply(webglToWebGPUMatrix, webglProjection);
```

### Texture Coordinate Flipping

**Problem**: Textures appear upside-down.

**Cause**: Different Y-axis origins between WebGL and WebGPU.

**Solution**:

Option 1 - Flip texture coordinates in shader:

```wgsl
// In fragment shader
let flippedTexCoord = vec2<f32>(texCoord.x, 1.0 - texCoord.y);
let color = textureSample(myTexture, mySampler, flippedTexCoord);
```

Option 2 - Flip image data during upload:

```javascript
// Flip image before uploading
function flipImageVertically(imageData) {
  const { width, height, data } = imageData;
  const flipped = new Uint8Array(data.length);

  for (let y = 0; y < height; y++) {
    const sourceRow = (height - 1 - y) * width * 4;
    const destRow = y * width * 4;
    flipped.set(data.slice(sourceRow, sourceRow + width * 4), destRow);
  }

  return flipped;
}
```

### Depth Buffer Configuration

**Problem**: Depth testing doesn't work correctly.

**Cause**: WebGPU requires explicit depth buffer creation and configuration.

**Solution**: Create and configure depth texture explicitly:

```javascript
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const renderPassDescriptor = {
  colorAttachments: [
    {
      /* ... */
    },
  ],
  depthStencilAttachment: {
    view: depthTexture.createView(),
    depthLoadOp: "clear",
    depthStoreOp: "store",
    depthClearValue: 1.0,
  },
};
```

Note: WebGPU depth clears to 1.0 (far plane) vs WebGL's 1.0 or -1.0 depending on convention.

### Shader Translation

**Problem**: GLSL shaders don't work in WebGPU.

**Cause**: Different shading languages (GLSL vs WGSL).

**Solution**: Manual translation or automated tools (with careful review):

```glsl
// WebGL GLSL
precision mediump float;
varying vec2 vTexCoord;
uniform sampler2D uTexture;

void main() {
    gl_FragColor = texture2D(uTexture, vTexCoord);
}
```

```wgsl
// WebGPU WGSL
@group(0) @binding(0) var myTexture: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

@fragment
fn fragmentMain(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(myTexture, mySampler, texCoord);
}
```

Key translation points:

- `varying` → `@location` inputs
- `uniform sampler2D` → separate `texture_2d` and `sampler` with bindings
- `texture2D()` → `textureSample()`
- `gl_FragColor` → return value with `@location(0)`

## Code Examples

### Side-by-Side: Triangle Rendering

**WebGL:**

```javascript
// WebGL triangle
const canvas = document.querySelector("canvas");
const gl = canvas.getContext("webgl");

const vertexShaderSource = `
    attribute vec2 position;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
    }
`;

const fragmentShaderSource = `
    precision mediump float;
    void main() {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
`;

// Compile shaders
const vertexShader = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vertexShader, vertexShaderSource);
gl.compileShader(vertexShader);

const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fragmentShader, fragmentShaderSource);
gl.compileShader(fragmentShader);

// Link program
const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);
gl.useProgram(program);

// Create buffer
const vertices = new Float32Array([0.0, 0.5, -0.5, -0.5, 0.5, -0.5]);

const buffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

// Configure attribute
const positionLoc = gl.getAttribLocation(program, "position");
gl.enableVertexAttribArray(positionLoc);
gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

// Draw
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT);
gl.drawArrays(gl.TRIANGLES, 0, 3);
```

**WebGPU:**

```javascript
// WebGPU triangle
const canvas = document.querySelector("canvas");
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu");

const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const shaderModule = device.createShaderModule({
  code: `
        @vertex
        fn vertexMain(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(position, 0.0, 1.0);
        }

        @fragment
        fn fragmentMain() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    `,
});

const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: {
    module: shaderModule,
    entryPoint: "vertexMain",
    buffers: [
      {
        arrayStride: 8,
        attributes: [
          {
            shaderLocation: 0,
            offset: 0,
            format: "float32x2",
          },
        ],
      },
    ],
  },
  fragment: {
    module: shaderModule,
    entryPoint: "fragmentMain",
    targets: [{ format }],
  },
  primitive: {
    topology: "triangle-list",
  },
});

const vertices = new Float32Array([0.0, 0.5, -0.5, -0.5, 0.5, -0.5]);

const vertexBuffer = device.createBuffer({
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    },
  ],
});

pass.setPipeline(pipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.draw(3);
pass.end();

device.queue.submit([encoder.finish()]);
```

### Key Differences Illustrated

1. **Initialization**: WebGPU requires async adapter/device request
2. **Shaders**: Combined in single module with named entry points
3. **Pipeline**: Created upfront with all state
4. **Buffers**: Explicit usage flags and write operations
5. **Drawing**: Command encoder → render pass → submit pattern

## Best Practices and Common Pitfalls

### Best Practices

**1. Pipeline Caching**

Create pipelines during initialization, not during rendering:

```javascript
// Good: Create once
const pipelines = {
  opaque: await device.createRenderPipelineAsync(opaqueDesc),
  transparent: await device.createRenderPipelineAsync(transparentDesc),
};

// Bad: Creating during render loop
function render() {
  const pipeline = device.createRenderPipeline(desc); // DON'T!
  // ...
}
```

**2. Buffer Reuse**

Use large buffers with dynamic offsets instead of many small buffers:

```javascript
// Good: Single large buffer with offsets
const uniformBuffer = device.createBuffer({
  size: 256 * maxObjects, // 256 bytes per object
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

pass.setBindGroup(0, bindGroup, [objectIndex * 256]); // Dynamic offset
```

**3. Explicit Synchronization**

Understand when GPU operations complete:

```javascript
// Writing to buffer
device.queue.writeBuffer(buffer, 0, data);

// This submits immediately—queue operations don't need explicit submit
// But command buffers do:
device.queue.submit([commandBuffer]);

// If you need to read results, use mapping:
await buffer.mapAsync(GPUMapMode.READ);
const result = new Float32Array(buffer.getMappedRange());
buffer.unmap();
```

**4. Validation During Development**

Enable validation layers during development for better error messages (browser DevTools often provide this).

### Common Pitfalls

**1. Forgetting Explicit Bindings**

Every resource needs explicit `@group` and `@binding` declarations that match JavaScript code exactly.

**2. Alignment Requirements**

Uniform buffers have strict alignment (typically 256 bytes). Padding is required:

```javascript
// Wrong: 64 bytes won't work
const size = 64;

// Right: Align to 256 bytes
const size = Math.ceil(64 / 256) * 256;
```

**3. Texture Format Mismatches**

Ensure texture formats match between creation and pipeline configuration.

**4. Asynchronous Pipeline Creation**

Use `createRenderPipelineAsync()` for initial pipeline creation to avoid blocking, but understand pipeline creation timing.

**5. Command Encoder Reuse**

Command encoders cannot be reused after `finish()`. Create a new encoder for each frame.

## Conclusion

WebGPU represents the future of graphics on the web, offering modern GPU capabilities, better performance, and explicit control over graphics hardware. The migration from WebGL requires understanding fundamental architectural differences—from stateless pipelines to coordinate system changes—but the benefits for compute-heavy, performance-critical applications are substantial.

Choose WebGPU for new projects that need cutting-edge features, performance, or compute capabilities. Stick with WebGL for maximum compatibility or simple graphics needs. For existing applications, evaluate migration on a case-by-case basis, potentially using hybrid approaches to gain benefits while maintaining compatibility.

The web platform is still in transition—WebGL isn't going anywhere soon, but WebGPU adoption is growing rapidly. Understanding both APIs and their tradeoffs positions you to make the right choice for your specific needs.

---

## Sources

- [Migrating from WebGL to WebGPU - MY.GAMES](https://medium.com/my-games-company/migrating-from-webgl-to-webgpu-057ae180f896)
- [From WebGL to WebGPU | Chrome for Developers](https://developer.chrome.com/docs/web-platform/webgpu/from-webgl-to-webgpu)
- [WebGPU from WebGL - WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-from-webgl.html)
- [WebGL vs. WebGPU Explained | Three.js Roadmap](https://threejsroadmap.com/blog/webgl-vs-webgpu-explained)
- [Migration from WebGL to WebGPU - Dmitrii Ivashchenko](https://gitnation.com/contents/migration-from-webgl-to-webgpu)
- [WebGPU vs. WebGL: What are the main differences? - Design4Real](https://design4real.de/en/webgpu-vs-webgl/)
- [WebGPU API - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [From GLSL to WGSL: the future of shaders on the Web - Damien Seguin](https://dmnsgn.me/blog/from-glsl-to-wgsl-the-future-of-shaders-on-the-web/)
- [Porting WebGL shaders to WebGPU - Ashley's blog](https://www.construct.net/en/blogs/ashleys-blog-2/porting-webgl-shaders-webgpu-1576)
- [Coordinate systems - gpuweb Issue #416](https://github.com/gpuweb/gpuweb/issues/416)
