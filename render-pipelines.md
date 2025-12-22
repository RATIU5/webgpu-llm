# Render Pipelines

Render pipelines form the backbone of modern GPU graphics programming, defining how raw vertex data transforms into the pixels you see on screen. In WebGPU and TypeGPU, understanding render pipelines is essential for creating any visual output, from simple 2D shapes to complex 3D scenes with lighting, textures, and post-processing effects.

A render pipeline is a configured sequence of programmable and fixed-function stages that processes geometric data through vertex transformation, primitive assembly, rasterization, fragment shading, and finally outputs pixels to render targets. Unlike the older graphics APIs, WebGPU requires explicit pipeline configuration, giving developers fine-grained control over every aspect of rendering while enabling powerful optimizations.

This document provides a comprehensive guide to render pipelines in both native WebGPU and the TypeGPU library, covering pipeline stages, configuration options, practical examples, and best practices to help you build efficient rendering systems.

## Pipeline Stages

The graphics rendering pipeline consists of several distinct stages, each responsible for specific transformations of your data. Understanding these stages is crucial for effective GPU programming.

### Vertex Stage: Position Transformation and Attribute Processing

The vertex stage is the first programmable stage in the render pipeline. It processes each vertex independently, transforming vertex positions from model space to clip space and preparing per-vertex data for subsequent stages.

**Primary Responsibilities:**
- Transform vertex positions using model-view-projection matrices
- Calculate per-vertex lighting values (for Gouraud shading)
- Generate texture coordinates for mapping
- Compute tangent-space vectors for normal mapping
- Pass through vertex attributes like colors, normals, and custom data

The vertex shader receives input from vertex buffers and can access uniform data and textures through bind groups. Each invocation processes one vertex, and the output includes at minimum a position in clip space (required for rasterization) and any additional data needed by the fragment shader.

Vertex shaders execute in parallel across all vertices, with no guaranteed execution order. This means vertex shaders cannot share data between invocations, ensuring maximum parallelization on the GPU.

**Clip Space Coordinates:** The vertex shader must output positions in clip space, a 4D homogeneous coordinate system where x, y, and z components are divided by w during the perspective divide. After this division, coordinates in the range [-1, 1] for x and y (and [0, 1] for z in WebGPU) are visible in the viewport.

### Primitive Assembly: Triangles, Lines, and Points

After vertex processing, the primitive assembly stage groups vertices into geometric primitives based on the specified topology. This fixed-function stage doesn't execute any programmable code but determines how vertices connect to form renderable shapes.

**Topology Types:**
- **triangle-list**: Every three consecutive vertices form a triangle (most common for 3D meshes)
- **triangle-strip**: Each vertex after the first two forms a triangle with the previous two vertices, sharing edges for efficiency
- **line-list**: Pairs of vertices form individual line segments
- **line-strip**: Connected line segments where each vertex connects to the next
- **point-list**: Each vertex renders as an individual point

The choice of topology significantly impacts both performance and vertex buffer organization. Triangle strips can reduce vertex count by sharing vertices between adjacent triangles, while indexed triangle lists offer more flexibility in mesh representation.

**Culling and Front Face:** During primitive assembly, face culling can eliminate primitives based on their winding order. The front face orientation (clockwise or counter-clockwise) determines which direction is considered "forward-facing." Back-face culling discards primitives facing away from the camera, roughly halving the fragment workload for closed meshes.

### Rasterization: Converting Primitives to Fragments

Rasterization is the process of determining which pixels (or more precisely, which samples) are covered by each primitive. This fixed-function stage converts continuous geometric primitives into discrete fragments that will be processed by the fragment shader.

**Rasterization Process:**
1. **Clipping**: Primitives are clipped against the view frustum, generating new vertices at intersection points
2. **Perspective Division**: Clip space coordinates are divided by their w component to produce normalized device coordinates
3. **Viewport Transform**: Normalized coordinates map to actual pixel coordinates in the framebuffer
4. **Scan Conversion**: The rasterizer determines which pixels are covered by the primitive
5. **Attribute Interpolation**: Vertex shader outputs are interpolated across the primitive using perspective-correct interpolation

For each generated fragment, the rasterizer interpolates vertex attributes based on the fragment's position within the primitive. This interpolation respects perspective, ensuring that textures and other attributes appear correctly even on surfaces receding into the distance.

**Multisampling**: Modern GPUs can generate multiple samples per pixel for antialiasing. With multisampling enabled, the rasterizer may generate multiple fragments per pixel, each representing a different sample location.

### Fragment Stage: Color Computation

The fragment shader is the second programmable stage, executing once for each fragment generated by the rasterizer. While the vertex shader determines where things appear, the fragment shader determines what color they are.

**Primary Responsibilities:**
- Calculate final pixel colors using lighting models (Phong, PBR, etc.)
- Sample textures and apply texture filtering
- Implement effects like bump mapping, parallax mapping, and reflections
- Perform alpha testing and transparency calculations
- Apply procedural patterns and noise functions
- Calculate fog, atmospheric effects, and post-processing

Fragment shaders receive interpolated values from the vertex shader and can access textures, samplers, and uniform data through bind groups. Each fragment shader invocation operates independently and must output at minimum a color value for each render target.

**Early Depth Testing**: Modern GPUs can perform depth testing before fragment shader execution in some cases. If a fragment is guaranteed to fail the depth test, the fragment shader can be skipped entirely, saving significant computation. However, if the fragment shader writes to depth or uses discard operations, early depth testing may be disabled.

**Multiple Render Targets**: Fragment shaders can output to multiple render targets simultaneously, enabling techniques like deferred rendering where geometry information is written to several textures in a single pass.

### Output Merging: Blending and Depth Testing

The final stage combines fragment shader outputs with the current contents of render targets. This fixed-function stage handles several critical operations:

**Depth Testing**: Compares the fragment's depth value against the value stored in the depth buffer. Common comparison functions include:
- **less**: Fragment passes if its depth is less than the stored value (standard for most 3D rendering)
- **less-equal**: Useful when rendering the same geometry multiple times
- **greater**: Used for reverse depth buffers, which can improve precision
- **always**: Fragment always passes (depth testing effectively disabled)

**Stencil Testing**: Uses a stencil buffer to mask pixels, enabling effects like:
- Outlining objects by rendering a slightly larger version to the stencil
- Creating portals and mirrors
- Implementing shadow volumes
- Restricting rendering to specific screen regions

**Blending**: Combines the fragment color with the existing framebuffer color using configurable blend operations. Common blend modes include:
- **Opaque**: No blending, fragment color replaces framebuffer color
- **Alpha Blending**: `srcColor * srcAlpha + dstColor * (1 - srcAlpha)` for transparency
- **Additive**: `srcColor + dstColor` for effects like particles and lights
- **Multiplicative**: `srcColor * dstColor` for modulation effects

**Write Masks**: Control which color channels are written to the framebuffer, enabling selective updates to red, green, blue, or alpha channels independently.

## WebGPU Render Pipeline

WebGPU provides explicit, low-level control over render pipeline configuration through the GPURenderPipeline interface. Understanding the native WebGPU API provides insight into what's happening under the hood, even when using higher-level libraries like TypeGPU.

### Creating GPURenderPipeline

WebGPU offers two methods for creating render pipelines, both accepting the same descriptor format but differing in their execution model.

#### device.createRenderPipeline()

The synchronous creation method blocks execution until the pipeline is ready:

```javascript
const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
        module: vertexShaderModule,
        entryPoint: 'main',
        buffers: vertexBufferLayouts
    },
    fragment: {
        module: fragmentShaderModule,
        entryPoint: 'main',
        targets: [{
            format: 'bgra8unorm'
        }]
    },
    primitive: {
        topology: 'triangle-list'
    }
});
```

This method is straightforward but can cause frame drops if called during rendering, as pipeline compilation can take several milliseconds.

#### device.createRenderPipelineAsync()

The asynchronous method returns a Promise, allowing pipeline creation to occur in the background:

```javascript
const pipelinePromise = device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: { /* ... */ },
    fragment: { /* ... */ },
    primitive: { /* ... */ }
});

// Continue other work...

const pipeline = await pipelinePromise;
```

This approach is preferred for production applications, especially when creating multiple pipelines or when creating pipelines in response to user actions. You can initiate pipeline creation during loading screens or idle periods, then use the compiled pipeline when needed.

#### GPURenderPipelineDescriptor

The descriptor object configures every aspect of the pipeline. Key properties include:

- **layout**: Either a GPUPipelineLayout object or 'auto' to derive the layout from shader bindings
- **vertex**: Configuration for the vertex stage (required)
- **fragment**: Configuration for the fragment stage (optional, but required for visual output)
- **primitive**: Primitive assembly configuration
- **depthStencil**: Depth and stencil testing configuration (optional)
- **multisample**: Multisampling configuration (optional)
- **label**: A debugging label for error messages and profiling

Using `layout: 'auto'` is convenient for simple cases but prevents pipeline compatibility validation at creation time. Explicit layouts are recommended for complex applications with multiple pipelines sharing bind group layouts.

### Shader Modules

Shader modules contain the compiled WGSL (WebGPU Shading Language) code that executes on the GPU. Each shader module can contain multiple entry points, allowing you to package related shaders together.

#### device.createShaderModule()

Create shader modules from WGSL source code:

```javascript
const shaderModule = device.createShaderModule({
    label: 'Triangle Shaders',
    code: `
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec4f,
        };

        @vertex
        fn vertexMain(@location(0) pos: vec2f,
                      @location(1) color: vec4f) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4f(pos, 0.0, 1.0);
            output.color = color;
            return output;
        }

        @fragment
        fn fragmentMain(@location(0) color: vec4f) -> @location(0) vec4f {
            return color;
        }
    `
});
```

#### WGSL Vertex and Fragment Shaders

WGSL is a modern shading language designed specifically for WebGPU. It features strong typing, memory safety guarantees, and straightforward syntax inspired by Rust and C-family languages.

**Vertex Shader Requirements:**
- Must output a `@builtin(position)` value of type `vec4f` in clip space
- Can output additional values at numbered locations for the fragment shader
- Can access vertex attributes via `@location` parameters
- Can access uniforms and storage buffers via `@group` and `@binding` attributes

**Fragment Shader Requirements:**
- Must accept inputs matching the vertex shader outputs (by location number)
- Must output colors for each render target at corresponding locations
- Can discard fragments using the `discard` statement
- Can output depth values using `@builtin(frag_depth)`

**Inter-Stage Variables**: Data passed from vertex to fragment shaders must match by location number and type. The interpolation type can be controlled:
- **perspective**: Default, perspective-correct interpolation
- **linear**: Linear interpolation in screen space
- **flat**: No interpolation, value from the provoking vertex

#### Entry Points

Entry points are functions marked with `@vertex`, `@fragment`, or `@compute` attributes. When creating a pipeline, you specify which entry point to use for each stage:

```javascript
vertex: {
    module: shaderModule,
    entryPoint: 'vertexMain'  // Must match the @vertex function name
}
```

This allows a single shader module to contain multiple related shaders, reducing the number of module objects you need to manage.

### Vertex State

The vertex state configures how vertex data flows from buffers into the vertex shader. This is one of the most complex parts of pipeline configuration but also one of the most important for performance.

#### Vertex Buffer Layouts

Vertex buffer layouts describe the structure of data in vertex buffers:

```javascript
vertex: {
    module: shaderModule,
    entryPoint: 'main',
    buffers: [
        {
            arrayStride: 24,  // Bytes between consecutive vertices
            stepMode: 'vertex',
            attributes: [
                {
                    shaderLocation: 0,
                    offset: 0,
                    format: 'float32x3'  // Position (12 bytes)
                },
                {
                    shaderLocation: 1,
                    offset: 12,
                    format: 'float32x3'  // Normal (12 bytes)
                }
            ]
        }
    ]
}
```

The `arrayStride` is the number of bytes from the start of one vertex to the start of the next. This must account for any padding required for alignment.

#### Attribute Formats

WebGPU supports numerous vertex attribute formats:

**Floating Point Formats:**
- `float32`, `float32x2`, `float32x3`, `float32x4`: Standard 32-bit floats
- `float16x2`, `float16x4`: Half-precision floats (smaller memory footprint)

**Integer Formats:**
- `sint32`, `sint32x2`, `sint32x3`, `sint32x4`: Signed 32-bit integers
- `uint32`, `uint32x2`, `uint32x3`, `uint32x4`: Unsigned 32-bit integers
- `sint8x2`, `sint8x4`, `sint16x2`, `sint16x4`: Smaller integer formats

**Normalized Formats:**
- `unorm8x2`, `unorm8x4`: Unsigned normalized (0-255 maps to 0.0-1.0)
- `snorm8x2`, `snorm8x4`: Signed normalized (-128-127 maps to -1.0-1.0)
- `unorm16x2`, `unorm16x4`: Higher precision normalized values

Normalized formats are excellent for vertex colors and texture coordinates, providing good precision while using less memory than floats.

#### Step Modes (vertex, instance)

The `stepMode` determines when the GPU advances to the next element in the buffer:

- **'vertex'**: Advance for each vertex (standard per-vertex attributes)
- **'instance'**: Advance for each instance when using instanced rendering

Instanced rendering allows you to draw multiple copies of the same geometry with different per-instance data (transforms, colors, etc.):

```javascript
buffers: [
    {
        arrayStride: 12,
        stepMode: 'vertex',
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
    },
    {
        arrayStride: 16,
        stepMode: 'instance',  // Per-instance data
        attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }]
    }
]
```

This enables efficient rendering of particle systems, vegetation, repeated architectural elements, and other scenarios with many similar objects.

### Fragment State

The fragment state configures the fragment shader and its outputs. While optional in the pipeline descriptor, it's required for any visible rendering.

#### Targets Array

The targets array specifies the format and blending configuration for each render target:

```javascript
fragment: {
    module: shaderModule,
    entryPoint: 'main',
    targets: [
        {
            format: 'bgra8unorm',
            blend: {
                color: {
                    srcFactor: 'src-alpha',
                    dstFactor: 'one-minus-src-alpha',
                    operation: 'add'
                },
                alpha: {
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                    operation: 'add'
                }
            },
            writeMask: GPUColorWrite.ALL
        }
    ]
}
```

Multiple render targets enable advanced techniques like deferred rendering:

```javascript
targets: [
    { format: 'rgba16float' },  // Position
    { format: 'rgba16float' },  // Normal
    { format: 'rgba8unorm' },   // Albedo
    { format: 'rgba8unorm' }    // Material properties
]
```

#### Color Formats

Common render target formats include:

- **bgra8unorm / rgba8unorm**: Standard 8-bit per channel color (most common for display)
- **rgba16float**: 16-bit floating point per channel (HDR rendering)
- **rgba32float**: 32-bit floating point per channel (maximum precision)
- **rgb10a2unorm**: 10 bits per RGB channel, 2 bits alpha (better precision than 8-bit)

The format must match the texture format of the render target you'll bind during rendering.

#### Blend State

Blending configuration controls how fragment colors combine with existing framebuffer values:

**Common Blend Configurations:**

Standard alpha blending (transparency):
```javascript
blend: {
    color: {
        srcFactor: 'src-alpha',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
    },
    alpha: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
    }
}
```

Additive blending (particles, lights):
```javascript
blend: {
    color: {
        srcFactor: 'one',
        dstFactor: 'one',
        operation: 'add'
    },
    alpha: {
        srcFactor: 'one',
        dstFactor: 'one',
        operation: 'add'
    }
}
```

Premultiplied alpha:
```javascript
blend: {
    color: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
    },
    alpha: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
    }
}
```

### Primitive State

The primitive state configures how vertices assemble into primitives and how those primitives are processed.

#### Topology: triangle-list, line-list, point-list

```javascript
primitive: {
    topology: 'triangle-list',  // Most common for 3D meshes
    stripIndexFormat: undefined,
    frontFace: 'ccw',
    cullMode: 'back'
}
```

**Topology Options:**
- **point-list**: Render vertices as points (useful for particle systems)
- **line-list**: Pairs of vertices form lines (wireframe, debug visualization)
- **line-strip**: Connected line segments (paths, trajectories)
- **triangle-list**: Standard triangle rendering
- **triangle-strip**: Connected triangles sharing edges

#### Strip Index Format

For strip topologies (line-strip, triangle-strip), you can specify a special index value to restart the strip:

```javascript
primitive: {
    topology: 'triangle-strip',
    stripIndexFormat: 'uint32'  // Use 0xFFFFFFFF to restart the strip
}
```

This allows multiple disconnected strips in a single draw call.

#### Front Face and Cull Mode

Control face culling to avoid rendering hidden geometry:

```javascript
primitive: {
    topology: 'triangle-list',
    frontFace: 'ccw',  // Counter-clockwise is front-facing
    cullMode: 'back'   // Cull back-facing triangles
}
```

**Front Face Options:**
- **'ccw'**: Counter-clockwise winding is front-facing (OpenGL/WebGL convention)
- **'cw'**: Clockwise winding is front-facing

**Cull Mode Options:**
- **'none'**: Render both sides (required for transparent objects, flat surfaces)
- **'front'**: Cull front-facing triangles (uncommon, used for special effects)
- **'back'**: Cull back-facing triangles (standard for closed meshes)

### Depth/Stencil State

Depth and stencil testing control which fragments contribute to the final image based on their depth values and stencil buffer contents.

#### Depth Testing Configuration

```javascript
depthStencil: {
    format: 'depth24plus',
    depthWriteEnabled: true,
    depthCompare: 'less',
    stencilFront: {
        compare: 'always',
        failOp: 'keep',
        depthFailOp: 'keep',
        passOp: 'keep'
    },
    stencilBack: {
        compare: 'always',
        failOp: 'keep',
        depthFailOp: 'keep',
        passOp: 'keep'
    },
    stencilReadMask: 0xFFFFFFFF,
    stencilWriteMask: 0xFFFFFFFF,
    depthBias: 0,
    depthBiasSlopeScale: 0,
    depthBiasClamp: 0
}
```

**Common Depth Formats:**
- **depth16unorm**: 16-bit depth (lower precision, smaller memory)
- **depth24plus**: At least 24-bit depth (recommended for most use cases)
- **depth32float**: 32-bit floating point depth (maximum precision)
- **depth24plus-stencil8**: Combined depth and stencil buffer

**Depth Compare Functions:**
- **'never'**: Fragments never pass
- **'less'**: Pass if fragment depth < stored depth (standard)
- **'equal'**: Pass if depths are equal
- **'less-equal'**: Pass if fragment depth <= stored depth
- **'greater'**: Pass if fragment depth > stored depth (reverse depth)
- **'not-equal'**: Pass if depths differ
- **'greater-equal'**: Pass if fragment depth >= stored depth
- **'always'**: Fragments always pass (depth test disabled)

#### Stencil Operations

Stencil testing enables masking and multi-pass rendering techniques:

```javascript
stencilFront: {
    compare: 'equal',           // When to pass the test
    failOp: 'keep',            // What to do if stencil test fails
    depthFailOp: 'keep',       // What to do if depth test fails
    passOp: 'increment-clamp'  // What to do if both tests pass
}
```

**Stencil Operations:**
- **'keep'**: Don't change the stencil value
- **'zero'**: Set stencil value to 0
- **'replace'**: Replace with reference value
- **'invert'**: Bitwise invert the stencil value
- **'increment-clamp'**: Increment, clamp to max
- **'decrement-clamp'**: Decrement, clamp to 0
- **'increment-wrap'**: Increment with wrapping
- **'decrement-wrap'**: Decrement with wrapping

## TypeGPU Render Pipelines

TypeGPU provides a type-safe, ergonomic API for creating render pipelines that leverages TypeScript's type system to catch errors at compile time rather than runtime.

### Builder Pattern: root.withVertex().withFragment()

TypeGPU uses a fluent builder pattern starting from a root object. The type system ensures you call methods in the correct order and with compatible types:

```typescript
const pipeline = root
    .withVertex(vertexShader, {
        position: positionAttribute,
        normal: normalAttribute,
        texCoord: texCoordAttribute
    })
    .withFragment(fragmentShader, {
        color: { format: 'bgra8unorm' }
    })
    .withDepthStencil({
        format: 'depth24plus',
        depthCompare: 'less',
        depthWriteEnabled: true
    })
    .withPrimitive({
        topology: 'triangle-list',
        cullMode: 'back'
    })
    .createPipeline();
```

The builder pattern provides several advantages:
- Method chaining creates readable, self-documenting code
- The type system prevents invalid configurations
- IDE autocomplete guides you through available options
- Missing required configuration causes compile-time errors

### TgpuVertexFn and TgpuFragmentFn

TypeGPU represents shaders as typed functions that the library can analyze to generate WGSL code and validate pipeline configuration:

```typescript
const vertexShader: TgpuVertexFn = (
    position: vec3f,
    normal: vec3f,
    texCoord: vec2f
) => {
    // Shader logic here
    return {
        position: vec4f(position, 1.0),
        worldNormal: normal,
        uv: texCoord
    };
};

const fragmentShader: TgpuFragmentFn = (
    worldNormal: vec3f,
    uv: vec2f
) => {
    // Fragment shader logic
    return {
        color: vec4f(1.0, 0.0, 0.0, 1.0)
    };
};
```

The type system automatically validates that:
- Fragment shader inputs match vertex shader outputs
- Shader parameter names correspond to vertex attributes
- Return types are compatible with render target formats

### tgpu.vertexLayout() for Attributes

Vertex layouts define the structure of vertex data and create typed attribute references:

```typescript
const vertexLayout = tgpu.vertexLayout({
    position: { format: 'float32x3', offset: 0 },
    normal: { format: 'float32x3', offset: 12 },
    texCoord: { format: 'float32x2', offset: 24 }
}, {
    arrayStride: 32,
    stepMode: 'vertex'
});

// Create vertex buffer matching this layout
const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});
```

The layout object provides typed attribute references that can be passed to `withVertex()`, ensuring compile-time validation of attribute compatibility.

### createPipeline() and Lazy Creation

The `createPipeline()` method completes the builder chain and returns a configured pipeline object:

```typescript
const pipeline = root
    .withVertex(vertexShader, attributes)
    .withFragment(fragmentShader, targets)
    .createPipeline();
```

TypeGPU pipelines are created lazily - the underlying GPURenderPipeline is not created until the pipeline is first used in a render pass. This enables:
- Faster application startup (no upfront compilation cost)
- Automatic pipeline caching (reusing compiled pipelines when possible)
- Deferred error reporting (compilation errors occur when the pipeline is used)

For applications requiring predictable performance, you can force immediate compilation by using the pipeline immediately after creation or by calling implementation-specific compilation methods.

### Drawing with TypeGPU

TypeGPU provides a chainable interface for binding resources and issuing draw calls:

```typescript
pipeline
    .with(vertexLayout, vertexBuffer)
    .with(bindGroup)
    .withColorAttachment({
        color: {
            view: colorTextureView,
            loadOp: 'clear',
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            storeOp: 'store'
        }
    })
    .draw(vertexCount, instanceCount);
```

#### pipeline.with(bindGroup).with(vertexBuffer)

The `with()` method binds resources to the pipeline:

```typescript
// Bind a uniform buffer
const uniformBindGroup = tgpu.bindGroup([uniformBuffer]);

// Bind vertex layout and buffer together
pipeline
    .with(vertexLayout, vertexBuffer)
    .with(uniformBindGroup)
    .withColorAttachment({ /* ... */ })
    .draw(vertexCount);
```

Resource bindings are validated at the type level - attempting to bind incompatible resources results in a compile-time error.

#### pipeline.draw(vertexCount, instanceCount)

The `draw()` method issues a non-indexed draw call:

```typescript
// Draw 36 vertices (12 triangles)
pipeline.draw(36);

// Draw 36 vertices, 100 instances
pipeline.draw(36, 100);

// Draw starting from vertex 10, with offset instance index
pipeline.draw(36, 100, 10, 50);
```

For indexed drawing, use `drawIndexed()`:

```typescript
pipeline
    .with(vertexLayout, vertexBuffer)
    .withIndexBuffer(indexBuffer, 'uint16')
    .withColorAttachment({ /* ... */ })
    .drawIndexed(indexCount);
```

#### Color and Depth Attachments

Attachments specify where the pipeline renders its output:

```typescript
pipeline
    .withColorAttachment({
        color: {
            view: colorTextureView,
            loadOp: 'clear',
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
            storeOp: 'store'
        }
    })
    .withDepthStencilAttachment({
        view: depthTextureView,
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'store'
    })
    .draw(vertexCount);
```

**Load Operations:**
- **'clear'**: Clear the attachment before rendering
- **'load'**: Preserve existing contents

**Store Operations:**
- **'store'**: Save rendering results
- **'discard'**: Don't save results (useful for temporary render targets)

## Complete Example

Here's a complete example demonstrating both WebGPU and TypeGPU approaches to creating and using a render pipeline.

### WebGPU Implementation

```javascript
// Create shader module
const shaderModule = device.createShaderModule({
    label: 'Basic Triangle Shader',
    code: `
        struct VertexInput {
            @location(0) position: vec3f,
            @location(1) color: vec3f,
        };

        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec3f,
        };

        struct Uniforms {
            modelViewProjection: mat4x4f,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = uniforms.modelViewProjection * vec4f(input.position, 1.0);
            output.color = input.color;
            return output;
        }

        @fragment
        fn fragmentMain(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
        }
    `
});

// Create pipeline
const pipeline = device.createRenderPipeline({
    label: 'Basic Triangle Pipeline',
    layout: 'auto',
    vertex: {
        module: shaderModule,
        entryPoint: 'vertexMain',
        buffers: [{
            arrayStride: 24,  // 3 floats position + 3 floats color
            attributes: [
                {
                    shaderLocation: 0,
                    offset: 0,
                    format: 'float32x3'  // position
                },
                {
                    shaderLocation: 1,
                    offset: 12,
                    format: 'float32x3'  // color
                }
            ]
        }]
    },
    fragment: {
        module: shaderModule,
        entryPoint: 'fragmentMain',
        targets: [{
            format: navigator.gpu.getPreferredCanvasFormat()
        }]
    },
    primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
        frontFace: 'ccw'
    },
    depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less'
    }
});

// Create vertex buffer
const vertices = new Float32Array([
    // Position          // Color
    0.0,  0.5,  0.0,     1.0, 0.0, 0.0,  // Top vertex (red)
   -0.5, -0.5,  0.0,     0.0, 1.0, 0.0,  // Bottom left (green)
    0.5, -0.5,  0.0,     0.0, 0.0, 1.0   // Bottom right (blue)
]);

const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

// Create uniform buffer
const uniformBuffer = device.createBuffer({
    size: 64,  // mat4x4f
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

// Create bind group
const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
    }]
});

// Render
const commandEncoder = device.createCommandEncoder();
const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        storeOp: 'store'
    }],
    depthStencilAttachment: {
        view: depthTextureView,
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'store'
    }
});

renderPass.setPipeline(pipeline);
renderPass.setVertexBuffer(0, vertexBuffer);
renderPass.setBindGroup(0, bindGroup);
renderPass.draw(3);
renderPass.end();

device.queue.submit([commandEncoder.finish()]);
```

### TypeGPU Implementation

```typescript
import tgpu from 'typegpu';

// Define types
const vec3f = tgpu.vec3f;
const vec4f = tgpu.vec4f;
const mat4x4f = tgpu.mat4x4f;

// Create vertex layout
const vertexLayout = tgpu.vertexLayout({
    position: { format: 'float32x3', offset: 0 },
    color: { format: 'float32x3', offset: 12 }
}, {
    arrayStride: 24,
    stepMode: 'vertex'
});

// Define shaders
const vertexShader = tgpu.vertexFn(
    [vertexLayout.attrib.position, vertexLayout.attrib.color],
    (position: vec3f, color: vec3f) => {
        const mvp = tgpu.uniform(mat4x4f);
        return {
            position: mvp.multiply(vec4f(position, 1.0)),
            color: color
        };
    }
);

const fragmentShader = tgpu.fragmentFn(
    (color: vec3f) => {
        return { color: vec4f(color, 1.0) };
    }
);

// Create pipeline
const pipeline = root
    .withVertex(vertexShader, {
        position: vertexLayout.attrib.position,
        color: vertexLayout.attrib.color
    })
    .withFragment(fragmentShader, {
        color: { format: 'bgra8unorm' }
    })
    .withDepthStencil({
        format: 'depth24plus',
        depthCompare: 'less',
        depthWriteEnabled: true
    })
    .withPrimitive({
        topology: 'triangle-list',
        cullMode: 'back'
    })
    .createPipeline();

// Create vertex data
const vertices = new Float32Array([
    0.0,  0.5,  0.0,     1.0, 0.0, 0.0,
   -0.5, -0.5,  0.0,     0.0, 1.0, 0.0,
    0.5, -0.5,  0.0,     0.0, 0.0, 1.0
]);

// Create and fill vertex buffer
const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(vertexBuffer, 0, vertices);

// Create uniform buffer
const uniformBuffer = tgpu.buffer(mat4x4f);

// Render
pipeline
    .with(vertexLayout, vertexBuffer)
    .with(uniformBuffer)
    .withColorAttachment({
        color: {
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            storeOp: 'store'
        }
    })
    .withDepthStencilAttachment({
        view: depthTextureView,
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'store'
    })
    .draw(3);
```

## Best Practices and Common Pitfalls

### Best Practices

**1. Create Pipelines During Initialization**

Pipeline creation is expensive, often taking several milliseconds. Create all pipelines during application startup or loading screens, not during rendering:

```javascript
// Good: Create during initialization
const pipelines = await Promise.all([
    device.createRenderPipelineAsync(desc1),
    device.createRenderPipelineAsync(desc2),
    device.createRenderPipelineAsync(desc3)
]);

// Bad: Creating during render loop
function render() {
    const pipeline = device.createRenderPipeline(desc);  // Major performance hit!
    // ...
}
```

**2. Reuse Pipelines Across Frames**

Store pipeline objects and reuse them for every frame. The GPU driver caches compiled pipeline state, making subsequent uses nearly free.

**3. Use Appropriate Vertex Formats**

Choose vertex attribute formats based on your needs:
- Use `float16` instead of `float32` for attributes where precision isn't critical
- Use normalized formats (`unorm8x4`) for colors
- Pack multiple small attributes into a single vector
- Align vertex data properly (4-byte alignment minimum)

**4. Enable Face Culling**

Unless rendering transparent or double-sided geometry, always enable back-face culling to reduce fragment shader invocations by roughly 50%:

```javascript
primitive: {
    cullMode: 'back',
    frontFace: 'ccw'
}
```

**5. Order Draw Calls Front-to-Back**

For opaque geometry, sort objects front-to-back and enable early depth testing. This allows the GPU to skip fragment shading for occluded fragments:

```javascript
depthStencil: {
    format: 'depth24plus',
    depthCompare: 'less',
    depthWriteEnabled: true
}
```

**6. Batch Draw Calls**

Minimize the number of draw calls by:
- Using instanced rendering for repeated geometry
- Combining static geometry into larger vertex buffers
- Using texture atlases instead of binding individual textures
- Employing indirect drawing for GPU-driven rendering

**7. Use Async Pipeline Creation**

For production code, always use `createRenderPipelineAsync()` to avoid blocking:

```javascript
const pipelinePromise = device.createRenderPipelineAsync(descriptor);
// Continue other initialization...
const pipeline = await pipelinePromise;
```

**8. Minimize State Changes**

Sort draw calls to minimize pipeline changes, bind group changes, and vertex buffer changes. State changes have overhead, so batching similar draws together improves performance.

### Common Pitfalls

**1. Mismatched Vertex and Fragment Shader Interfaces**

The most common error is mismatched `@location` attributes between vertex shader outputs and fragment shader inputs:

```wgsl
// Vertex shader outputs
@location(0) color: vec3f,
@location(1) texCoord: vec2f,

// Fragment shader must match exactly
fn fragmentMain(@location(0) color: vec3f,
                @location(1) texCoord: vec2f) -> @location(0) vec4f
```

**2. Incorrect Vertex Buffer Stride**

The `arrayStride` must account for all data in each vertex, including padding:

```javascript
// Wrong: Missing 4 bytes of padding for alignment
buffers: [{
    arrayStride: 20,  // 3 floats (12 bytes) + 2 floats (8 bytes)
    attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x2' }
    ]
}]

// Correct: Include padding for proper alignment
buffers: [{
    arrayStride: 24,  // Padded to multiple of 4 bytes
    attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x2' }
    ]
}]
```

**3. Forgetting Depth Buffer**

When using depth testing, you must create a depth texture and attach it to the render pass:

```javascript
const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT
});

// Use in render pass
depthStencilAttachment: {
    view: depthTexture.createView(),
    depthLoadOp: 'clear',
    depthClearValue: 1.0,
    depthStoreOp: 'store'
}
```

**4. Incorrect Blend State for Transparency**

Transparent objects require correct blend configuration and must be drawn after opaque objects with depth writes disabled:

```javascript
// Transparent object pipeline
fragment: {
    targets: [{
        format: 'bgra8unorm',
        blend: {
            color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
            },
            alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
            }
        }
    }]
},
depthStencil: {
    format: 'depth24plus',
    depthWriteEnabled: false,  // Don't write depth for transparent objects
    depthCompare: 'less'       // But still test against existing depth
}
```

**5. Using Wrong Topology**

Ensure your vertex data matches the topology. Using `triangle-list` when vertices are arranged for `triangle-strip` produces incorrect results.

**6. Not Handling Resize**

When the canvas resizes, depth textures and render targets must be recreated to match the new dimensions. Mismatched sizes cause validation errors.

**7. Forgetting sRGB Color Space**

For correct color rendering, ensure your render target format and display configuration match:

```javascript
// For correct color rendering
const context = canvas.getContext('webgpu');
context.configure({
    device: device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    colorSpace: 'srgb'  // Important for correct colors
});
```

**8. Pipeline Creation Validation Failures**

Always check for pipeline creation errors, especially when using async creation:

```javascript
try {
    const pipeline = await device.createRenderPipelineAsync(descriptor);
} catch (error) {
    console.error('Pipeline creation failed:', error);
    // Handle error appropriately
}
```

Enable WebGPU error scopes for detailed error reporting during development:

```javascript
device.pushErrorScope('validation');
const pipeline = device.createRenderPipeline(descriptor);
device.popErrorScope().then(error => {
    if (error) {
        console.error('Pipeline validation error:', error.message);
    }
});
```

### Performance Tips

**1. Profile GPU Performance**

Use browser developer tools and WebGPU timestamp queries to identify bottlenecks. Focus optimization efforts on the most expensive operations.

**2. Minimize Fragment Shader Complexity**

Fragment shaders execute far more often than vertex shaders (every pixel vs. every vertex). Keep fragment shaders simple and move calculations to the vertex shader when possible.

**3. Use Mipmaps**

For textured geometry, always generate and use mipmaps. This dramatically improves cache efficiency and reduces texture sampling cost.

**4. Consider Compute Shaders**

For complex per-vertex calculations or skinning, compute shaders can be more efficient than vertex shaders, especially when the same vertices are used multiple times.

**5. Optimize Vertex Data**

Smaller vertices mean better cache utilization:
- Use quantized normals and tangents
- Pack texture coordinates into fewer bytes
- Use vertex pulling (loading vertex data in the shader) for complex scenarios

By following these best practices and avoiding common pitfalls, you'll create efficient, robust rendering systems that take full advantage of modern GPU capabilities through WebGPU and TypeGPU.
