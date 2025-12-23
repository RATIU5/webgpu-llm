---
title: Command Encoders and Submission
sidebar:
  order: 30
---

## Overview

WebGPU employs a command buffer architecture that fundamentally shapes how GPU operations are recorded and executed. Unlike immediate-mode graphics APIs where commands execute as soon as they're issued, WebGPU follows a deferred execution model. This design pattern provides several critical advantages: it enables sophisticated command validation before GPU execution, allows optimal batching and reordering of operations by the driver, reduces synchronization overhead between CPU and GPU, and provides better multi-threading support for command recording.

The command buffer model works through a three-stage pipeline. First, you create a command encoder that acts as a recording device for GPU operations. Second, you issue commands to this encoder—these commands don't execute immediately but are instead recorded into an internal command buffer. Finally, you finish the encoder to produce a GPUCommandBuffer object and submit it to the device's queue, where it enters the execution pipeline and runs on the GPU.

This architecture mirrors the design of modern graphics APIs like Vulkan, Metal, and Direct3D 12, which all recognize that explicit command buffer management gives developers fine-grained control over GPU workloads while enabling better performance through reduced driver overhead. The separation between recording and execution is particularly valuable for complex rendering scenarios where you might want to record command buffers on background threads or reuse previously recorded work across multiple frames.

## Key Concepts

### GPUCommandEncoder and Command Buffers

The GPUCommandEncoder is the primary interface for recording GPU commands in WebGPU. It's an opaque object that accumulates commands into an internal buffer without executing them. You create encoders through `device.createCommandEncoder()`, record various operations, and then finalize the encoder with `finish()` to obtain a GPUCommandBuffer.

A GPUCommandBuffer is an immutable, opaque handle representing a completed sequence of GPU commands ready for execution. Once created, command buffers cannot be modified or reused—they're single-use objects that become invalid after submission. This immutability is by design: it allows the WebGPU implementation to optimize and validate the command sequence without worrying about concurrent modifications.

### Command Buffer Lifecycle

The lifecycle of command buffers follows a strict progression through four distinct states:

**Recording**: When you create a command encoder, it enters the recording state. During this phase, you can call various encoding methods to add operations to the buffer. The encoder validates commands as you record them, generating validation errors if you attempt invalid operations like binding incompatible resources or using destroyed objects.

**Finishing**: Calling `encoder.finish()` transitions the encoder to a finished state and returns a GPUCommandBuffer. After finishing, the encoder becomes invalid and cannot be used for further recording. This finalization step performs additional validation and prepares the command sequence for submission.

**Submission**: The command buffer is passed to `queue.submit()` along with other buffers. At this point, it transitions to the device timeline where the GPU scheduler manages its execution. Command buffers execute in submission order within a single submit call, but the timing of actual GPU execution is asynchronous.

**Execution**: The GPU processes commands from the buffer, performing rendering, compute work, or data transfers. Once execution completes, the command buffer is implicitly destroyed and its resources can be reclaimed.

### Render Pass Encoding with GPURenderPassEncoder

A GPURenderPassEncoder records graphics rendering commands within a defined render pass. Render passes are bounded regions of command recording that specify which textures will receive rendered output and how their contents should be handled. The encoder provides methods for setting graphics pipeline state, binding resources, and issuing draw calls.

Render passes are fundamental to WebGPU's rendering model because they explicitly declare which attachments will be written during a sequence of draw operations. This declaration enables important optimizations like tile-based rendering on mobile GPUs, where knowing all the attachments upfront allows the GPU to keep framebuffer data in fast on-chip memory rather than writing it out to main memory after each draw call.

### Compute Pass Encoding with GPUComputePassEncoder

A GPUComputePassEncoder records compute shader dispatch commands. Compute passes are simpler than render passes because they don't involve attachments—compute shaders read from and write to buffer and texture resources through bind groups. The encoder manages compute pipeline state and issues workgroup dispatches that execute shader invocations across a specified grid.

### Copy Operations Between Resources

Command encoders support direct copy operations that transfer data between GPU resources without requiring shaders. These copies are often more efficient than shader-based transfers because they can use specialized hardware paths like DMA engines. WebGPU provides several copy operations for different resource type combinations: buffer-to-buffer, texture-to-texture, buffer-to-texture, and texture-to-buffer.

## Creating Command Encoders

Creating a command encoder is straightforward and uses the `device.createCommandEncoder()` method. This method accepts an optional descriptor object that currently supports only a label property for debugging:

```javascript
const encoder = device.createCommandEncoder({
  label: "main-render-encoder",
});
```

The label property is invaluable for debugging and profiling. When validation errors occur, WebGPU will include the label in error messages, making it much easier to identify which encoder caused the problem. Similarly, GPU profiling tools can display these labels to help you understand performance characteristics of different command sequences.

It's generally good practice to create a fresh encoder for each frame in a rendering loop rather than trying to reuse encoders. The overhead of creating an encoder is minimal, and creating new ones each frame keeps your code simpler and avoids potential issues with encoder state.

## Render Pass Commands

Render passes are initiated with `beginRenderPass()` and configured through a GPURenderPassDescriptor. This descriptor defines the structure of the render pass, including which textures to render into and how to handle their contents.

### Beginning a Render Pass with GPURenderPassDescriptor

The render pass descriptor is a rich object that controls many aspects of rendering behavior:

```javascript
const renderPassDescriptor = {
  label: "main-render-pass",
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.0, g: 0.0, b: 0.2, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    },
  ],
  depthStencilAttachment: {
    view: depthTexture.createView(),
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
    stencilClearValue: 0,
    stencilLoadOp: "clear",
    stencilStoreOp: "discard",
  },
};

const passEncoder = encoder.beginRenderPass(renderPassDescriptor);
```

### Color Attachments Configuration

Color attachments define the textures that receive color output from fragment shaders. Each attachment specifies:

- **view**: A GPUTextureView representing the texture to render into
- **resolveTarget**: Optional texture view for multisample resolve operations
- **clearValue**: The color to clear the attachment to if loadOp is 'clear'
- **loadOp**: Either 'clear' to wipe the texture before rendering, or 'load' to preserve existing contents
- **storeOp**: Either 'store' to keep rendered results, or 'discard' to throw them away

The loadOp and storeOp properties are critical for performance on tile-based GPUs. Using 'clear' for loadOp and 'discard' for storeOp when you don't need the previous or final contents can significantly improve performance by eliminating memory bandwidth.

You can have multiple color attachments (up to the device's limit, typically 8), enabling techniques like deferred rendering where you output to several textures simultaneously:

```javascript
colorAttachments: [
  { view: albedoView, loadOp: "clear", storeOp: "store" },
  { view: normalView, loadOp: "clear", storeOp: "store" },
  { view: depthView, loadOp: "clear", storeOp: "store" },
];
```

### Depth-Stencil Attachment

The depth-stencil attachment handles depth testing and stencil operations. Unlike color attachments, there can be only one depth-stencil attachment per render pass. It supports separate configuration for depth and stencil aspects:

- **depthLoadOp/depthStoreOp**: Control depth buffer handling
- **stencilLoadOp/stencilStoreOp**: Control stencil buffer handling
- **depthClearValue**: Clear value for depth (typically 1.0 or 0.0 depending on depth range)
- **stencilClearValue**: Clear value for stencil (typically 0)
- **depthReadOnly/stencilReadOnly**: Mark aspects as read-only for depth testing without writes

### Drawing Commands

Once you've set up the render pass and configured the pipeline state, you issue drawing commands to execute geometry rendering:

```javascript
// Set the rendering pipeline
passEncoder.setPipeline(renderPipeline);

// Bind uniform buffers and textures
passEncoder.setBindGroup(0, bindGroup);

// Set vertex buffers
passEncoder.setVertexBuffer(0, vertexBuffer);

// Set index buffer if using indexed drawing
passEncoder.setIndexBuffer(indexBuffer, "uint16");

// Non-indexed drawing
passEncoder.draw(vertexCount, instanceCount, firstVertex, firstInstance);

// Indexed drawing
passEncoder.drawIndexed(
  indexCount,
  instanceCount,
  firstIndex,
  baseVertex,
  firstInstance,
);

// Indirect drawing (parameters come from GPU buffer)
passEncoder.drawIndirect(indirectBuffer, indirectOffset);
passEncoder.drawIndexedIndirect(indirectBuffer, indirectOffset);
```

The `draw()` command executes the vertex shader once for each vertex, with optional instancing for drawing multiple copies of geometry. `drawIndexed()` uses an index buffer to specify which vertices to process, enabling efficient reuse of vertex data. The indirect variants read draw parameters from a GPU buffer, enabling GPU-driven rendering where draw calls can be generated by compute shaders.

### Ending the Render Pass

After issuing all draw commands, you must end the render pass:

```javascript
passEncoder.end();
```

This signals that you've finished recording commands for this render pass. The encoder performs final validation and prepares the recorded commands for inclusion in the command buffer. After calling `end()`, the pass encoder becomes invalid and cannot be used further.

## Compute Pass Commands

Compute passes enable general-purpose GPU computation through compute shaders. They're simpler than render passes because they don't involve attachments or rasterization state.

### Beginning a Compute Pass

Create a compute pass encoder with `beginComputePass()`:

```javascript
const computePassDescriptor = {
  label: "particle-update-pass",
  timestampWrites: {
    querySet: timestampQuerySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  },
};

const passEncoder = encoder.beginComputePass(computePassDescriptor);
```

The descriptor is simpler than render passes, primarily supporting timestamp queries for performance measurement.

### Dispatch Workgroups

The primary command in compute passes is `dispatchWorkgroups()`, which executes compute shader invocations:

```javascript
passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, computeBindGroup);

// Dispatch a grid of workgroups
passEncoder.dispatchWorkgroups(
  workgroupCountX,
  workgroupCountY,
  workgroupCountZ,
);

// Indirect dispatch with parameters from GPU buffer
passEncoder.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
```

The workgroup count parameters define a 3D grid of workgroups to execute. Each workgroup contains a shader-defined number of invocations (specified by `@workgroup_size` in WGSL). The total number of shader invocations is the product of the workgroup counts and the workgroup size.

### Compute Pass Descriptors

While compute pass descriptors are simpler than render pass descriptors, they support timestamp queries for performance profiling:

```javascript
const timestampQuerySet = device.createQuerySet({
  type: "timestamp",
  count: 2,
});

const computePassDescriptor = {
  label: "physics-simulation",
  timestampWrites: {
    querySet: timestampQuerySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  },
};
```

This enables precise GPU timing measurements by writing timestamps at the beginning and end of the compute pass.

## Copy Commands

Command encoders provide several methods for copying data between GPU resources without using shaders.

### copyBufferToBuffer

Copies data from one buffer to another:

```javascript
encoder.copyBufferToBuffer(
  sourceBuffer,
  sourceOffset, // offset in bytes
  destinationBuffer,
  destinationOffset, // offset in bytes
  size, // number of bytes to copy
);
```

This is useful for staging data, duplicating buffers, or moving data between different buffer types. The copy operates on raw bytes, so offsets and size must be multiples of 4 bytes.

### copyTextureToTexture

Copies data between textures, useful for mipmap generation, texture arrays, or post-processing effects:

```javascript
encoder.copyTextureToTexture(
  {
    texture: sourceTexture,
    mipLevel: 0,
    origin: { x: 0, y: 0, z: 0 },
  },
  {
    texture: destinationTexture,
    mipLevel: 0,
    origin: { x: 0, y: 0, z: 0 },
  },
  {
    width: 512,
    height: 512,
    depthOrArrayLayers: 1,
  },
);
```

The textures must have compatible formats, and the copy region must fit within both textures' dimensions.

### copyBufferToTexture and copyTextureToBuffer

These operations transfer data between linear buffer layouts and texture formats:

```javascript
// Upload texture data from a buffer
encoder.copyBufferToTexture(
  {
    buffer: stagingBuffer,
    offset: 0,
    bytesPerRow: 256 * 4, // 256 pixels * 4 bytes per pixel
    rowsPerImage: 256,
  },
  {
    texture: targetTexture,
    mipLevel: 0,
    origin: { x: 0, y: 0, z: 0 },
  },
  {
    width: 256,
    height: 256,
    depthOrArrayLayers: 1,
  },
);

// Download texture data to a buffer
encoder.copyTextureToBuffer(
  {
    texture: sourceTexture,
    mipLevel: 0,
    origin: { x: 0, y: 0, z: 0 },
  },
  {
    buffer: readbackBuffer,
    offset: 0,
    bytesPerRow: 256 * 4,
    rowsPerImage: 256,
  },
  {
    width: 256,
    height: 256,
    depthOrArrayLayers: 1,
  },
);
```

These are essential for texture uploads from CPU memory and readback operations for debugging or screenshot capture.

## Command Buffer Finalization

After recording all commands, you finalize the encoder by calling `finish()`:

```javascript
const commandBuffer = encoder.finish({
  label: "main-command-buffer",
});
```

This method returns a GPUCommandBuffer object representing the complete sequence of recorded commands. The encoder becomes invalid after finishing and cannot be used for further recording.

The finish operation performs final validation of the command sequence. If any validation errors occurred during encoding, they'll typically be reported when finish() is called or when the command buffer is submitted.

Command buffers are opaque and cannot be inspected or modified after creation. They're single-use objects—once submitted to a queue, they cannot be resubmitted.

## Queue Submission

Every WebGPU device has a queue accessible through `device.queue`. This queue manages the execution of command buffers and data transfer operations.

### Submitting Command Buffers

The `submit()` method accepts an array of command buffers and schedules them for execution:

```javascript
device.queue.submit([commandBuffer]);

// Multiple command buffers execute in order
device.queue.submit([
  setupCommandBuffer,
  renderCommandBuffer,
  postProcessCommandBuffer,
]);
```

Command buffers submitted in a single call execute in the order they appear in the array. However, submission doesn't mean immediate execution—the buffers enter the device's command queue and execute asynchronously.

### Execution Ordering

WebGPU guarantees that command buffers submitted in the same `submit()` call execute in order. Commands within a command buffer also execute in the order they were recorded. However, multiple submit calls are only guaranteed to execute in submission order—there's no strict synchronization between separate submits unless you explicitly wait for completion.

```javascript
// These execute in order within this submit
device.queue.submit([buffer1, buffer2, buffer3]);

// This executes after the previous submit completes
device.queue.submit([buffer4]);
```

For fine-grained synchronization, you can use `queue.onSubmittedWorkDone()` which returns a promise that resolves when all previously submitted work has finished executing:

```javascript
await device.queue.onSubmittedWorkDone();
console.log("All GPU work completed");
```

### Write Operations

The queue also provides convenience methods for writing data directly to GPU resources without creating command encoders:

```javascript
// Write data to a buffer
device.queue.writeBuffer(
  targetBuffer,
  bufferOffset,
  arrayBufferData,
  dataOffset,
  size,
);

// Write data to a texture
device.queue.writeTexture(
  { texture: targetTexture, mipLevel: 0 },
  imageData,
  { bytesPerRow: 256 * 4, rowsPerImage: 256 },
  { width: 256, height: 256, depthOrArrayLayers: 1 },
);
```

These operations are more convenient than creating buffers and copy commands for simple data uploads, and they allow the implementation to optimize the transfer.

## Synchronization

WebGPU handles synchronization through a combination of implicit guarantees and explicit barriers.

### Write-After-Read Hazards

WebGPU automatically manages read-after-write and write-after-write hazards within a single command buffer. If you write to a resource and then read from it, the implementation ensures the write completes before the read:

```javascript
// Safe: write followed by read in same command buffer
encoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, 256);
const computePass = encoder.beginComputePass();
computePass.setBindGroup(0, bindGroupUsingDstBuffer);
computePass.dispatchWorkgroups(1);
computePass.end();
```

However, write-after-read hazards require careful attention. If you read from a resource in one operation and then write to it in another, you must ensure they're properly ordered:

```javascript
// Potential hazard: reading from texture in render pass
// then writing to it in compute pass
const renderPass = encoder.beginRenderPass({
  colorAttachments: [
    {
      view: textureView,
      loadOp: "load",
      storeOp: "store",
    },
  ],
});
// ... rendering that reads from someTexture ...
renderPass.end();

// This is safe: render pass ends before compute pass begins
const computePass = encoder.beginComputePass();
// ... compute that writes to someTexture ...
computePass.end();
```

### Implicit Barriers

WebGPU inserts implicit memory barriers between passes and between different command buffers. When a render or compute pass ends, all writes from that pass are guaranteed to be visible to subsequent passes. This means you generally don't need to worry about memory synchronization within a single command buffer—just ensure operations are properly ordered.

### Resource State Tracking

WebGPU tracks resource usage and enforces strict rules about conflicting access:

- **Exclusive write access**: If a resource is used for writing, it cannot be used for reading or writing elsewhere in the same pass
- **Multiple read access**: Resources can be read from multiple bind groups simultaneously
- **Subresource independence**: Different mipmap levels or array layers can be used independently

Violating these rules produces validation errors:

```javascript
// ERROR: Using same buffer as both uniform (read) and storage (write)
const bindGroup = device.createBindGroup({
  layout: layout,
  entries: [
    { binding: 0, resource: { buffer: myBuffer } }, // uniform buffer
    { binding: 1, resource: { buffer: myBuffer } }, // storage buffer
  ],
});
```

## Best Practices and Common Pitfalls

### Reusing Command Encoders

**Pitfall**: Attempting to reuse a command encoder after calling `finish()`.

```javascript
// WRONG
const encoder = device.createCommandEncoder();
const buffer1 = encoder.finish();
encoder.beginRenderPass(descriptor); // ERROR: encoder is invalid
```

**Best Practice**: Create a new encoder for each command buffer you need to produce.

### Forgetting to End Passes

**Pitfall**: Not calling `end()` on pass encoders before finishing the command encoder.

```javascript
// WRONG
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass(descriptor);
pass.draw(3);
// Forgot to call pass.end()!
const buffer = encoder.finish(); // ERROR
```

**Best Practice**: Always match `beginRenderPass()` or `beginComputePass()` with a corresponding `end()` call.

### Submitting Empty Command Buffers

While not an error, submitting command buffers with no actual work wastes CPU time:

```javascript
// Wasteful
const encoder = device.createCommandEncoder();
const buffer = encoder.finish();
device.queue.submit([buffer]); // Does nothing
```

**Best Practice**: Only create and submit command buffers when you have actual work to record.

### Proper Resource Lifecycle Management

Ensure resources used by command buffers remain valid until execution completes:

```javascript
// WRONG
function render() {
  const buffer = device.createBuffer(/* ... */);
  device.queue.writeBuffer(buffer, 0, data);

  const encoder = device.createCommandEncoder();
  // ... use buffer ...
  device.queue.submit([encoder.finish()]);

  buffer.destroy(); // ERROR: buffer destroyed while still in use
}
```

**Best Practice**: Keep resources alive until you're certain the GPU has finished using them, either by waiting for queue completion or by managing resource lifetimes carefully.

### Optimizing Submission Patterns

Submit command buffers in larger batches rather than many small submits:

```javascript
// Less optimal
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  // ... record work ...
  device.queue.submit([encoder.finish()]);
}

// Better
const commandBuffers = [];
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  // ... record work ...
  commandBuffers.push(encoder.finish());
}
device.queue.submit(commandBuffers);
```

### Complete Render Loop Example

Here's a complete example showing proper command encoder usage in a typical render loop:

```javascript
async function init() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "opaque",
  });

  // Create pipeline
  const pipeline = device.createRenderPipeline({
    label: "render-pipeline",
    layout: "auto",
    vertex: {
      module: device.createShaderModule({
        code: `
          @vertex
          fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            const pos = array(
              vec2f(-0.5, -0.5),
              vec2f( 0.5, -0.5),
              vec2f( 0.0,  0.5)
            );
            return vec4f(pos[i], 0.0, 1.0);
          }
        `,
      }),
      entryPoint: "vertexMain",
    },
    fragment: {
      module: device.createShaderModule({
        code: `
          @fragment
          fn fragmentMain() -> @location(0) vec4f {
            return vec4f(1.0, 0.5, 0.2, 1.0);
          }
        `,
      }),
      entryPoint: "fragmentMain",
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  function frame() {
    // Create command encoder for this frame
    const encoder = device.createCommandEncoder({
      label: "frame-encoder",
    });

    // Begin render pass
    const pass = encoder.beginRenderPass({
      label: "main-render-pass",
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    // Record rendering commands
    pass.setPipeline(pipeline);
    pass.draw(3);

    // End render pass
    pass.end();

    // Finish encoding and submit
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // Request next frame
    requestAnimationFrame(frame);
  }

  // Start render loop
  frame();
}

init();
```

This example demonstrates the complete pattern: creating an encoder per frame, recording a render pass with drawing commands, finishing the encoder, and submitting the command buffer to the queue. The render loop continues indefinitely, creating fresh command encoders for each frame.

Understanding command encoders and submission is fundamental to effective WebGPU programming. The explicit command buffer model gives you precise control over GPU workloads while enabling the implementation to optimize execution. By following best practices—creating fresh encoders per frame, properly ending passes, batching submissions, and managing resource lifetimes—you can build efficient, reliable WebGPU applications.
