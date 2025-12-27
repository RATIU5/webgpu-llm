---
title: Command Encoders and Submission
sidebar:
  order: 30
---

## Overview

WebGPU employs a command buffer architecture where GPU operations are recorded and then executed. Unlike immediate-mode graphics APIs, WebGPU follows a deferred execution model: you create a command encoder, record commands into it, finish to produce a command buffer, and submit to the device queue.

This architecture mirrors Vulkan, Metal, and Direct3D 12, providing explicit control over GPU workloads while enabling driver optimizations through reduced synchronization overhead.

## Command Buffer Lifecycle

The lifecycle progresses through four stages:

**Recording**: Create a command encoder and call methods to add operations. Validation occurs during recording.

**Finishing**: Call `encoder.finish()` to produce a GPUCommandBuffer. The encoder becomes invalid after this.

**Submission**: Pass command buffers to `queue.submit()`. Commands enter the device timeline for execution.

**Execution**: The GPU processes commands. Command buffers are single-use and cannot be resubmitted.

## Creating Command Encoders

```javascript
const encoder = device.createCommandEncoder({
  label: "main-render-encoder",
});
```

The `label` property appears in error messages and profiling tools. Create a fresh encoder for each frame—the overhead is minimal, and it keeps code simpler.

## Render Pass Commands

Render passes are bounded regions that specify which textures receive output and how their contents are handled.

### Beginning a Render Pass

```javascript
const renderPassDescriptor = {
  label: "main-render-pass",
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    clearValue: { r: 0.0, g: 0.0, b: 0.2, a: 1.0 },
    loadOp: "clear",
    storeOp: "store",
  }],
  depthStencilAttachment: {
    view: depthTexture.createView(),
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
  },
};

const passEncoder = encoder.beginRenderPass(renderPassDescriptor);
```

### Color Attachments

Each color attachment specifies:
- **view**: GPUTextureView to render into
- **resolveTarget**: Optional, for multisample resolve
- **clearValue**: Color to clear if loadOp is 'clear'
- **loadOp**: 'clear' or 'load' (preserve existing contents)
- **storeOp**: 'store' or 'discard'

Using 'clear' for loadOp and 'discard' for storeOp when you don't need previous or final contents improves performance on tile-based GPUs by eliminating memory bandwidth.

Multiple color attachments (up to device limit, typically 8) enable deferred rendering:

```javascript
colorAttachments: [
  { view: albedoView, loadOp: "clear", storeOp: "store" },
  { view: normalView, loadOp: "clear", storeOp: "store" },
  { view: depthView, loadOp: "clear", storeOp: "store" },
]
```

### Depth-Stencil Attachment

Only one depth-stencil attachment per pass:
- **depthLoadOp/depthStoreOp**: Depth buffer handling
- **stencilLoadOp/stencilStoreOp**: Stencil buffer handling
- **depthClearValue**: Typically 1.0 or 0.0
- **stencilClearValue**: Typically 0
- **depthReadOnly/stencilReadOnly**: Mark as read-only

### Drawing Commands

```javascript
passEncoder.setPipeline(renderPipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.setVertexBuffer(0, vertexBuffer);
passEncoder.setIndexBuffer(indexBuffer, "uint16");

// Non-indexed drawing
passEncoder.draw(vertexCount, instanceCount, firstVertex, firstInstance);

// Indexed drawing
passEncoder.drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);

// Indirect drawing (parameters from GPU buffer)
passEncoder.drawIndirect(indirectBuffer, indirectOffset);
passEncoder.drawIndexedIndirect(indirectBuffer, indirectOffset);
```

End the pass before encoding more commands:

```javascript
passEncoder.end();
```

After `end()`, the pass encoder is invalid.

## Compute Pass Commands

Compute passes execute general-purpose GPU computation.

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
passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, computeBindGroup);

// Dispatch workgroups
passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);

// Or indirect dispatch
passEncoder.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);

passEncoder.end();
```

The workgroup count defines a 3D grid. Total invocations = workgroup counts × workgroup size (from `@workgroup_size` in WGSL).

## Copy Commands

Copy operations transfer data between resources without shaders, often using specialized hardware (DMA engines).

### Buffer to Buffer

```javascript
encoder.copyBufferToBuffer(
  sourceBuffer,
  sourceOffset,     // bytes, multiple of 4
  destinationBuffer,
  destinationOffset, // bytes, multiple of 4
  size              // bytes, multiple of 4
);
```

### Texture to Texture

```javascript
encoder.copyTextureToTexture(
  { texture: sourceTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
  { texture: destinationTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
  { width: 512, height: 512, depthOrArrayLayers: 1 }
);
```

### Buffer to/from Texture

```javascript
// Upload texture data
encoder.copyBufferToTexture(
  { buffer: stagingBuffer, offset: 0, bytesPerRow: 256 * 4, rowsPerImage: 256 },
  { texture: targetTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
  { width: 256, height: 256, depthOrArrayLayers: 1 }
);

// Readback texture data
encoder.copyTextureToBuffer(
  { texture: sourceTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
  { buffer: readbackBuffer, offset: 0, bytesPerRow: 256 * 4, rowsPerImage: 256 },
  { width: 256, height: 256, depthOrArrayLayers: 1 }
);
```

## Finishing and Submission

```javascript
const commandBuffer = encoder.finish({
  label: "main-command-buffer",
});

device.queue.submit([commandBuffer]);
```

The encoder becomes invalid after `finish()`. Command buffers are single-use—submit once, then discard.

### Multiple Command Buffers

Batch command buffers in a single submit for efficiency:

```javascript
// Less optimal: many small submits
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  // ... record work ...
  device.queue.submit([encoder.finish()]);
}

// Better: batch in single submit
const commandBuffers = [];
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  // ... record work ...
  commandBuffers.push(encoder.finish());
}
device.queue.submit(commandBuffers);
```

Command buffers in a single submit execute in array order.

### Direct Queue Writes

For convenience, write data directly without command encoders:

```javascript
device.queue.writeBuffer(targetBuffer, bufferOffset, arrayBufferData, dataOffset, size);

device.queue.writeTexture(
  { texture: targetTexture, mipLevel: 0 },
  imageData,
  { bytesPerRow: 256 * 4, rowsPerImage: 256 },
  { width: 256, height: 256, depthOrArrayLayers: 1 }
);
```

### Waiting for Completion

```javascript
await device.queue.onSubmittedWorkDone();
console.log("All GPU work completed");
```

## Synchronization

WebGPU handles synchronization through implicit barriers between passes and command buffers.

### Resource State Tracking

- **Exclusive write access**: A resource used for writing cannot be read or written elsewhere in the same pass
- **Multiple read access**: Resources can be read from multiple bind groups simultaneously
- **Subresource independence**: Different mip levels or array layers can be used independently

### Write-After-Read Safety

Within a command buffer, writes followed by reads are automatically synchronized:

```javascript
encoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, 256);
const computePass = encoder.beginComputePass();
computePass.setBindGroup(0, bindGroupUsingDstBuffer);
computePass.dispatchWorkgroups(1);
computePass.end();
```

Between passes, all writes complete before subsequent reads:

```javascript
const renderPass = encoder.beginRenderPass(/* writes to texture */);
renderPass.end();

const computePass = encoder.beginComputePass();
// Can safely read the texture written above
computePass.end();
```

## Complete Render Loop

```javascript
async function init() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({ device, format, alphaMode: "opaque" });

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
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  function frame() {
    const encoder = device.createCommandEncoder({ label: "frame-encoder" });

    const pass = encoder.beginRenderPass({
      label: "main-render-pass",
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    pass.setPipeline(pipeline);
    pass.draw(3);
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  frame();
}

init();
```

---

The command buffer model provides explicit control over GPU workloads. Create fresh encoders per frame, always end passes before finishing the encoder, batch submissions for efficiency, and let resources remain valid until GPU execution completes. This architecture enables driver optimizations while giving developers precise control over rendering and compute operations.
