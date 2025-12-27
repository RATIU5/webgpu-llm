---
title: Command Encoders and Submission
sidebar:
  order: 30
---

## Overview

WebGPU employs a command buffer architecture where GPU operations are recorded and then executed. Unlike immediate-mode graphics APIs, WebGPU follows a deferred execution model: you create a command encoder, record commands into it, finish to produce a command buffer, and submit to the device queue.

:::note[Architecture]
This architecture mirrors Vulkan, Metal, and Direct3D 12, providing explicit control over GPU workloads while enabling driver optimizations through reduced synchronization overhead.
:::

## Command Buffer Lifecycle

| Stage | Description |
|-------|-------------|
| **Recording** | Create a command encoder and call methods to add operations |
| **Finishing** | Call `encoder.finish()` to produce a GPUCommandBuffer |
| **Submission** | Pass command buffers to `queue.submit()` |
| **Execution** | GPU processes commands; buffers are single-use |

## Creating Command Encoders

```javascript title="Create command encoder"
const encoder = device.createCommandEncoder({
  label: "main-render-encoder",
});
```

:::tip
The `label` property appears in error messages and profiling tools. Create a fresh encoder for each frame—the overhead is minimal, and it keeps code simpler.
:::

## Render Pass Commands

Render passes are bounded regions that specify which textures receive output and how their contents are handled.

### Beginning a Render Pass

```javascript title="Render pass with depth" {3-9,10-15}
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

| Property | Description |
|----------|-------------|
| `view` | GPUTextureView to render into |
| `resolveTarget` | Optional, for multisample resolve |
| `clearValue` | Color to clear if loadOp is 'clear' |
| `loadOp` | `'clear'` or `'load'` (preserve existing) |
| `storeOp` | `'store'` or `'discard'` |

:::tip[Performance on Tile-Based GPUs]
Using `'clear'` for loadOp and `'discard'` for storeOp when you don't need previous or final contents improves performance by eliminating memory bandwidth.
:::

Multiple color attachments (up to 8) enable deferred rendering:

```javascript title="Multiple render targets"
colorAttachments: [
  { view: albedoView, loadOp: "clear", storeOp: "store" },
  { view: normalView, loadOp: "clear", storeOp: "store" },
  { view: depthView, loadOp: "clear", storeOp: "store" },
]
```

### Depth-Stencil Attachment

Only one depth-stencil attachment per pass:

| Property | Description |
|----------|-------------|
| `depthLoadOp/depthStoreOp` | Depth buffer handling |
| `stencilLoadOp/stencilStoreOp` | Stencil buffer handling |
| `depthClearValue` | Typically 1.0 or 0.0 |
| `stencilClearValue` | Typically 0 |
| `depthReadOnly/stencilReadOnly` | Mark as read-only |

### Drawing Commands

```javascript title="Drawing commands"
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

:::caution
End the pass before encoding more commands. After `end()`, the pass encoder is invalid.

```javascript
passEncoder.end();
```
:::

## Compute Pass Commands

Compute passes execute general-purpose GPU computation.

```javascript title="Compute pass with timestamp" {2-6}
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

:::note
The workgroup count defines a 3D grid. Total invocations = workgroup counts × workgroup size (from `@workgroup_size` in WGSL).
:::

## Copy Commands

Copy operations transfer data between resources without shaders, often using specialized hardware (DMA engines).

### Buffer to Buffer

```javascript title="Buffer copy"
encoder.copyBufferToBuffer(
  sourceBuffer,
  sourceOffset,     // bytes, multiple of 4
  destinationBuffer,
  destinationOffset, // bytes, multiple of 4
  size              // bytes, multiple of 4
);
```

### Texture to Texture

```javascript title="Texture copy"
encoder.copyTextureToTexture(
  { texture: sourceTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
  { texture: destinationTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
  { width: 512, height: 512, depthOrArrayLayers: 1 }
);
```

### Buffer to/from Texture

```javascript title="Texture upload and readback"
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

```javascript title="Finish and submit"
const commandBuffer = encoder.finish({
  label: "main-command-buffer",
});

device.queue.submit([commandBuffer]);
```

:::danger[Single-Use Buffers]
The encoder becomes invalid after `finish()`. Command buffers are single-use—submit once, then discard.
:::

### Multiple Command Buffers

:::tip[Batch Submissions]
Batch command buffers in a single submit for efficiency:

```javascript
// ✗ Less optimal: many small submits
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  // ... record work ...
  device.queue.submit([encoder.finish()]);
}

// ✓ Better: batch in single submit
const commandBuffers = [];
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  // ... record work ...
  commandBuffers.push(encoder.finish());
}
device.queue.submit(commandBuffers);
```

Command buffers in a single submit execute in array order.
:::

### Direct Queue Writes

For convenience, write data directly without command encoders:

```javascript title="Direct queue writes"
device.queue.writeBuffer(targetBuffer, bufferOffset, arrayBufferData, dataOffset, size);

device.queue.writeTexture(
  { texture: targetTexture, mipLevel: 0 },
  imageData,
  { bytesPerRow: 256 * 4, rowsPerImage: 256 },
  { width: 256, height: 256, depthOrArrayLayers: 1 }
);
```

### Waiting for Completion

```javascript title="Wait for GPU"
await device.queue.onSubmittedWorkDone();
console.log("All GPU work completed");
```

## Synchronization

WebGPU handles synchronization through implicit barriers between passes and command buffers.

### Resource State Tracking

| Rule | Description |
|------|-------------|
| Exclusive write | Resource used for writing cannot be read/written elsewhere in same pass |
| Multiple read | Resources can be read from multiple bind groups simultaneously |
| Subresource independence | Different mip levels or array layers can be used independently |

### Write-After-Read Safety

Within a command buffer, writes followed by reads are automatically synchronized:

```javascript title="Automatic synchronization"
encoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, 256);
const computePass = encoder.beginComputePass();
computePass.setBindGroup(0, bindGroupUsingDstBuffer);
computePass.dispatchWorkgroups(1);
computePass.end();
```

Between passes, all writes complete before subsequent reads:

```javascript title="Pass-to-pass synchronization"
const renderPass = encoder.beginRenderPass(/* writes to texture */);
renderPass.end();

const computePass = encoder.beginComputePass();
// Can safely read the texture written above
computePass.end();
```

## Complete Render Loop

```javascript title="Full render loop example" {29-40,42-54}
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
