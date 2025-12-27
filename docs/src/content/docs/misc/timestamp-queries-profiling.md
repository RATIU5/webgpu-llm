---
title: Timestamp Queries and Profiling
sidebar:
  order: 40
---

## Overview

GPU performance profiling requires specialized mechanisms because GPU operations execute asynchronously on separate hardware. WebGPU provides timestamp queries to measure GPU execution time with high precision.

:::danger[CPU Timing Won't Work]
```typescript
const startTime = performance.now();
device.queue.submit([commandEncoder.finish()]);
const endTime = performance.now();  // WRONG!
```

This only measures command submission time, not GPU execution. The GPU may not even start executing until milliseconds later.
:::

## The timestamp-query Feature

Timestamp queries are an optional feature requiring explicit request:

```typescript title="Request timestamp-query feature" {5-9}
const adapter = await navigator.gpu.requestAdapter();

if (!adapter.features.has("timestamp-query")) {
  console.warn("Timestamp queries not supported");
  return;
}

const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query"],
});
```

:::caution[Security Consideration]
Timestamp queries are optional because high-precision timing can potentially be exploited for side-channel attacks. Not all hardware supports them.
:::

## Query Sets

Query sets are GPU-managed objects that store timestamp values:

```typescript title="Create query set"
const querySet = device.createQuerySet({
  type: "timestamp",
  count: 2,  // Start and end timestamps
  label: "render-pass-timing",
});
```

### Writing Timestamps

Timestamps are written via the `timestampWrites` parameter when beginning passes:

```typescript title="Timestamp writes in compute pass" {2-6}
const passEncoder = commandEncoder.beginComputePass({
  timestampWrites: {
    querySet: querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  },
});

passEncoder.setPipeline(computePipeline);
passEncoder.dispatchWorkgroups(64, 64);
passEncoder.end();
```

```typescript title="Timestamp writes in render pass"
const passEncoder = commandEncoder.beginRenderPass({
  colorAttachments: [{ /* ... */ }],
  timestampWrites: {
    querySet: querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  },
});
```

## Reading Query Results

Query results must be resolved to a buffer, then mapped for CPU reading:

```typescript title="Resolve and read timestamps" {2-8,15-20,23-30}
// Create buffers for resolution
const queryBuffer = device.createBuffer({
  size: 2 * 8,  // 2 timestamps × 8 bytes (64-bit)
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});

const resultBuffer = device.createBuffer({
  size: 2 * 8,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// After recording commands, resolve queries
commandEncoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
commandEncoder.copyBufferToBuffer(queryBuffer, 0, resultBuffer, 0, 16);

// Submit and wait
device.queue.submit([commandEncoder.finish()]);
await device.queue.onSubmittedWorkDone();

// Read results
await resultBuffer.mapAsync(GPUMapMode.READ);
const timestamps = new BigInt64Array(resultBuffer.getMappedRange());

const durationNs = timestamps[1] - timestamps[0];
const durationMs = Number(durationNs) / 1_000_000;
console.log(`GPU execution: ${durationMs.toFixed(3)}ms`);

resultBuffer.unmap();
```

:::note[64-bit Timestamps]
Timestamps are returned as `BigInt` (64-bit integers in nanoseconds). Use `BigInt64Array` to read them correctly.
:::

## TypeGPU Query API

TypeGPU provides higher-level abstractions for profiling:

### Performance Callbacks

```typescript title="Automatic performance measurement" {4-8}
const pipeline = root["~unstable"]
  .withCompute(computeShader)
  .createPipeline()
  .withPerformanceCallback((start, end) => {
    const durationMs = Number(end - start) / 1_000_000;
    console.log(`Execution: ${durationMs.toFixed(3)}ms`);
  });

pipeline.execute();
```

### Manual Query Sets

```typescript title="TypeGPU query set"
const querySet = root.createQuerySet("timestamp", 2);

const pipeline = root["~unstable"]
  .withCompute(computeShader)
  .createPipeline()
  .withTimestampWrites({
    querySet: querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  });

pipeline.execute();

if (querySet.available) {
  querySet.resolve();
  const timestamps = await querySet.read();
  const durationNs = Number(timestamps[1] - timestamps[0]);
  console.log(`Execution: ${durationNs / 1_000_000}ms`);
}
```

## Profiling Patterns

### Frame Time Breakdown

```typescript title="Profile multiple passes"
const querySet = root.createQuerySet("timestamp", 10);

// Shadow pass: indices 0-1
shadowPipeline.withTimestampWrites({
  querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
});

// Geometry pass: indices 2-3
geometryPipeline.withTimestampWrites({
  querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3,
});

// Lighting pass: indices 4-5
lightingPipeline.withTimestampWrites({
  querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5,
});

// Execute all passes, then read
if (querySet.available) {
  querySet.resolve();
  const times = await querySet.read();

  console.log(`Shadow: ${toMs(times[1] - times[0])}ms`);
  console.log(`Geometry: ${toMs(times[3] - times[2])}ms`);
  console.log(`Lighting: ${toMs(times[5] - times[4])}ms`);
}

function toMs(ns) {
  return (Number(ns) / 1_000_000).toFixed(2);
}
```

### Statistical Analysis

```typescript title="Collect performance statistics"
class PerformanceTracker {
  private samples: number[] = [];
  private maxSamples = 100;

  addSample(durationMs: number) {
    this.samples.push(durationMs);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
  }

  getStats() {
    if (this.samples.length === 0) return null;

    const sorted = [...this.samples].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);

    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: sum / sorted.length,
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
    };
  }
}
```

:::tip[Performance Tips]
1. **Reuse query sets**: Create once, reuse across frames
2. **Batch queries**: Use one large query set for multiple passes
3. **Limit read frequency**: Don't read every frame in production
4. **Statistical analysis**: Single measurements can be misleading due to variance
:::

## Browser Developer Tools

### Chrome GPU Profiling

- **`chrome://gpu`**: View GPU capabilities and driver info
- **DevTools Performance tab**: GPU process activity and frame timeline

### Firefox Graphics Tools

- **`about:support`**: Graphics section shows GPU and feature status
- **Performance profiler**: GPU timing in development builds

## Complete Example

```typescript title="Full profiling setup" {17-20,28-31}
import tgpu from "typegpu";

const root = await tgpu.init();

class GPUProfiler {
  private frameTimes: number[] = [];

  recordFrame(durationMs: number) {
    this.frameTimes.push(durationMs);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
  }

  getAvgFPS() {
    if (this.frameTimes.length === 0) return 0;
    const avg = this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length;
    return 1000 / avg;
  }
}

const profiler = new GPUProfiler();

const pipeline = root["~unstable"]
  .withCompute(computeShader)
  .createPipeline()
  .withPerformanceCallback((start, end) => {
    profiler.recordFrame(Number(end - start) / 1_000_000);
  });

function renderLoop() {
  pipeline.execute();

  if (frameCount % 60 === 0) {
    console.log(`FPS: ${profiler.getAvgFPS().toFixed(1)}`);
  }

  requestAnimationFrame(renderLoop);
}

renderLoop();
```

:::caution[Production Considerations]
- **Development**: Enable comprehensive profiling
- **Production**: Disable or use selective profiling
- **User-triggered**: Provide opt-in profiling via keyboard shortcut

```typescript
const ENABLE_PROFILING = import.meta.env.DEV ||
  localStorage.getItem("enable-gpu-profiling");
```
:::

## Resources

:::note[Official Documentation]
- [WebGPU Query Sets](https://gpuweb.github.io/gpuweb/#query-sets)
- [TypeGPU Performance](https://docs.swmansion.com/TypeGPU/)
- [WebGPU Fundamentals: Timing](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html)
:::
