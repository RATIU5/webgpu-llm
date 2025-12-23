---
title: Timestamp Queries and Profiling
sidebar:
  order: 40
---

GPU performance profiling is essential for understanding and optimizing the execution of graphics and compute workloads. Unlike CPU code where simple timing functions suffice, GPU operations require specialized mechanisms to accurately measure execution time. WebGPU provides timestamp queries as a powerful tool for measuring GPU performance with high precision, enabling developers to identify bottlenecks, optimize shader code, and ensure smooth rendering experiences.

This comprehensive guide covers everything you need to know about timestamp queries in WebGPU, from basic concepts to advanced profiling patterns, including integration with TypeGPU's high-level APIs.

## Understanding GPU Timing

### Why CPU Timing Doesn't Work for GPU

One of the most common mistakes developers make when profiling GPU code is attempting to use CPU-based timing methods like `performance.now()` or `Date.now()`. While these functions work perfectly for CPU operations, they fail to accurately measure GPU execution time for several critical reasons:

**Asynchronous Execution Model**: The GPU operates independently from the CPU on a separate timeline. When you submit commands to the GPU through WebGPU, the CPU doesn't wait for those commands to complete before continuing execution. Instead, commands are queued and executed asynchronously. By the time you measure elapsed time on the CPU, you're only measuring how long it took to submit the commands to the queue, not how long the GPU actually spent executing them.

**Command Buffering and Batching**: WebGPU batches commands into command buffers that are submitted to the GPU queue. The CPU-side API calls return immediately after queuing these commands, long before the GPU begins processing them. This means CPU timestamps would only measure the overhead of command recording, not actual GPU work.

**Driver Overhead**: The graphics driver introduces additional layers of abstraction between your code and the GPU hardware. Command submission involves driver processing, validation, and scheduling that happens on the CPU side but doesn't reflect actual GPU execution time.

**Parallel Execution**: Modern GPUs can execute multiple operations in parallel across thousands of shader cores. CPU timing has no visibility into this parallel execution model and cannot accurately measure when work actually starts and completes on the GPU hardware.

### Asynchronous Execution

The asynchronous nature of GPU execution creates a fundamental timing challenge. Consider this typical WebGPU workflow:

```typescript
const startTime = performance.now();

const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(computePipeline);
passEncoder.dispatchWorkgroups(64, 64);
passEncoder.end();
device.queue.submit([commandEncoder.finish()]);

const endTime = performance.now();
console.log(`Time: ${endTime - startTime}ms`); // WRONG!
```

This code appears to measure GPU execution time, but it actually only measures how long the CPU took to record and submit commands. The GPU might not even start executing the workgroup dispatches until milliseconds or even frames later. The measured time is typically in the microseconds range, while actual GPU execution might take milliseconds.

### Query Sets for Measurement

WebGPU solves this timing problem through **query sets** - special GPU-managed resources that can record precise timestamps directly on the GPU timeline. Query sets allow you to insert timestamp markers at specific points in your GPU command stream. These timestamps are recorded by the GPU hardware itself as commands execute, providing accurate measurements of actual GPU execution time.

Query sets work by:

1. **Creating markers**: You specify points in your command stream where timestamps should be recorded
2. **GPU-side recording**: As the GPU processes commands, it writes timestamp values to the query set when it reaches each marker
3. **Resolution**: After GPU execution completes, you resolve the query results into a GPU buffer
4. **Reading results**: Finally, you map the buffer to CPU-accessible memory to read the timestamp values

This approach ensures timestamps reflect actual GPU execution, not CPU-side overhead.

## The timestamp-query Feature

Timestamp queries are not available by default in WebGPU. They are exposed as an optional feature that must be explicitly requested. This design decision addresses several concerns:

**Security and Privacy**: High-precision timing can potentially be exploited for side-channel attacks or fingerprinting. By making timestamp queries opt-in, browsers can apply appropriate security policies and user consent mechanisms.

**Hardware Support**: Not all GPU hardware supports timestamp queries with the same precision or capabilities. The feature flag allows graceful fallback on unsupported hardware.

**Performance Overhead**: Even minimal profiling overhead may be undesirable in production applications. Explicit feature requests make this trade-off visible to developers.

### Feature Detection

Before requesting a device with timestamp query support, you should check if the adapter supports this feature:

```typescript
const adapter = await navigator.gpu.requestAdapter();

if (!adapter) {
  throw new Error("WebGPU not supported");
}

if (adapter.features.has("timestamp-query")) {
  console.log("Timestamp queries are supported!");
} else {
  console.warn("Timestamp queries not available on this device");
  // Implement fallback or disable profiling features
}
```

The `adapter.features` property returns a `GPUSupportedFeatures` set-like object containing all features supported by the physical GPU and driver. Checking for feature support before device creation prevents runtime errors and allows your application to adapt gracefully.

### Requesting the Feature

Once you've confirmed that timestamp queries are supported, you must include them in the `requiredFeatures` array when requesting a device:

```typescript
const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query"],
});
```

If you request a feature that isn't supported by the adapter, the `requestDevice()` call will fail and return `null`. For more robust code, handle both required and optional features:

```typescript
const requiredFeatures = [];
const optionalFeatures = ["timestamp-query"];

const supportedOptionalFeatures = optionalFeatures.filter((feature) =>
  adapter.features.has(feature),
);

const device = await adapter.requestDevice({
  requiredFeatures: [...requiredFeatures, ...supportedOptionalFeatures],
});

const timingEnabled = device.features.has("timestamp-query");
```

This pattern allows your application to work with or without profiling capabilities, enabling them only when supported.

## Query Sets in WebGPU

Query sets are GPU-managed objects that store query results. For timestamp queries, each query in the set holds a 64-bit timestamp value representing a point in GPU execution time.

### Creating Query Sets

Query sets are created using `device.createQuerySet()` with a descriptor specifying the query type and capacity:

```typescript
const querySet = device.createQuerySet({
  type: "timestamp",
  count: 2,
});
```

**Parameters**:

- `type`: Must be `'timestamp'` for timestamp queries (or `'occlusion'` for visibility queries)
- `count`: The number of individual queries the set can hold. Each timestamp write consumes one query slot.

For profiling a single compute or render pass, you typically need two queries: one for the start time and one for the end time. For more complex profiling scenarios, you might create larger query sets:

```typescript
// Profile multiple passes in a single frame
const querySet = device.createQuerySet({
  type: "timestamp",
  count: 20, // Can hold timestamps for up to 10 passes
});
```

You can optionally include a `label` for debugging:

```typescript
const querySet = device.createQuerySet({
  type: "timestamp",
  count: 2,
  label: "main-render-pass-timing",
});
```

### Writing Timestamps

Timestamps are written to query sets through the `timestampWrites` parameter when beginning render or compute passes. This parameter specifies which query set to use and which indices should receive the beginning and end timestamps.

**In Compute Passes**:

```typescript
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

**In Render Passes**:

```typescript
const passEncoder = commandEncoder.beginRenderPass({
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    },
  ],
  timestampWrites: {
    querySet: querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  },
});

// Render commands...
passEncoder.end();
```

**Parameter Details**:

- `querySet`: The GPUQuerySet object to write timestamps into
- `beginningOfPassWriteIndex`: The query index to receive the timestamp when the pass begins execution on the GPU. Can be `undefined` to skip beginning timestamp.
- `endOfPassWriteIndex`: The query index to receive the timestamp when the pass completes execution. Can be `undefined` to skip end timestamp.

The indices must be less than the query set's `count`, and typically you use consecutive indices like 0 and 1. For profiling multiple passes, you'd use different index ranges:

```typescript
// Pass 1: indices 0-1
// Pass 2: indices 2-3
// Pass 3: indices 4-5

const pass1 = commandEncoder.beginComputePass({
  timestampWrites: {
    querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  },
});
// ...

const pass2 = commandEncoder.beginComputePass({
  timestampWrites: {
    querySet,
    beginningOfPassWriteIndex: 2,
    endOfPassWriteIndex: 3,
  },
});
// ...
```

## Reading Query Results

Timestamp values recorded in query sets aren't immediately accessible to JavaScript. You must first resolve them into a GPU buffer, then map that buffer to CPU memory for reading.

### Resolving to Buffer

The `resolveQuerySet()` method copies query results from the query set into a GPU buffer:

```typescript
// Create a buffer to hold query results
// Each timestamp is 8 bytes (64-bit integer)
const queryBuffer = device.createBuffer({
  size: 2 * 8, // 2 timestamps × 8 bytes each
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});

// Resolve queries into the buffer
commandEncoder.resolveQuerySet(
  querySet, // Source query set
  0, // First query index to resolve
  2, // Number of queries to resolve
  queryBuffer, // Destination buffer
  0, // Byte offset in destination buffer
);
```

The destination buffer must have the `QUERY_RESOLVE` usage flag. Typically, you also include `COPY_SRC` so you can copy the data to a mappable buffer for reading:

```typescript
// Create a mappable buffer for reading results
const resultBuffer = device.createBuffer({
  size: 2 * 8,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// Copy from query buffer to mappable buffer
commandEncoder.copyBufferToBuffer(
  queryBuffer,
  0, // Source
  resultBuffer,
  0, // Destination
  2 * 8, // Size in bytes
);
```

### Reading the Buffer

After submitting commands and waiting for GPU execution to complete, you can map the result buffer and read timestamp values:

```typescript
// Submit all commands
device.queue.submit([commandEncoder.finish()]);

// Wait for GPU to complete
await device.queue.onSubmittedWorkDone();

// Map the result buffer for reading
await resultBuffer.mapAsync(GPUMapMode.READ);

// Get the mapped range as a BigInt64Array
const arrayBuffer = resultBuffer.getMappedRange();
const timestamps = new BigInt64Array(arrayBuffer);

console.log("Start timestamp:", timestamps[0]);
console.log("End timestamp:", timestamps[1]);

// Clean up
resultBuffer.unmap();
```

**Important considerations**:

- Timestamp values are returned as `BigInt` (64-bit integers) to preserve full precision
- You must use `BigInt64Array` to interpret the buffer data correctly
- Always call `unmap()` when done reading to free GPU resources
- The buffer remains mapped until you explicitly unmap it

### Calculating Duration

Timestamp values are in nanoseconds and represent points on the GPU timeline. To calculate execution duration, subtract the start timestamp from the end timestamp:

```typescript
const durationNs = timestamps[1] - timestamps[0]; // Returns BigInt
const durationMs = Number(durationNs) / 1_000_000; // Convert to milliseconds

console.log(`GPU execution time: ${durationMs.toFixed(3)}ms`);
```

**Converting BigInt to Number**: While BigInt preserves full precision, JavaScript's Number type is more convenient for calculations and display. Converting to Number is safe for typical duration measurements (even microsecond precision is preserved up to about 104 days of accumulated time).

**Handling Wraparound**: The timestamp counter is implementation-dependent and may wrap around after reaching maximum value. For typical frame-time measurements (microseconds to milliseconds), wraparound is not a concern. If you need to handle wraparound:

```typescript
function calculateDuration(start: bigint, end: bigint): bigint {
  if (end >= start) {
    return end - start;
  } else {
    // Counter wrapped around
    const maxValue = (1n << 64n) - 1n; // 2^64 - 1
    return maxValue - start + end + 1n;
  }
}
```

However, in practice, with 64-bit nanosecond timestamps, wraparound takes approximately 584 years, so this is rarely necessary.

## TypeGPU Query API

TypeGPU provides a higher-level abstraction over WebGPU's query APIs, simplifying common profiling workflows. It offers both manual query set management and automatic performance callbacks.

### tgpu.createQuerySet()

TypeGPU's `createQuerySet()` method wraps WebGPU query sets with additional conveniences:

```typescript
const querySet = root.createQuerySet("timestamp", 2);
```

The TypeGPU query set provides:

- Automatic buffer management for resolving and reading results
- An `available` property indicating whether results can be read
- Simplified `resolve()` and `read()` methods

**Manual profiling workflow**:

```typescript
const querySet = root.createQuerySet("timestamp", 2);

const pipeline = root["~unstable"]
  .withCompute(computeShader)
  .createPipeline()
  .withTimestampWrites({
    querySet: querySet,
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  });

// Execute pipeline
pipeline.execute();

// Check if results are available
if (querySet.available) {
  querySet.resolve(); // Resolve to internal buffer
  const timestamps = await querySet.read(); // Read as BigInt array

  const durationNs = Number(timestamps[1] - timestamps[0]);
  console.log(`Execution time: ${durationNs / 1_000_000}ms`);
}
```

### Performance Callbacks

TypeGPU's most convenient profiling feature is automatic performance callbacks via `withPerformanceCallback()`:

```typescript
const pipeline = root["~unstable"]
  .withCompute(computeShader)
  .createPipeline()
  .withPerformanceCallback((start, end) => {
    const durationNs = Number(end - start);
    const durationMs = durationNs / 1_000_000;
    console.log(`Pipeline execution: ${durationMs.toFixed(3)}ms`);
  });
```

**How it works**:

- TypeGPU automatically creates a query set and resolve buffers
- Timestamp writes are automatically added to the pipeline
- After each execution, the callback receives start and end timestamps as BigInts
- The callback can be synchronous or asynchronous (returning a Promise)

**Benefits**:

- No manual query set creation or buffer management
- Automatic integration with pipeline execution
- Clean, declarative API for common profiling scenarios

**Limitations**:

- Only the last callback is retained if you call `withPerformanceCallback()` multiple times
- Less control over query set allocation compared to manual management
- Creates separate query resources for each pipeline

## Profiling Patterns

Effective GPU profiling requires more than just measuring execution time. These patterns help you extract meaningful insights from timestamp data.

### Frame Time Breakdown

Understanding how time is distributed across different rendering stages is crucial for optimization:

```typescript
const querySet = root.createQuerySet("timestamp", 10);

// Shadow pass: queries 0-1
const shadowPass = shadowPipeline.withTimestampWrites({
  querySet,
  beginningOfPassWriteIndex: 0,
  endOfPassWriteIndex: 1,
});

// Geometry pass: queries 2-3
const geometryPass = geometryPipeline.withTimestampWrites({
  querySet,
  beginningOfPassWriteIndex: 2,
  endOfPassWriteIndex: 3,
});

// Lighting pass: queries 4-5
const lightingPass = lightingPipeline.withTimestampWrites({
  querySet,
  beginningOfPassWriteIndex: 4,
  endOfPassWriteIndex: 5,
});

// Post-processing: queries 6-7
const postPass = postProcessPipeline.withTimestampWrites({
  querySet,
  beginningOfPassWriteIndex: 6,
  endOfPassWriteIndex: 7,
});

// Execute all passes
shadowPass.execute();
geometryPass.execute();
lightingPass.execute();
postPass.execute();

// Read and analyze
if (querySet.available) {
  querySet.resolve();
  const times = await querySet.read();

  const shadowTime = Number(times[1] - times[0]) / 1_000_000;
  const geometryTime = Number(times[3] - times[2]) / 1_000_000;
  const lightingTime = Number(times[5] - times[4]) / 1_000_000;
  const postTime = Number(times[7] - times[6]) / 1_000_000;

  console.log(`Shadow: ${shadowTime.toFixed(2)}ms`);
  console.log(`Geometry: ${geometryTime.toFixed(2)}ms`);
  console.log(`Lighting: ${lightingTime.toFixed(2)}ms`);
  console.log(`Post: ${postTime.toFixed(2)}ms`);
}
```

### Multiple Query Points

For fine-grained profiling, you can insert timestamps at multiple points within complex rendering pipelines:

```typescript
// Profile different shader variants
const variants = [
  { name: "Baseline", pipeline: baselinePipeline },
  { name: "Optimized v1", pipeline: optimizedV1Pipeline },
  { name: "Optimized v2", pipeline: optimizedV2Pipeline },
];

const querySet = root.createQuerySet("timestamp", variants.length * 2);

variants.forEach((variant, i) => {
  variant.pipeline.withTimestampWrites({
    querySet,
    beginningOfPassWriteIndex: i * 2,
    endOfPassWriteIndex: i * 2 + 1,
  });
});

// Execute all variants
variants.forEach((v) => v.pipeline.execute());

// Compare performance
if (querySet.available) {
  querySet.resolve();
  const times = await querySet.read();

  variants.forEach((variant, i) => {
    const duration = Number(times[i * 2 + 1] - times[i * 2]) / 1_000_000;
    console.log(`${variant.name}: ${duration.toFixed(3)}ms`);
  });
}
```

### Statistical Analysis

Single measurements can be misleading due to variance. Collect statistics over multiple frames:

```typescript
class PerformanceTracker {
  private samples: number[] = [];
  private maxSamples = 100;

  addSample(durationMs: number) {
    this.samples.push(durationMs);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
  }

  getStatistics() {
    if (this.samples.length === 0) return null;

    const sorted = [...this.samples].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);

    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean: sum / sorted.length,
      median: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
    };
  }
}

const tracker = new PerformanceTracker();

pipeline.withPerformanceCallback((start, end) => {
  const durationMs = Number(end - start) / 1_000_000;
  tracker.addSample(durationMs);

  const stats = tracker.getStatistics();
  if (stats) {
    console.log(
      `Mean: ${stats.mean.toFixed(2)}ms, P95: ${stats.p95.toFixed(2)}ms`,
    );
  }
});
```

## Browser Developer Tools

Modern browsers provide built-in tools for GPU profiling and debugging that complement timestamp queries.

### Chrome GPU Profiling

**chrome://gpu**: Navigate to `chrome://gpu` to view detailed information about GPU capabilities, driver versions, and WebGPU feature support. This page shows:

- Graphics adapter details
- Supported WebGPU features (including timestamp-query)
- Driver information
- Hardware acceleration status

**DevTools Performance Tab**:

1. Open Chrome DevTools (F12)
2. Navigate to the Performance tab
3. Enable "Screenshots" and "Web Vitals" if desired
4. Click Record and interact with your WebGPU application
5. Stop recording to analyze the timeline

The Performance tab shows:

- GPU process activity
- Frame rendering timeline
- JavaScript execution overlaid with GPU work
- Long tasks and layout shifts

While it doesn't show timestamp query results directly, it provides context for understanding overall application performance.

### Firefox Graphics Tools

**about:config settings**:

- `webgl.enable-debug-renderer-info`: Enable detailed GPU information
- `layers.acceleration.force-enabled`: Force GPU acceleration
- `gfx.webgpu.force-enabled`: Enable WebGPU support

**Graphics Debugging**:
Firefox provides graphics debugging through about:support:

1. Navigate to `about:support`
2. Scroll to the "Graphics" section
3. View GPU information, feature status, and compositing details

## Complete Example

Here's a complete profiling setup demonstrating TypeGPU integration with statistical analysis:

```typescript
import { init, createRoot, createQuerySet } from "typegpu";

// Initialize TypeGPU
const root = await init();

// Define compute shader
const computeShader = /* wgsl */ `
  @group(0) @binding(0) var<storage, read_write> data: array<f32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    // Complex computation
    var value = data[index];
    for (var i = 0u; i < 1000u; i = i + 1u) {
      value = sin(value) * cos(value) + 0.1;
    }
    data[index] = value;
  }
`;

// Performance tracking
class GPUProfiler {
  private frameTimes: number[] = [];
  private readonly historySize = 60;

  recordFrame(durationMs: number) {
    this.frameTimes.push(durationMs);
    if (this.frameTimes.length > this.historySize) {
      this.frameTimes.shift();
    }
  }

  getStats() {
    if (this.frameTimes.length === 0) return null;

    const sorted = [...this.frameTimes].sort((a, b) => a - b);
    const sum = sorted.reduce((acc, val) => acc + val, 0);

    return {
      fps: 1000 / (sum / sorted.length),
      avg: sum / sorted.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p90: sorted[Math.floor(sorted.length * 0.9)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
    };
  }

  displayStats() {
    const stats = this.getStats();
    if (!stats) return;

    console.clear();
    console.log("=== GPU Performance Stats ===");
    console.log(`FPS: ${stats.fps.toFixed(1)}`);
    console.log(`Average: ${stats.avg.toFixed(2)}ms`);
    console.log(`Min: ${stats.min.toFixed(2)}ms`);
    console.log(`Max: ${stats.max.toFixed(2)}ms`);
    console.log(`P50: ${stats.p50.toFixed(2)}ms`);
    console.log(`P90: ${stats.p90.toFixed(2)}ms`);
    console.log(`P99: ${stats.p99.toFixed(2)}ms`);
  }
}

const profiler = new GPUProfiler();

// Create pipeline with profiling
const pipeline = root["~unstable"]
  .withCompute(computeShader)
  .createPipeline()
  .withPerformanceCallback((start, end) => {
    const durationMs = Number(end - start) / 1_000_000;
    profiler.recordFrame(durationMs);
  });

// Render loop
let frameCount = 0;
function renderLoop() {
  pipeline.execute();

  frameCount++;
  if (frameCount % 60 === 0) {
    profiler.displayStats();
  }

  requestAnimationFrame(renderLoop);
}

renderLoop();
```

## Best Practices

### Minimal Performance Overhead

While timestamp queries are designed to be low-overhead, following these practices minimizes their impact:

**Reuse Query Sets**: Create query sets once and reuse them across frames rather than allocating new ones each frame:

```typescript
// Good: Reuse across frames
const querySet = root.createQuerySet("timestamp", 2);

function render() {
  if (querySet.available) {
    querySet.resolve();
    const times = await querySet.read();
    // Process timing data
  }
  // Use same querySet for this frame
  pipeline.execute();
}

// Bad: Create new query set each frame
function render() {
  const querySet = root.createQuerySet("timestamp", 2); // Don't do this!
  pipeline.execute();
}
```

**Batch Queries**: Use a single large query set for multiple passes rather than creating separate sets:

```typescript
// Good: One query set for all passes
const querySet = root.createQuerySet("timestamp", 20);
// Use indices 0-1, 2-3, 4-5, etc. for different passes

// Less optimal: Separate query sets
const querySet1 = root.createQuerySet("timestamp", 2);
const querySet2 = root.createQuerySet("timestamp", 2);
const querySet3 = root.createQuerySet("timestamp", 2);
```

**Limit Read Frequency**: Don't read query results every frame in production. Consider reading only periodically or when performance issues are detected.

### When to Enable Profiling

**Development**: Enable comprehensive profiling during development to identify bottlenecks and validate optimizations.

**Production**: Disable detailed profiling in production builds, or implement selective profiling that activates only when needed:

```typescript
const ENABLE_PROFILING =
  import.meta.env.DEV || localStorage.getItem("enable-gpu-profiling");

const pipeline = root["~unstable"].withCompute(computeShader).createPipeline();

if (ENABLE_PROFILING) {
  pipeline.withPerformanceCallback((start, end) => {
    // Profiling logic
  });
}
```

**User-Triggered**: Provide an option for users to enable profiling for troubleshooting:

```typescript
window.addEventListener("keydown", (e) => {
  if (e.key === "p" && e.ctrlKey) {
    enableProfiling = !enableProfiling;
    console.log(`GPU profiling: ${enableProfiling ? "ON" : "OFF"}`);
  }
});
```

### Production vs Development

Maintain separate profiling strategies:

**Development**:

- Detailed per-pass timing
- Statistical analysis over many frames
- Console logging or on-screen display
- Integration with development tools

**Production**:

- Minimal or no profiling by default
- Aggregate metrics only (e.g., average frame time)
- Send metrics to analytics backend
- Enable detailed profiling via feature flags for debugging

## Common Pitfalls

### Feature Not Available

The most common error is attempting to use timestamp queries without requesting the feature:

```typescript
// Error: Feature not requested
const device = await adapter.requestDevice();
const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
// Throws error or returns invalid query set
```

**Solution**: Always check feature support and request the feature:

```typescript
if (!adapter.features.has("timestamp-query")) {
  console.warn("Timestamp queries not supported");
  return;
}

const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query"],
});
```

### Query Capacity Limits

Query sets have a maximum size (implementation-dependent, but typically 8192 queries):

```typescript
// May fail on some implementations
const querySet = device.createQuerySet({
  type: "timestamp",
  count: 100000, // Too large!
});
```

**Solution**: Use reasonable query counts and check device limits if needed. For extensive profiling, reuse query sets across frames.

### Timing Interpretation

**Pitfall**: Assuming timestamp values represent wall-clock time or are synchronized with CPU time.

**Reality**: Timestamp values are on the GPU timeline and may not correlate directly with CPU timestamps. They represent relative GPU execution time, not absolute time.

**Solution**: Use timestamps for measuring durations (differences) rather than absolute timing:

```typescript
// Correct: Measure duration
const duration = timestamps[1] - timestamps[0];

// Incorrect: Compare to CPU time
const cpuTime = BigInt(Date.now() * 1_000_000);
const offset = timestamps[0] - cpuTime; // Meaningless!
```

### Reading Before Availability

Attempting to read query results before GPU execution completes causes errors:

```typescript
// Wrong: Immediate read
device.queue.submit([commandEncoder.finish()]);
querySet.resolve(); // Error: GPU hasn't finished yet!
const times = await querySet.read();
```

**Solution**: Always wait for GPU work to complete and check availability:

```typescript
// Correct: Wait for completion
device.queue.submit([commandEncoder.finish()]);
await device.queue.onSubmittedWorkDone();

if (querySet.available) {
  querySet.resolve();
  const times = await querySet.read();
}
```

### Buffer Usage Flags

Forgetting proper usage flags on resolve buffers:

```typescript
// Wrong: Missing QUERY_RESOLVE usage
const buffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.COPY_SRC, // Missing QUERY_RESOLVE!
});

commandEncoder.resolveQuerySet(querySet, 0, 2, buffer, 0); // Error!
```

**Solution**: Include `QUERY_RESOLVE` in buffer usage:

```typescript
const buffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
});
```

---

## Conclusion

Timestamp queries are an essential tool for GPU performance optimization in WebGPU applications. By understanding the asynchronous nature of GPU execution and leveraging query sets properly, you can gain accurate insights into where GPU time is spent. TypeGPU's high-level APIs further simplify profiling workflows while maintaining access to low-level control when needed.

Remember to:

- Always check for and request the `timestamp-query` feature
- Use query sets to measure GPU execution time, not CPU timing functions
- Collect statistical data over multiple frames for reliable measurements
- Minimize profiling overhead in production builds
- Leverage browser developer tools to complement timestamp query data

With these techniques and best practices, you can confidently profile and optimize your WebGPU applications for peak performance.
