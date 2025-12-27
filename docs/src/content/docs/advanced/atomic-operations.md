---
title: Atomic Operations
sidebar:
  order: 20
---

## Overview

Atomic operations provide thread-safe access to shared memory in GPU compute shaders. When thousands of invocations execute concurrently, atomics guarantee that read-modify-write sequences complete without interference.

:::note[Why Atomics?]
Without atomics, concurrent increments can lose updates:

```wgsl
counter += 1u;  // NOT THREAD-SAFE!
```

This compiles to load → add → store. Two threads reading the same value will both compute the same result, losing one increment.
:::

## Atomic Types

### Declaration

```wgsl title="Atomic variable declarations"
// Storage buffer atomics
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;

// Workgroup shared atomics
var<workgroup> localCounter: atomic<i32>;
```

### Supported Types

| Type | Description |
|------|-------------|
| `atomic<u32>` | Unsigned 32-bit integer |
| `atomic<i32>` | Signed 32-bit integer |

:::danger[No Float Atomics]
WGSL does not support `atomic<f32>`. Use fixed-point representation or compare-exchange loops for floating-point atomics.
:::

### Address Space Restrictions

Atomics only work in:
- `var<storage, read_write>` — Storage buffers
- `var<workgroup>` — Workgroup shared memory

## Atomic Operations

### Load and Store

```wgsl title="Basic atomic access"
let value = atomicLoad(&counter);
atomicStore(&counter, 42u);
```

### Arithmetic Operations

All return the **previous value**:

```wgsl title="Atomic arithmetic"
let old = atomicAdd(&counter, 1u);    // Add, return old
let old = atomicSub(&counter, 1u);    // Subtract, return old
let old = atomicMax(&counter, value); // Store max, return old
let old = atomicMin(&counter, value); // Store min, return old
```

### Bitwise Operations

```wgsl title="Atomic bitwise"
let old = atomicAnd(&flags, mask);  // AND, return old
let old = atomicOr(&flags, mask);   // OR, return old
let old = atomicXor(&flags, mask);  // XOR, return old
```

### Exchange and Compare-Exchange

```wgsl title="Atomic exchange operations"
// Simple exchange
let old = atomicExchange(&lock, 1u);

// Compare and exchange (weak)
let result = atomicCompareExchangeWeak(&value, expected, newValue);
if (result.exchanged) {
  // Successfully updated
} else {
  // Failed, result.old_value contains current value
}
```

:::caution[Spurious Failure]
`atomicCompareExchangeWeak` can fail even when values match. Always check `result.exchanged` and retry in a loop.
:::

## Common Patterns

<details>
<summary>**Histogram**</summary>

```wgsl
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn computeHistogram(@builtin(global_invocation_id) gid: vec3u) {
  let value = data[gid.x];
  let bin = value & 0xFFu;
  atomicAdd(&histogram[bin], 1u);
}
```

</details>

<details>
<summary>**Two-Level Histogram (Optimized)**</summary>

```wgsl
var<workgroup> localHistogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn optimizedHistogram(
  @builtin(local_invocation_index) lid: u32,
  @builtin(global_invocation_id) gid: vec3u
) {
  // Initialize local histogram
  atomicStore(&localHistogram[lid], 0u);
  workgroupBarrier();

  // Accumulate in local histogram
  let bin = data[gid.x] & 0xFFu;
  atomicAdd(&localHistogram[bin], 1u);
  workgroupBarrier();

  // Merge to global
  let count = atomicLoad(&localHistogram[lid]);
  if (count > 0u) {
    atomicAdd(&globalHistogram[lid], count);
  }
}
```

</details>

<details>
<summary>**Work Distribution**</summary>

```wgsl
@group(0) @binding(0) var<storage, read_write> workQueue: atomic<u32>;

@compute @workgroup_size(64)
fn processWork() {
  loop {
    let workIndex = atomicAdd(&workQueue, 1u);
    if (workIndex >= totalItems) { break; }
    processItem(workItems[workIndex]);
  }
}
```

</details>

<details>
<summary>**Floating-Point Atomic Max**</summary>

```wgsl
fn atomicMaxFloat(ptr: ptr<storage, atomic<u32>>, value: f32) {
  var expected = atomicLoad(ptr);
  loop {
    if (bitcast<f32>(expected) >= value) { break; }
    let result = atomicCompareExchangeWeak(ptr, expected, bitcast<u32>(value));
    if (result.exchanged) { break; }
    expected = result.old_value;
  }
}
```

</details>

## Memory Ordering

### Acquire-Release Semantics

- **Atomic write** (release): All prior memory ops complete before write visible
- **Atomic read** (acquire): Read completes before subsequent memory ops begin

### Synchronization with Barriers

```wgsl title="Combining atomics with barriers"
var<workgroup> ready: atomic<u32>;
var<workgroup> data: f32;

@compute @workgroup_size(64)
fn process(@builtin(local_invocation_index) idx: u32) {
  if (idx == 0u) {
    data = 42.0;
    workgroupBarrier();  // Ensure write completes
    atomicStore(&ready, 1u);
  }

  // Consumer threads
  if (idx > 0u) {
    loop {
      if (atomicLoad(&ready) == 1u) { break; }
    }
    workgroupBarrier();  // Ensure we see data write
    let value = data;
  }
}
```

## Performance Considerations

:::tip[Reduce Contention]
1. **Use workgroup memory first**: Local atomics are faster than global
2. **Distribute hot spots**: Use multiple counters and distribute access
3. **Batch updates**: Accumulate locally, then atomic update once

```wgsl
// Bad: All threads hit same counter
atomicAdd(&globalCounter, 1u);

// Better: Use multiple counters
let idx = hash(gid.x) % 32u;
atomicAdd(&counters[idx], 1u);
```
:::

### Throughput Impact

| Operation | Relative Cost |
|-----------|---------------|
| Regular memory | 1× |
| Atomic (low contention) | 10-100× |
| Atomic (high contention) | 100-1000× |

## TypeGPU Atomic Support

```typescript title="TypeGPU atomic types"
import { d } from "typegpu/data";
import { std } from "typegpu/std";

const atomicCounter = d.atomic(d.u32);

const CounterBuffer = d.struct({
  hitCount: d.atomic(d.u32),
  missCount: d.atomic(d.u32),
});

// In shader
const shader = tgpu.computeFn([counterBuffer], () => {
  const old = std.atomicAdd(counterBuffer.value.hitCount, 1);
});
```

## Common Pitfalls

:::danger[Race on Non-Atomic Data]
```wgsl
// Atomics only protect the atomic variable itself!
let idx = atomicAdd(&counter, 1u);
data[idx] = value;  // data array is NOT protected

// Use workgroupBarrier() after atomics if other threads read data
```
:::

:::danger[Spinlock Deadlock]
```wgsl
// DANGEROUS: GPU threads in same warp execute in lockstep
loop {
  if (atomicExchange(&lock, 1u) == 0u) { break; }
}
// One thread holding lock blocks others in same warp forever
```

**Solution**: Avoid locks on GPUs. Redesign for lock-free algorithms.
:::

:::danger[Missing Barrier for Non-Atomic Memory]
```wgsl
// Thread A
result = 42.0;
atomicStore(&ready, 1u);

// Thread B - might see old value!
if (atomicLoad(&ready) == 1u) {
  let value = result;  // Not guaranteed to be 42.0
}

// Fix: Add workgroupBarrier() before and after the atomic
```
:::

:::danger[Unnecessary Atomics]
```wgsl
// Bad: Atomic when each thread writes unique location
atomicStore(&output[gid.x], value);

// Good: Regular store is sufficient
output[gid.x] = value;
```
:::

## Resources

:::note[Official Documentation]
- [WGSL Atomic Built-in Functions](https://www.w3.org/TR/WGSL/#atomic-builtin-functions)
- [WebGPU Fundamentals: Compute Histograms](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram.html)
- [TypeGPU Data Schemas](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/)
:::
