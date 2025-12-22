# Atomic Operations

Atomic operations are specialized built-in functions in WGSL that provide thread-safe access to shared memory in highly parallel GPU compute shaders. When thousands of shader invocations execute concurrently and need to update the same memory locations, atomic operations guarantee that each read-modify-write sequence completes without interference, preventing data races and ensuring predictable results.

In GPU programming, where massive parallelism is the norm, atomics are essential for implementing concurrent algorithms like histograms, counters, reductions, and coordination primitives. They enable safe concurrent access without requiring explicit locks or complex synchronization protocols.

## Why Atomics?

### The Problem: Race Conditions in Parallel Code

GPU shaders execute with extreme parallelism. A single compute shader dispatch might launch hundreds of thousands of invocations running simultaneously across multiple execution units. When multiple invocations share variables in the `workgroup` or `storage` address spaces, concurrent access creates potential for data races.

Consider a simple increment operation that appears atomic at the source level:

```wgsl
var<storage, read_write> counter: u32;

@compute @workgroup_size(256)
fn incrementCounter(@builtin(global_invocation_id) id: vec3u) {
    counter += 1u;  // NOT THREAD-SAFE!
}
```

This innocent-looking line actually compiles to three distinct operations:

1. **Load**: Read the current value from memory
2. **Add**: Compute the new value (current + 1)
3. **Store**: Write the result back to memory

When two invocations execute these steps concurrently on the same memory location, race conditions occur. Both might read the same initial value (say, 100), independently compute 101, and both write 101 back. The counter increases by only 1 instead of 2—a lost update. With thousands of concurrent invocations, the final result becomes completely unpredictable.

As the WGSL specification states: "Data races can have **unpredictable results with non-local effects**." These aren't just off-by-one errors—they can corrupt unrelated data and cause non-deterministic failures that are extremely difficult to debug.

### The Solution: Synchronized Access

Atomic operations solve this by guaranteeing that the entire read-modify-write sequence executes as an indivisible unit. From the perspective of all other invocations, the operation appears to happen instantaneously. The system ensures that "atomic accesses to a single memory word will occur as if they happened in some order, one after the other."

### Use Cases for Atomic Operations

**Counters and Accumulators**: The most straightforward use case. Multiple invocations can safely increment or decrement shared counters for tracking events, counting elements, or generating unique IDs.

**Histogram Computation**: When processing images or data streams, many invocations might need to increment the same histogram bin. Atomics prevent lost updates when multiple pixels map to identical bins.

**Lock-Free Data Structures**: Implementing simple synchronization primitives like spinlocks, semaphores, or lock-free queues for coordinating work across invocations.

**Reduction Operations**: Computing aggregate values (sum, min, max) across all invocations by safely updating shared accumulators.

**Work Distribution**: Implementing work-stealing or dynamic load-balancing algorithms where invocations claim work items from a shared queue.

## Atomic Types in WGSL

### Declaring Atomic Variables

WGSL provides atomic types using the syntax `atomic<T>`, where `T` must be either `i32` (signed 32-bit integer) or `u32` (unsigned 32-bit integer). These are the only supported types—**there are no floating-point atomics** in WGSL.

Atomic variables can only exist in specific memory locations:

```wgsl
// Storage buffer atomics
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;

// Workgroup shared atomics
var<workgroup> localCounter: atomic<i32>;
var<workgroup> locks: array<atomic<u32>, 32>;

// Atomics within structs in storage
struct Statistics {
    totalCount: atomic<u32>,
    errorCount: atomic<u32>,
    minValue: atomic<i32>,
    maxValue: atomic<i32>,
}

@group(0) @binding(2) var<storage, read_write> stats: Statistics;
```

### Address Space Restrictions

Atomic types can **only** appear in variables declared with the `workgroup` or `storage` address space. You cannot use atomics in:

- Function-local variables (`function` address space)
- Private per-invocation variables (`private` address space)
- Uniform buffers (which are read-only)

This restriction makes sense: atomics exist to coordinate concurrent access to shared memory, so they're only useful where sharing occurs.

### Supported Types

**`atomic<u32>`**: Unsigned 32-bit integer atomics. Most common for counters, histograms, and flags since they naturally represent non-negative counts and can use the full 0 to 4,294,967,295 range.

**`atomic<i32>`**: Signed 32-bit integer atomics. Useful when you need to represent negative values, such as in algorithms computing differences or tracking values that can increase or decrease.

**No `atomic<f32>`**: WGSL does not support atomic floating-point operations. If you need to atomically update floating-point values, you must use workarounds like:
- Converting to fixed-point representation using integers
- Using `atomicCompareExchangeWeak` in a loop with bitcast operations (advanced and inefficient)
- Redesigning your algorithm to avoid the need

### Usage Limitations

Atomic types have severe usage restrictions. They are **not constructible**, meaning you cannot:

- Use them directly in expressions
- Pass atomic values as function parameters
- Return atomic values from functions
- Assign atomic values to other variables
- Initialize them with literal values

Instead, you must always access atomics through **pointers** passed to atomic built-in functions:

```wgsl
var<workgroup> counter: atomic<u32>;

fn processData() {
    // WRONG: Cannot use atomic directly
    // let value = counter;  // ERROR: atomics are not constructible

    // CORRECT: Use atomic built-in functions with pointers
    let value = atomicLoad(&counter);
    atomicStore(&counter, 42u);
    atomicAdd(&counter, 1u);
}
```

This design enforces that all access goes through the proper atomic operations, preventing accidental non-atomic access that would break thread-safety guarantees.

## Atomic Operations

WGSL provides a comprehensive set of atomic built-in functions. All atomic functions take a pointer to the atomic variable as their first parameter and return the **previous value** stored before the operation (except for `atomicStore` and `atomicLoad`).

### Load and Store

The most basic operations read and write atomic variables:

```wgsl
// Atomically load current value
fn atomicLoad(atomic_ptr: ptr<AS, atomic<T>, read_write>) -> T

// Atomically store new value
fn atomicStore(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T)

// Example usage
var<workgroup> sharedValue: atomic<u32>;

@compute @workgroup_size(64)
fn example() {
    // Read current value
    let currentValue = atomicLoad(&sharedValue);

    // Write new value
    atomicStore(&sharedValue, 100u);
}
```

`atomicLoad` ensures you read a consistent value even if other invocations are concurrently modifying it. `atomicStore` ensures the write completes fully before any subsequent atomic operation sees the old value.

### Arithmetic Operations

These perform atomic read-modify-write arithmetic:

```wgsl
// Add value, return old value
fn atomicAdd(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T

// Subtract value, return old value
fn atomicSub(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T

// Store maximum of current and v, return old value
fn atomicMax(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T

// Store minimum of current and v, return old value
fn atomicMin(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T
```

Example usage:

```wgsl
@group(0) @binding(0) var<storage, read_write> globalCounter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> bounds: array<atomic<i32>, 2>;

@compute @workgroup_size(256)
fn processData(@builtin(global_invocation_id) id: vec3u) {
    // Increment counter and get unique ID
    let myId = atomicAdd(&globalCounter, 1u);

    // Track minimum and maximum values
    let dataValue = i32(computeSomeValue(id.x));
    atomicMin(&bounds[0], dataValue);  // Update minimum
    atomicMax(&bounds[1], dataValue);  // Update maximum

    // Use myId for unique processing...
}
```

All arithmetic atomics return the **old value** before the operation. This is crucial for algorithms that need to know the previous state, such as when generating unique IDs or tracking changes.

### Bitwise Operations

Atomic bitwise operations are useful for flag manipulation and bit-level synchronization:

```wgsl
// Bitwise AND, return old value
fn atomicAnd(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T

// Bitwise OR, return old value
fn atomicOr(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T

// Bitwise XOR, return old value
fn atomicXor(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T
```

Example using bitwise atomics for flag management:

```wgsl
var<workgroup> statusFlags: atomic<u32>;

const FLAG_INITIALIZED: u32 = 0x01u;
const FLAG_PROCESSING: u32 = 0x02u;
const FLAG_COMPLETE: u32 = 0x04u;

@compute @workgroup_size(64)
fn worker(@builtin(local_invocation_index) localIdx: u32) {
    if (localIdx == 0u) {
        // First thread sets initialization flag
        atomicOr(&statusFlags, FLAG_INITIALIZED);
    }

    workgroupBarrier();

    // Process work...
    atomicOr(&statusFlags, FLAG_PROCESSING);

    // Work done, clear processing flag
    atomicAnd(&statusFlags, ~FLAG_PROCESSING);
    atomicOr(&statusFlags, FLAG_COMPLETE);
}
```

### Exchange Operations

`atomicExchange` swaps the atomic value with a new value, returning the old value:

```wgsl
// Replace value, return old value
fn atomicExchange(atomic_ptr: ptr<AS, atomic<T>, read_write>, v: T) -> T

// Example: Implement a simple spinlock
var<workgroup> lock: atomic<u32>;

fn acquireLock() {
    // Spin until we can set lock from 0 to 1
    loop {
        let oldValue = atomicExchange(&lock, 1u);
        if (oldValue == 0u) {
            break;  // We acquired the lock
        }
        // Lock was already held, try again
    }
}

fn releaseLock() {
    atomicStore(&lock, 0u);
}
```

**Warning**: While this implements a functional spinlock, busy-waiting on GPUs is extremely inefficient and should be avoided in production code. GPUs lack the sophisticated cache coherency protocols that make spinlocks viable on CPUs.

### Compare and Exchange

The most powerful atomic operation is `atomicCompareExchangeWeak`:

```wgsl
struct __atomic_compare_exchange_result<T> {
    old_value: T,
    exchanged: bool,
}

fn atomicCompareExchangeWeak(
    atomic_ptr: ptr<AS, atomic<T>, read_write>,
    compare: T,
    value: T
) -> __atomic_compare_exchange_result<T>
```

This operation atomically performs:

1. Load the current value from `atomic_ptr`
2. Compare it with `compare`
3. If they match, store `value` and return `{old_value, true}`
4. If they don't match, return `{old_value, false}` without modifying memory

The "weak" variant means the comparison can **spuriously fail** even when the values match. This is an optimization allowing more efficient implementations on some hardware. You must always check the `exchanged` flag, never assume success.

Example implementing atomic maximum for floating-point values:

```wgsl
@group(0) @binding(0) var<storage, read_write> maxValue: atomic<u32>;

fn atomicMaxFloat(ptr: ptr<storage, atomic<u32>, read_write>, value: f32) {
    var expected = atomicLoad(ptr);

    loop {
        let expectedFloat = bitcast<f32>(expected);

        // If current value is already >= new value, we're done
        if (expectedFloat >= value) {
            break;
        }

        // Try to update to new maximum
        let newBits = bitcast<u32>(value);
        let result = atomicCompareExchangeWeak(ptr, expected, newBits);

        if (result.exchanged) {
            break;  // Successfully updated
        }

        // Someone else modified it, retry with new value
        expected = result.old_value;
    }
}
```

This pattern—loading current value, computing new value, attempting compare-exchange, and looping on failure—is the foundation for building complex atomic operations.

## Memory Ordering

WGSL atomics provide specific memory ordering guarantees that determine when other invocations observe atomic operations.

### Happens-Before Relationships

The memory model defines "happens-before" relationships between operations. When operation A happens-before operation B:

- A's effects are visible to B
- A completes before B begins
- Memory updates from A are guaranteed to be visible when B executes

Atomic operations establish happens-before relationships across invocations, providing the synchronization needed for correct concurrent algorithms.

### Acquire-Release Semantics

WGSL atomics use **acquire-release semantics**:

- **Atomic write** (store, add, exchange, etc.) has **release semantics**: All memory operations before the atomic in program order complete before the atomic write becomes visible to other invocations.

- **Atomic read** (load, compare-exchange result, value returned from read-modify-write) has **acquire semantics**: The atomic read completes before any subsequent memory operations begin.

This ensures that when one invocation performs an atomic write and another invocation reads that value, the reader sees not just the atomic value but also all memory updates the writer performed before the atomic operation.

### Consistency Guarantees

For a **single memory word**, atomics guarantee sequential consistency: all invocations observe atomic operations in some single, consistent order. If invocation A's atomic write happens before invocation B's atomic write, no invocation will see B's write without also seeing A's.

However, this guarantee applies **only to individual memory words**. When reasoning about multiple atomic variables, causality can appear violated. If you need ordering across multiple atomics, you must use explicit synchronization like `workgroupBarrier()` or `storageBarrier()`.

### Synchronization with Barriers

Atomic operations synchronize individual memory locations, but barriers synchronize broader program execution:

```wgsl
var<workgroup> ready: atomic<u32>;
var<workgroup> data: array<f32, 256>;

@compute @workgroup_size(256)
fn process(@builtin(local_invocation_index) idx: u32) {
    // Producer thread
    if (idx == 0u) {
        data[0] = 42.0;  // Non-atomic write
        workgroupBarrier();  // Ensure write completes
        atomicStore(&ready, 1u);  // Signal ready
    }

    // Consumer threads
    if (idx > 0u) {
        // Wait for ready signal
        loop {
            if (atomicLoad(&ready) == 1u) {
                break;
            }
        }
        workgroupBarrier();  // Ensure we see data writes

        // Safe to read data[0] now
        let value = data[0];
    }
}
```

`workgroupBarrier()` ensures both execution and memory synchronization within a workgroup. `storageBarrier()` provides memory synchronization for storage buffers but not execution synchronization.

## Common Patterns

### Histogram Computation

Histograms are the canonical use case for atomics. Multiple invocations process data elements and increment bins based on values:

```wgsl
@group(0) @binding(0) var<storage, read> imageData: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn computeHistogram(@builtin(global_invocation_id) gid: vec3u) {
    let pixelIndex = gid.x;

    if (pixelIndex < arrayLength(&imageData)) {
        let pixelValue = imageData[pixelIndex];
        let binIndex = pixelValue & 0xFFu;  // Use low 8 bits

        // Atomic increment prevents lost updates
        atomicAdd(&histogram[binIndex], 1u);
    }
}
```

Without atomics, when multiple invocations simultaneously process pixels with the same value, increments would be lost. Atomics ensure every pixel contributes exactly once to the histogram.

### Optimized Two-Level Histogram

For better performance, use workgroup memory for local histograms, then merge into global storage:

```wgsl
var<workgroup> localHistogram: array<atomic<u32>, 256>;

@group(0) @binding(0) var<storage, read> imageData: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHistogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn computeHistogramOptimized(
    @builtin(local_invocation_index) localIdx: u32,
    @builtin(global_invocation_id) gid: vec3u
) {
    // Initialize local histogram
    atomicStore(&localHistogram[localIdx], 0u);
    workgroupBarrier();

    // Process pixels, updating local histogram
    var pixelIdx = gid.x;
    let stride = 256u;
    while (pixelIdx < arrayLength(&imageData)) {
        let pixelValue = imageData[pixelIdx];
        let binIndex = pixelValue & 0xFFu;
        atomicAdd(&localHistogram[binIndex], 1u);
        pixelIdx += stride;
    }

    workgroupBarrier();

    // Merge local histogram into global
    let localCount = atomicLoad(&localHistogram[localIdx]);
    if (localCount > 0u) {
        atomicAdd(&globalHistogram[localIdx], localCount);
    }
}
```

This reduces contention on global memory atomics by performing most increments on faster workgroup memory.

### Reduction Operations

Computing a sum across all invocations using atomics:

```wgsl
@group(0) @binding(0) var<storage, read> inputData: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: atomic<u32>;

@compute @workgroup_size(256)
fn reduceSum(@builtin(global_invocation_id) gid: vec3u) {
    let value = inputData[gid.x];

    // Convert to fixed-point (16.16) for atomic addition
    let fixedPoint = u32(value * 65536.0);
    atomicAdd(&result, fixedPoint);
}
```

For better performance, use a two-stage reduction with workgroup memory first, then atomic merge to global memory (similar to the histogram pattern).

### Parallel Prefix Sum (Scan)

Using atomics for coordination in a prefix sum algorithm:

```wgsl
var<workgroup> blockSum: atomic<u32>;
@group(0) @binding(0) var<storage, read_write> blockPrefixes: array<atomic<u32>>;

@compute @workgroup_size(256)
fn prefixSumPhase1(@builtin(workgroup_id) wgId: vec3u,
                    @builtin(local_invocation_index) localIdx: u32) {
    // Each workgroup computes local prefix sum (standard parallel scan)
    // ...

    // Last thread in workgroup stores total to global prefix array
    if (localIdx == 255u) {
        let total = computeWorkgroupTotal();
        atomicStore(&blockPrefixes[wgId.x], total);
    }
}
```

### Global Work Distribution

Implementing work-stealing for dynamic load balancing:

```wgsl
@group(0) @binding(0) var<storage, read_write> workQueue: atomic<u32>;
@group(0) @binding(1) var<storage, read> workItems: array<WorkItem>;

const NUM_WORK_ITEMS: u32 = 10000u;

@compute @workgroup_size(64)
fn processWorkItems() {
    loop {
        // Atomically claim next work item
        let workIndex = atomicAdd(&workQueue, 1u);

        if (workIndex >= NUM_WORK_ITEMS) {
            break;  // No more work
        }

        // Process the work item
        let item = workItems[workIndex];
        processItem(item);
    }
}
```

Each invocation atomically increments the queue pointer to claim its next work item, enabling automatic load balancing even when work items have variable cost.

### Simple Semaphore

Implementing a counting semaphore for resource allocation:

```wgsl
var<workgroup> availableSlots: atomic<i32>;

fn acquireSlot() -> bool {
    loop {
        let current = atomicLoad(&availableSlots);
        if (current <= 0) {
            return false;  // No slots available
        }

        let result = atomicCompareExchangeWeak(
            &availableSlots,
            current,
            current - 1
        );

        if (result.exchanged) {
            return true;  // Successfully acquired
        }
        // Retry if compare-exchange failed
    }
}

fn releaseSlot() {
    atomicAdd(&availableSlots, 1);
}
```

## Performance Considerations

### Atomic Contention

Atomics serialize access to memory locations. When many invocations simultaneously attempt atomic operations on the same location, hardware must serialize these operations, creating a bottleneck.

**Hot spot problem**: If all invocations atomically update the same counter, performance degrades to sequential execution. With 10,000 invocations all incrementing one counter, you lose the benefit of parallelism.

**Contention visualization**:

```
Low contention (good):         High contention (bad):
Invocation 1 → Bin 7          Invocation 1 ↘
Invocation 2 → Bin 23         Invocation 2 → Bin 0 (BOTTLENECK)
Invocation 3 → Bin 145        Invocation 3 ↗
Invocation 4 → Bin 67         Invocation 4 ↗
```

### Strategies to Reduce Contention

**1. Use workgroup memory for local aggregation**: Perform most atomics on fast workgroup memory, then merge to global storage in a final step (as shown in the two-level histogram pattern).

**2. Distribute hot spots**: Instead of one global counter, use multiple counters and distribute invocations across them:

```wgsl
@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>, 32>;

@compute @workgroup_size(256)
fn distributedCount(@builtin(global_invocation_id) gid: vec3u) {
    // Hash invocation ID to select counter
    let counterIdx = hash(gid.x) % 32u;
    atomicAdd(&counters[counterIdx], 1u);
}
```

**3. Batch updates**: Accumulate multiple updates locally before performing a single atomic operation:

```wgsl
@compute @workgroup_size(256)
fn batchedHistogram(@builtin(local_invocation_index) localIdx: u32) {
    var localCounts: array<u32, 256>;

    // Process many pixels, updating local counts
    for (var i = 0u; i < 100u; i++) {
        let bin = procesPixel(i);
        localCounts[bin] += 1u;
    }

    // Single atomic update per bin at the end
    for (var bin = 0u; bin < 256u; bin++) {
        if (localCounts[bin] > 0u) {
            atomicAdd(&globalHistogram[bin], localCounts[bin]);
        }
    }
}
```

### Throughput Impact

Atomic operations are significantly slower than regular memory operations. On typical GPU hardware:

- Regular memory read/write: ~1-10 cycles
- Atomic operation: ~100-1000 cycles (depending on contention)

The performance gap widens with contention. Use atomics judiciously and only where necessary for correctness.

### When Atomics Are Appropriate

**Good use cases**:
- Infrequent updates (histogram bins hit rarely)
- Distributed access patterns (different invocations update different locations)
- Coordination primitives (flags, semaphores) used occasionally
- Final aggregation step after local computation

**Poor use cases**:
- Inner loop increments (every iteration of tight loop)
- Single hot-spot counter (all invocations update same location)
- Large data structure updates (consider organizing work differently)
- Replacing proper algorithm design (atomics aren't a substitute for good parallel algorithms)

## TypeGPU Atomic Support

TypeGPU provides type-safe atomic operations through its data schema system, bringing TypeScript's type safety to GPU atomic programming.

### Declaring Atomic Types

In TypeGPU, use `d.atomic()` to mark integer schemas as atomic:

```typescript
import { d } from 'typegpu/data';

// Declare atomic types
const atomicU32 = d.atomic(d.u32);
const atomicI32 = d.atomic(d.i32);

// Use in buffer schemas
const CounterBuffer = d.buffer(
  d.struct({
    hitCount: atomicU32,
    missCount: atomicU32,
    errorCount: atomicI32,
  })
);
```

### Type Inference

TypeGPU's type system ensures atomic types are used correctly. The `d.Infer` utility extracts the TypeScript type:

```typescript
type AtomicCounterType = d.Infer<typeof atomicU32>;  // number
```

Atomic values are represented as `number` on the JavaScript side, with TypeGPU handling the conversion to/from atomic types in shaders.

### Using Atomics in TGSL

When writing shaders in TGSL (TypeScript Shader Language), atomic operations from the standard library work with atomic types:

```typescript
import { std } from 'typegpu/std';

const counterBuffer = /* ... */;

const shader = tgpu.bindgroup({
  counter: counterBuffer,
})
.withCode(({counter}) => {
  return tgpu.computeFn([], () => {
    // Use std functions for atomic operations
    const oldValue = std.atomicAdd(counter.hitCount, 1);
    std.atomicStore(counter.missCount, 0);

    const current = std.atomicLoad(counter.errorCount);
    if (current > 100) {
      std.atomicSub(counter.errorCount, 50);
    }
  });
});
```

### Type Safety Benefits

TypeGPU prevents common mistakes at compile time:

```typescript
// Error: Cannot use atomic type in regular expression
const counterValue = counter.hitCount;  // TypeScript error!

// Correct: Must use atomic operations
const counterValue = std.atomicLoad(counter.hitCount);  // ✓
```

This catches bugs early that would otherwise cause shader compilation failures or subtle runtime errors.

### Limitations

TypeGPU atomic support follows WGSL limitations:

- Only `u32` and `i32` types supported
- Atomic variables must be in `storage` or `workgroup` address spaces
- No floating-point atomics
- Cannot pass atomic values directly (must use pointers)

When TGSL doesn't support a specific atomic operation or pattern, you can always fall back to raw WGSL within the same shader using TypeGPU's interoperability features.

## Best Practices

**1. Minimize atomic operations**: Every atomic operation serializes access and adds overhead. Use them only where necessary for correctness, not as a default.

**2. Reduce contention through locality**: Use workgroup-local atomics for most work, then merge to global storage. This exploits the faster workgroup memory and reduces global memory traffic.

**3. Consider algorithm redesign**: Sometimes restructuring your algorithm to avoid shared state entirely is better than adding atomics. Can you partition work so each invocation updates independent memory?

**4. Batch updates when possible**: Accumulate multiple changes locally before performing a single atomic update. This amortizes the atomic operation cost.

**5. Use appropriate types**: Choose `u32` for non-negative counts and `i32` when you need signed arithmetic. Don't use signed integers unnecessarily.

**6. Profile and measure**: Use timestamp queries to measure performance impact. Atomic contention effects are often worse than expected and may not be obvious without profiling.

**7. Document synchronization assumptions**: Clearly comment what each atomic variable protects and what ordering guarantees your algorithm requires. Concurrent code is hard to understand; good documentation is essential.

**8. Understand memory ordering**: Know when you need barriers versus atomics. Atomics synchronize single memory locations; barriers synchronize execution and broader memory state.

## Common Pitfalls

### Race Conditions Despite Atomics

Atomics only protect the single memory location they operate on. Adjacent operations can still race:

```wgsl
var<workgroup> counter: atomic<u32>;
var<workgroup> data: array<f32, 256>;

// BUG: Race condition on data array!
@compute @workgroup_size(256)
fn racyCode(@builtin(local_invocation_index) idx: u32) {
    let myIndex = atomicAdd(&counter, 1u);
    data[myIndex] = computeValue();  // Not protected by atomic!
}
```

Another invocation might read `data[myIndex]` before this invocation writes it. Use `workgroupBarrier()` after the atomic to ensure ordering.

### Forgetting Barriers Between Atomic and Non-Atomic

Atomics don't automatically synchronize non-atomic memory:

```wgsl
var<workgroup> ready: atomic<u32>;
var<workgroup> result: f32;

// Thread A
result = 42.0;
atomicStore(&ready, 1u);

// Thread B - BUG: Might not see result update!
if (atomicLoad(&ready) == 1u) {
    let value = result;  // Might see old value!
}
```

**Fix**: Insert `workgroupBarrier()` to ensure memory visibility:

```wgsl
// Thread A
result = 42.0;
workgroupBarrier();
atomicStore(&ready, 1u);

// Thread B
if (atomicLoad(&ready) == 1u) {
    workgroupBarrier();
    let value = result;  // Guaranteed to see update
}
```

### Deadlock Potential with Spinlocks

Spinlocks on GPUs are extremely dangerous:

```wgsl
var<workgroup> lock: atomic<u32>;

// DANGEROUS: Can deadlock!
fn acquireLock() {
    loop {
        if (atomicExchange(&lock, 1u) == 0u) {
            break;
        }
        // Spin waiting - this might never complete!
    }
}
```

GPUs execute invocations in SIMD groups (warps/subgroups). If one invocation in a group acquires the lock and another in the same group tries to acquire it, the second will spin forever because the first cannot make progress (they execute in lockstep).

**Better approach**: Avoid locks on GPUs. Redesign algorithms to be lock-free or partition work to eliminate contention.

### Excessive Atomic Usage

Don't use atomics when they're unnecessary:

```wgsl
// BAD: Using atomics when each invocation has unique data
@compute @workgroup_size(256)
fn inefficientCode(@builtin(global_invocation_id) gid: vec3u) {
    atomicStore(&output[gid.x], computeValue());  // Unnecessary atomic!
}

// GOOD: Regular store when no sharing occurs
@compute @workgroup_size(256)
fn efficientCode(@builtin(global_invocation_id) gid: vec3u) {
    output[gid.x] = computeValue();  // Much faster
}
```

If each invocation writes to a unique location, atomics add overhead without benefit.

### Assuming Sequential Consistency Across Variables

Atomics guarantee ordering only for individual memory locations. This assumption is wrong:

```wgsl
var<workgroup> flag: atomic<u32>;
var<workgroup> data: atomic<u32>;

// Thread A
atomicStore(&data, 42u);
atomicStore(&flag, 1u);

// Thread B - WRONG ASSUMPTION
if (atomicLoad(&flag) == 1u) {
    // CANNOT assume data is 42!
    let value = atomicLoad(&data);
}
```

Even though both are atomic, without barriers, thread B might observe `flag = 1` but `data = old_value`. Use `workgroupBarrier()` for cross-variable ordering guarantees.

### Spurious Failures with Compare-Exchange

Always check the `exchanged` flag from `atomicCompareExchangeWeak`:

```wgsl
// BUG: Assumes compare-exchange always succeeds when values match
let result = atomicCompareExchangeWeak(&counter, 0u, 1u);
// Using result.old_value here without checking result.exchanged is wrong!

// CORRECT: Always check success
if (result.exchanged) {
    // Update succeeded, safe to use result.old_value
} else {
    // Update failed, must retry or handle failure
}
```

The "weak" variant can spuriously fail even when the comparison would succeed. This is a hardware optimization; always design your loop to handle retries.

---

## Summary

Atomic operations are essential tools for safe concurrent programming on GPUs. They provide the synchronization primitives needed to implement counters, histograms, reductions, and coordination mechanisms in massively parallel compute shaders.

Key takeaways:

- **Atomics prevent data races** by making read-modify-write operations indivisible
- **Limited types**: Only `atomic<u32>` and `atomic<i32>` are supported
- **Address space restrictions**: Only in `storage` and `workgroup` memory
- **Performance costs**: Atomics serialize access and can create bottlenecks
- **Use workgroup memory** for local aggregation before merging to global storage
- **Barriers complement atomics**: Use `workgroupBarrier()` for execution and memory synchronization
- **TypeGPU provides type safety**: Atomic types integrate with TypeGPU's schema system

When used appropriately, atomics enable correct and efficient parallel algorithms. When overused or misused, they can create performance bottlenecks or subtle bugs. Understanding their semantics, costs, and best practices is essential for effective GPU programming.

## Further Reading

- [WGSL Specification - Atomic Built-in Functions](https://www.w3.org/TR/WGSL/)
- [WebGPU Fundamentals - Compute Shaders Histogram](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram.html)
- [WebGPU.rocks - Synchronization & Atomic Functions](https://webgpu.rocks/wgsl/functions/synchronization-atomic/)
- [Tour of WGSL - Atomic Types](https://google.github.io/tour-of-wgsl/types/atomics/atomic-types/)
- [TypeGPU Documentation - Data Schemas](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/)
