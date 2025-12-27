---
title: WGSL Builtin Functions
sidebar:
  order: 35
---

## Overview

WGSL provides builtin functions for mathematical operations, vector/matrix manipulation, texture sampling, and more. This reference covers commonly used functions organized by category.

:::note[Component-wise Operations]
Most scalar functions also work on vectors, applying the operation to each component independently.
:::

## Math Functions

### Trigonometric

| Function | Description | Domain |
|----------|-------------|--------|
| `sin(x)` | Sine | Any |
| `cos(x)` | Cosine | Any |
| `tan(x)` | Tangent | x ≠ π/2 + nπ |
| `asin(x)` | Arc sine | [-1, 1] |
| `acos(x)` | Arc cosine | [-1, 1] |
| `atan(x)` | Arc tangent | Any |
| `atan2(y, x)` | Arc tangent of y/x | Any (handles quadrants) |
| `sinh(x)` | Hyperbolic sine | Any |
| `cosh(x)` | Hyperbolic cosine | Any |
| `tanh(x)` | Hyperbolic tangent | Any |
| `asinh(x)` | Inverse hyperbolic sine | Any |
| `acosh(x)` | Inverse hyperbolic cosine | x ≥ 1 |
| `atanh(x)` | Inverse hyperbolic tangent | (-1, 1) |

```wgsl title="Trigonometry example"
let angle = atan2(direction.y, direction.x);
let rotated = vec2f(cos(angle), sin(angle));
```

### Exponential

| Function | Description | Domain |
|----------|-------------|--------|
| `pow(x, y)` | x raised to power y | x > 0, or x = 0 and y > 0 |
| `exp(x)` | e^x | Any |
| `exp2(x)` | 2^x | Any |
| `log(x)` | Natural logarithm | x > 0 |
| `log2(x)` | Base-2 logarithm | x > 0 |
| `sqrt(x)` | Square root | x ≥ 0 |
| `inverseSqrt(x)` | 1 / sqrt(x) | x > 0 |

:::danger[Undefined Behavior]
`pow(-2.0, 0.5)` is undefined (would produce imaginary number). Results vary by GPU.
:::

### Rounding

| Function | Description |
|----------|-------------|
| `floor(x)` | Largest integer ≤ x |
| `ceil(x)` | Smallest integer ≥ x |
| `round(x)` | Nearest integer (ties to even) |
| `trunc(x)` | Integer part (toward zero) |
| `fract(x)` | Fractional part: `x - floor(x)` |

```wgsl title="Rounding examples"
floor(2.7)   // 2.0
ceil(2.3)    // 3.0
round(2.5)   // 2.0 (ties to even)
fract(2.7)   // 0.7
trunc(-2.7)  // -2.0
```

### Common

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `sign(x)` | -1, 0, or 1 |
| `min(a, b)` | Smaller value |
| `max(a, b)` | Larger value |
| `clamp(x, lo, hi)` | Constrain x to [lo, hi] |
| `saturate(x)` | Clamp to [0, 1] |
| `fma(a, b, c)` | Fused multiply-add: a*b + c |

## Vector Functions

### Geometric

| Function | Description |
|----------|-------------|
| `dot(a, b)` | Dot product |
| `cross(a, b)` | Cross product (vec3 only) |
| `length(v)` | Vector magnitude |
| `distance(a, b)` | Distance between points |
| `normalize(v)` | Unit vector |
| `faceForward(n, i, ref)` | Flip n if dot(ref, i) < 0 |
| `reflect(i, n)` | Reflection vector |
| `refract(i, n, eta)` | Refraction vector |

```wgsl title="Common vector operations"
let dir = normalize(target - position);
let dist = distance(a, b);
let reflection = reflect(incident, normal);
```

### Component Access

| Function | Description |
|----------|-------------|
| `all(v)` | True if all components true |
| `any(v)` | True if any component true |
| `select(f, t, cond)` | Per-component ternary |

```wgsl title="select replaces ternary operator"
// WGSL has no ?: operator
let result = select(falseValue, trueValue, condition);

// Component-wise selection
let mixed = select(vec3f(0), vec3f(1), vec3<bool>(true, false, true));
// Result: vec3f(1, 0, 1)
```

## Interpolation

| Function | Description |
|----------|-------------|
| `mix(a, b, t)` | Linear interpolation: a*(1-t) + b*t |
| `step(edge, x)` | 0 if x < edge, else 1 |
| `smoothstep(lo, hi, x)` | Smooth Hermite interpolation |

```wgsl title="Interpolation examples"
let blend = mix(colorA, colorB, 0.5);

// smoothstep: 0 when x <= lo, 1 when x >= hi
let fade = smoothstep(0.0, 1.0, t);

// step: hard threshold
let mask = step(0.5, value);  // 0 or 1
```

## Texture Functions

### Sampling (Fragment Only)

| Function | Description |
|----------|-------------|
| `textureSample(t, s, coords)` | Sample with filtering |
| `textureSampleBias(t, s, coords, bias)` | Sample with LOD bias |
| `textureSampleLevel(t, s, coords, level)` | Sample specific mip level |
| `textureSampleGrad(t, s, coords, ddx, ddy)` | Sample with explicit gradients |
| `textureSampleCompare(t, s, coords, ref)` | Depth comparison sample |

```wgsl title="Texture sampling"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureSample(tex, samp, uv);
}
```

:::caution[Fragment Shader Only]
`textureSample` requires implicit derivatives, available only in fragment shaders. Use `textureSampleLevel` in vertex/compute shaders.
:::

### Direct Access

| Function | Description |
|----------|-------------|
| `textureLoad(t, coords, level)` | Load exact texel |
| `textureStore(t, coords, value)` | Write to storage texture |
| `textureDimensions(t)` | Get texture size |
| `textureNumLayers(t)` | Get array layer count |
| `textureNumLevels(t)` | Get mip level count |
| `textureNumSamples(t)` | Get MSAA sample count |

```wgsl title="Direct texel access"
// Read exact pixel (no filtering)
let texel = textureLoad(tex, vec2i(x, y), 0);

// Get texture dimensions
let size = textureDimensions(tex, 0);  // Returns vec2u
```

## Derivative Functions

Fragment shader only—compute rate of change across pixels.

| Function | Description |
|----------|-------------|
| `dpdx(v)` | Partial derivative w.r.t. x |
| `dpdy(v)` | Partial derivative w.r.t. y |
| `fwidth(v)` | abs(dpdx) + abs(dpdy) |
| `dpdxCoarse(v)` | Coarse x derivative |
| `dpdyCoarse(v)` | Coarse y derivative |
| `dpdxFine(v)` | Fine x derivative |
| `dpdyFine(v)` | Fine y derivative |

```wgsl title="Anti-aliased edge detection"
@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let value = someFunction(uv);

  // fwidth gives rate of change across pixel
  let edge = fwidth(value);
  let antialiased = smoothstep(0.0, edge, value);

  return vec4f(antialiased, antialiased, antialiased, 1.0);
}
```

:::tip[Use Cases]
- **Procedural textures**: Anti-alias edges
- **Text rendering**: SDF font smoothing
- **Normal mapping**: Compute tangent space
:::

## Pack/Unpack Functions

Compress/decompress data for efficient storage.

### Packing (vec → u32)

| Function | Description |
|----------|-------------|
| `pack4x8snorm(v)` | 4 floats [-1,1] → 4 signed bytes |
| `pack4x8unorm(v)` | 4 floats [0,1] → 4 unsigned bytes |
| `pack2x16snorm(v)` | 2 floats [-1,1] → 2 signed shorts |
| `pack2x16unorm(v)` | 2 floats [0,1] → 2 unsigned shorts |
| `pack2x16float(v)` | 2 floats → 2 half floats |

### Unpacking (u32 → vec)

| Function | Description |
|----------|-------------|
| `unpack4x8snorm(u)` | 4 signed bytes → vec4f [-1,1] |
| `unpack4x8unorm(u)` | 4 unsigned bytes → vec4f [0,1] |
| `unpack2x16snorm(u)` | 2 signed shorts → vec2f [-1,1] |
| `unpack2x16unorm(u)` | 2 unsigned shorts → vec2f [0,1] |
| `unpack2x16float(u)` | 2 half floats → vec2f |

```wgsl title="Pack/unpack normals"
// Pack normal into single u32
let packed = pack4x8snorm(vec4f(normal, 0.0));

// Unpack back to vec4f
let unpacked = unpack4x8snorm(packed);
let normal = unpacked.xyz;
```

## Matrix Functions

| Function | Description |
|----------|-------------|
| `transpose(m)` | Transpose matrix |
| `determinant(m)` | Matrix determinant |

```wgsl title="Matrix operations"
let normalMatrix = transpose(inverse3x3(modelMatrix));
```

:::note[No Built-in Inverse]
WGSL has no `inverse()` function. Implement manually or compute on CPU.
:::

## Atomic Functions

For `atomic<T>` types in workgroup or storage memory.

| Function | Description |
|----------|-------------|
| `atomicLoad(a)` | Read value |
| `atomicStore(a, v)` | Write value |
| `atomicAdd(a, v)` | Add and return old |
| `atomicSub(a, v)` | Subtract and return old |
| `atomicMax(a, v)` | Max and return old |
| `atomicMin(a, v)` | Min and return old |
| `atomicAnd(a, v)` | Bitwise AND and return old |
| `atomicOr(a, v)` | Bitwise OR and return old |
| `atomicXor(a, v)` | Bitwise XOR and return old |
| `atomicExchange(a, v)` | Swap and return old |
| `atomicCompareExchangeWeak(a, cmp, v)` | CAS operation |

```wgsl title="Atomic counter"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(64)
fn main() {
  let oldValue = atomicAdd(&counter, 1u);
}
```

## Synchronization

| Function | Description |
|----------|-------------|
| `workgroupBarrier()` | Sync all invocations in workgroup |
| `storageBarrier()` | Ensure storage writes visible |
| `textureBarrier()` | Ensure texture writes visible |

## Bit Manipulation

| Function | Description |
|----------|-------------|
| `countOneBits(v)` | Count set bits |
| `countLeadingZeros(v)` | Leading zero count |
| `countTrailingZeros(v)` | Trailing zero count |
| `firstLeadingBit(v)` | Position of highest set bit |
| `firstTrailingBit(v)` | Position of lowest set bit |
| `reverseBits(v)` | Reverse bit order |
| `extractBits(v, offset, count)` | Extract bit field |
| `insertBits(v, n, offset, count)` | Insert bit field |

## Resources

:::note[Official References]
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [WebGPU Fundamentals: WGSL Reference](https://webgpufundamentals.org/webgpu/lessons/webgpu-wgsl-function-reference.html)
- [Tour of WGSL](https://google.github.io/tour-of-wgsl/)
:::
