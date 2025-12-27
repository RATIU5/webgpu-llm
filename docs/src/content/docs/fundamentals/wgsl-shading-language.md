---
title: WGSL Shading Language
sidebar:
  order: 40
---

## Overview

WGSL (WebGPU Shading Language) is the shading language for WebGPU, designed to be safe, expressive, and portable across GPU architectures. Drawing from Rust's syntax and safety principles, WGSL provides strong typing, eliminates undefined behavior, and compiles to SPIR-V, Metal Shading Language, or DXIL for cross-platform execution.

WGSL enforces strict type safety and explicit handling of operations that other shading languages leave implicit. This design ensures shaders run safely in browsers while providing the performance needed for modern GPU programming.

## Scalar Types

### Boolean

```wgsl
var is_active: bool = true;
var should_render: bool = false;
```

### Integers

32-bit signed (`i32`) and unsigned (`u32`) integers:

```wgsl
var signed_value: i32 = -42;
var unsigned_value: u32 = 100u;  // 'u' suffix for unsigned

var sum: i32 = signed_value + 10;
var product: u32 = unsigned_value * 2u;
```

Integer division truncates toward zero; overflow uses wrapping semantics.

### Floating-Point

```wgsl
var pi: f32 = 3.14159265;
var speed_of_light: f32 = 2.998e8;  // Scientific notation
```

### f16 (Optional Feature)

16-bit floats require the `f16` extension:

```wgsl
enable f16;
var half_precision: f16 = 1.5h;
```

### Type Casting

WGSL requires explicit type conversions:

```wgsl
var int_value: i32 = 42;
var float_value: f32 = f32(int_value);
var unsigned_value: u32 = u32(int_value);
```

## Vector Types

Vectors represent positions, colors, and directions with 2, 3, or 4 components.

### Declaration

```wgsl
// Explicit types
var position: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
var color: vec4<f32> = vec4<f32>(1.0, 0.5, 0.0, 1.0);

// Shorthand types
var pos2: vec2f = vec2f(1.0, 2.0);
var pos3: vec3f = vec3f(1.0, 2.0, 3.0);
var offset: vec3i = vec3i(0, 1, -1);
var mask: vec4u = vec4u(1u, 1u, 0u, 1u);
```

### Constructors

```wgsl
// Scalar broadcast
var ones: vec3f = vec3f(1.0);  // (1.0, 1.0, 1.0)

// Mixed construction
var xy: vec2f = vec2f(1.0, 2.0);
var xyz: vec3f = vec3f(xy, 3.0);  // (1.0, 2.0, 3.0)
var xyzw: vec4f = vec4f(xy, 3.0, 4.0);
```

### Swizzling

Access and rearrange components using `xyzw` or `rgba`:

```wgsl
var position: vec4f = vec4f(1.0, 2.0, 3.0, 4.0);

var x: f32 = position.x;
var xy: vec2f = position.xy;
var zyx: vec3f = position.zyx;  // Reversed
var xxx: vec3f = position.xxx;  // Repeated

var color: vec4f = vec4f(1.0, 0.5, 0.0, 1.0);
var rgb: vec3f = color.rgb;
var bgr: vec3f = color.bgr;
```

Do not mix `xyzw` and `rgba` in the same swizzle.

### Operations

```wgsl
var a: vec3f = vec3f(1.0, 2.0, 3.0);
var b: vec3f = vec3f(4.0, 5.0, 6.0);

var sum: vec3f = a + b;           // Component-wise
var product: vec3f = a * b;       // Component-wise
var scaled: vec3f = a * 2.0;      // Scalar multiply

var len: f32 = length(a);
var normalized: vec3f = normalize(a);
var d: f32 = dot(a, b);
var cross_prod: vec3f = cross(a, b);  // vec3 only
```

## Matrix Types

Matrices are used for transformations. WGSL uses column-major storage.

### Declaration

```wgsl
var transform: mat4x4f = mat4x4f(
    1.0, 0.0, 0.0, 0.0,  // Column 0
    0.0, 1.0, 0.0, 0.0,  // Column 1
    0.0, 0.0, 1.0, 0.0,  // Column 2
    0.0, 0.0, 0.0, 1.0   // Column 3
);
```

### Column-Major Storage

Values are stored by column, not row:

```wgsl
var m: mat3x3f = mat3x3f(
    1.0, 2.0, 3.0,  // Column 0 (not row!)
    4.0, 5.0, 6.0,  // Column 1
    7.0, 8.0, 9.0   // Column 2
);

// Represents:
// [ 1.0  4.0  7.0 ]
// [ 2.0  5.0  8.0 ]
// [ 3.0  6.0  9.0 ]

var col0: vec3f = m[0];      // (1.0, 2.0, 3.0)
var element: f32 = m[1][2];  // Column 1, row 2 = 6.0
```

### Matrix Operations

```wgsl
var mat_a: mat4x4f = mat4x4f(/* ... */);
var position: vec4f = vec4f(1.0, 2.0, 3.0, 1.0);

var combined: mat4x4f = mat_a * mat_b;
var transformed: vec4f = mat_a * position;
var transposed: mat4x4f = transpose(mat_a);
```

## Arrays and Structs

### Fixed-Size Arrays

```wgsl
var colors: array<vec3f, 4> = array<vec3f, 4>(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0),
    vec3f(0.0, 0.0, 1.0),
    vec3f(1.0, 1.0, 0.0)
);

var first: vec3f = colors[0];
```

### Runtime-Sized Arrays

Only allowed as the last member of a struct in storage buffers:

```wgsl
struct ParticleBuffer {
    count: u32,
    particles: array<vec4f>
}

@group(0) @binding(0) var<storage, read> data: ParticleBuffer;
```

### Structs

```wgsl
struct Vertex {
    position: vec3f,
    normal: vec3f,
    uv: vec2f,
    color: vec4f
}

struct Light {
    position: vec3f,
    color: vec3f,
    intensity: f32,
    radius: f32
}

var light: Light = Light(
    vec3f(10.0, 10.0, 10.0),
    vec3f(1.0, 1.0, 1.0),
    100.0,
    50.0
);

var pos = light.position;
```

## Memory Alignment

WGSL enforces strict alignment rules that differ from typical CPU layouts.

### Alignment Rules

- `f32`, `i32`, `u32`: 4-byte alignment
- `vec2<T>`: 8-byte alignment
- `vec3<T>`: **16-byte alignment** (despite being 12 bytes)
- `vec4<T>`: 16-byte alignment
- `mat4x4<f32>`: 16-byte alignment

### The vec3 Padding Issue

`vec3` aligns to 16 bytes, causing unexpected struct sizes:

```wgsl
// Problematic layout
struct TwoVec3s {
    a: vec3f,  // Offset 0, size 12, padded to 16
    b: vec3f   // Offset 16, size 12, padded to 16
}
// Total: 32 bytes (not 24)

// Solution: Use vec4 or explicit padding
struct BetterLayout {
    a: vec4f,  // Set w to 0.0 if unused
    b: vec4f
}

struct ExplicitPadding {
    a: vec3f,
    _pad1: f32,
    b: vec3f,
    _pad2: f32
}
```

### Minimizing Padding

Order struct members by alignment (largest first):

```wgsl
// Suboptimal - lots of padding
struct Wasteful {
    a: f32,      // Offset 0
    b: vec4f,    // Offset 16 (12 bytes padding)
    c: f32,      // Offset 32
    d: vec4f     // Offset 48 (12 bytes padding)
}

// Optimized - minimal padding
struct Efficient {
    b: vec4f,    // Offset 0
    d: vec4f,    // Offset 16
    a: f32,      // Offset 32
    c: f32       // Offset 36
}
```

## Functions

### Syntax

```wgsl
fn add(a: f32, b: f32) -> f32 {
    return a + b;
}

fn transform_point(point: vec3f, matrix: mat4x4f, scale: f32) -> vec3f {
    let transformed = matrix * vec4f(point, 1.0);
    return transformed.xyz * scale;
}
```

Parameters are passed by value. For multiple return values, use a struct.

### Entry Points

```wgsl
@vertex
fn vertex_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5)
    );
    return vec4f(positions[idx], 0.0, 1.0);
}

@fragment
fn fragment_main(@builtin(position) coord: vec4f) -> @location(0) vec4f {
    return vec4f(1.0, 0.5, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x + id.y * 256u;
}
```

## Attributes

### @location

Shader stage inputs and outputs:

```wgsl
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) tex_coords: vec2f
}
```

### @group and @binding

Resource bindings:

```wgsl
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;

@group(1) @binding(0) var my_texture: texture_2d<f32>;
@group(1) @binding(1) var my_sampler: sampler;
```

### @builtin

Access GPU-provided values:

```wgsl
// Vertex shader
@builtin(vertex_index) v_idx: u32
@builtin(instance_index) inst_idx: u32

// Fragment shader
@builtin(position) frag_coord: vec4f
@builtin(front_facing) is_front: bool

// Compute shader
@builtin(global_invocation_id) global_id: vec3u
@builtin(local_invocation_id) local_id: vec3u
@builtin(workgroup_id) workgroup_id: vec3u
@builtin(local_invocation_index) local_idx: u32
```

### @workgroup_size

Compute shader workgroup dimensions:

```wgsl
@compute @workgroup_size(256)
fn compute_1d(@builtin(global_invocation_id) id: vec3u) { }

@compute @workgroup_size(16, 16)
fn compute_2d(@builtin(global_invocation_id) id: vec3u) { }

@compute @workgroup_size(8, 8, 4)
fn compute_3d(@builtin(global_invocation_id) id: vec3u) { }
```

## Control Flow

### Conditionals

```wgsl
fn classify(x: f32) -> i32 {
    if (x > 0.0) {
        return 1;
    } else if (x < 0.0) {
        return -1;
    } else {
        return 0;
    }
}

// Braces are always required
if (condition) {
    do_something();
}
```

### Switch

```wgsl
fn get_color(channel: u32) -> vec3f {
    switch (channel) {
        case 0u: { return vec3f(1.0, 0.0, 0.0); }
        case 1u: { return vec3f(0.0, 1.0, 0.0); }
        case 2u: { return vec3f(0.0, 0.0, 1.0); }
        default: { return vec3f(0.0, 0.0, 0.0); }
    }
}

// Multiple cases
switch (channel) {
    case 0u, 1u, 2u: { return true; }
    default: { return false; }
}
```

### Loops

```wgsl
// For loop
for (var i = 0; i < 100; i++) {
    sum += data[i];
}

// While loop
while (value < threshold) {
    value += 0.1;
    count++;
}

// Loop with break
loop {
    if (i >= 10) { break; }
    i++;
}
```

### select()

Branchless conditional:

```wgsl
var max_val = select(a, b, b > a);  // b > a ? b : a
var clamped = select(x, 0.0, x < 0.0);

// Component-wise for vectors
var max_vec = select(a_vec, b_vec, b_vec > a_vec);
```

### discard

Kill fragment in fragment shaders:

```wgsl
@fragment
fn alpha_test(@location(0) uv: vec2f) -> @location(0) vec4f {
    let color = textureSample(tex, samp, uv);
    if (color.a < 0.5) {
        discard;
    }
    return color;
}
```

## Built-in Functions

### Math

```wgsl
// Trigonometric
sin(angle); cos(angle); tan(angle);
asin(x); acos(x); atan(x); atan2(y, x);

// Exponential
exp(x); exp2(x); log(x); log2(x);
pow(base, exp); sqrt(x); inverseSqrt(x);

// Common
abs(x); sign(x);
floor(x); ceil(x); round(x); fract(x);
min(a, b); max(a, b); clamp(x, lo, hi);
mix(a, b, t);  // Linear interpolation
step(edge, x); smoothstep(lo, hi, x);
```

### Vector and Matrix

```wgsl
length(v); distance(a, b); normalize(v);
dot(a, b); cross(a, b);
reflect(incident, normal);
refract(incident, normal, eta);
transpose(m); determinant(m);
```

### Textures

```wgsl
// Sampling (fragment shaders only)
textureSample(tex, sampler, uv);
textureSampleBias(tex, sampler, uv, bias);
textureSampleLevel(tex, sampler, uv, level);

// Loading (any shader)
textureLoad(tex, coord, mip_level);
textureDimensions(tex);
textureNumLevels(tex);

// Storage textures
textureStore(tex, coord, value);
```

### Synchronization

```wgsl
var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn compute(@builtin(local_invocation_index) idx: u32) {
    shared[idx] = f32(idx);
    workgroupBarrier();  // Wait for all writes
    let neighbor = shared[(idx + 1u) % 256u];
}
```

### Atomics

```wgsl
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(64)
fn atomic_ops(@builtin(global_invocation_id) id: vec3u) {
    let old = atomicAdd(&counter, 1u);
    atomicSub(&counter, 1u);
    atomicMax(&counter, 100u);
    atomicMin(&counter, 0u);
    atomicExchange(&counter, 42u);
}
```

## Uniformity Requirements

Texture sampling with implicit derivatives requires uniform control flow—all threads in a group must take the same path:

```wgsl
// Invalid: non-uniform control flow
@fragment
fn bad(@location(0) uv: vec2f, @location(1) flag: f32) -> @location(0) vec4f {
    if (flag > 0.5) {  // Non-uniform!
        return textureSample(tex, samp, uv);  // Error
    }
    return vec4f(0.0);
}

// Valid: sample unconditionally, then select
@fragment
fn good(@location(0) uv: vec2f, @location(1) flag: f32) -> @location(0) vec4f {
    let sampled = textureSample(tex, samp, uv);
    return select(vec4f(0.0), sampled, flag > 0.5);
}
```

## Performance Tips

Cache uniform values in local variables:

```wgsl
// Suboptimal: multiple reads
for (var i = 0u; i < 100u; i++) {
    output[i] = input[i] * uniforms.scale + uniforms.offset;
}

// Better: cache values
let scale = uniforms.scale;
let offset = uniforms.offset;
for (var i = 0u; i < 100u; i++) {
    output[i] = input[i] * scale + offset;
}
```

Use workgroup memory for shared data:

```wgsl
var<workgroup> cache: array<vec4f, 256>;

@compute @workgroup_size(256)
fn efficient(@builtin(local_invocation_index) idx: u32) {
    cache[idx] = storage_buffer[idx];
    workgroupBarrier();
    let value = cache[idx];  // Fast access
}
```

---

WGSL prioritizes safety and portability while maintaining high performance. Its Rust-inspired syntax and strict type system catch errors at compile time, and explicit memory layout rules ensure consistent behavior across platforms. Understanding these fundamentals enables writing efficient, maintainable GPU code for WebGPU applications.
