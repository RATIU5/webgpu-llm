# WGSL Shading Language

WGSL (WebGPU Shading Language) is the official shading language for WebGPU, designed to be safe, expressive, and portable across different GPU architectures. Drawing inspiration from Rust's syntax and safety principles, WGSL provides a modern, strongly-typed language for writing compute and graphics shaders that execute on the GPU.

Unlike its predecessors (GLSL and HLSL), WGSL was designed from the ground up with web security in mind. The language enforces strict type safety, eliminates undefined behavior, and requires explicit handling of many operations that other shading languages leave implicit. This design philosophy ensures that WGSL shaders can run safely in web browsers without compromising system security, while still providing the performance and expressiveness needed for modern GPU programming.

WGSL's syntax will feel familiar to Rust developers, featuring explicit type annotations, clear function signatures, and a focus on readability. The language compiles to an intermediate representation that can be efficiently translated to native GPU instruction sets like SPIR-V, Metal Shading Language, or DXIL, making it truly portable across platforms.

## Scalar Types

WGSL provides five fundamental scalar types that form the building blocks for all other data types:

### Boolean Type

The `bool` type represents logical true/false values:

```wgsl
var is_active: bool = true;
var should_render: bool = false;

if (is_active && should_render) {
    // Execute rendering logic
}
```

### Integer Types

WGSL supports both signed and unsigned 32-bit integers:

```wgsl
var signed_value: i32 = -42;
var unsigned_value: u32 = 100u;  // Note the 'u' suffix for unsigned literals

// Integer arithmetic
var sum: i32 = signed_value + 10;
var product: u32 = unsigned_value * 2u;
```

The `i32` type represents signed integers in the range [-2,147,483,648, 2,147,483,647], while `u32` represents unsigned integers from [0, 4,294,967,295]. Integer division truncates toward zero, and overflow behavior is well-defined (wrapping semantics).

### Floating-Point Types

The primary floating-point type is `f32`, a 32-bit IEEE 754 floating-point number:

```wgsl
var pi: f32 = 3.14159265;
var e: f32 = 2.71828;
var fraction: f32 = 0.5;

// Floating-point literals can use scientific notation
var speed_of_light: f32 = 2.998e8;
var planck: f32 = 6.626e-34;
```

### f16 Optional Feature

WGSL also supports 16-bit floating-point numbers (`f16`), but this is an optional feature that may not be available on all hardware:

```wgsl
// Requires the 'f16' extension to be enabled
enable f16;

var half_precision: f16 = 1.5h;  // Note the 'h' suffix
```

The `f16` type provides lower precision but uses half the memory, making it useful for certain graphics applications where full precision isn't necessary.

### Type Literals and Casting

WGSL requires explicit type conversions between different numeric types:

```wgsl
var int_value: i32 = 42;
var float_value: f32 = f32(int_value);  // Explicit cast

var unsigned_value: u32 = u32(int_value);  // i32 to u32

// Type constructors can also be used with expressions
var result: f32 = f32(int_value * 2 + 10);

// Boolean conversions
var b: bool = bool(1);  // Non-zero values become true
```

Abstract numeric literals (those without type suffixes) will infer their type from context:

```wgsl
var inferred_float: f32 = 3.14;  // Inferred as f32
var inferred_int: i32 = 42;      // Inferred as i32
```

## Vector Types

Vectors are fundamental to GPU programming, representing positions, colors, directions, and more. WGSL provides comprehensive vector support with intuitive syntax.

### Vector Declaration

Vectors are declared using the `vec` prefix followed by the dimension (2, 3, or 4) and element type:

```wgsl
// Explicit type syntax
var position: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
var color: vec4<f32> = vec4<f32>(1.0, 0.5, 0.0, 1.0);
var indices: vec2<i32> = vec2<i32>(0, 1);
var flags: vec4<u32> = vec4<u32>(1u, 0u, 1u, 1u);
```

### Shorthand Vector Types

WGSL provides convenient shorthand notations for common vector types:

```wgsl
// Float vectors (most common in graphics)
var pos2: vec2f = vec2f(1.0, 2.0);        // vec2<f32>
var pos3: vec3f = vec3f(1.0, 2.0, 3.0);   // vec3<f32>
var pos4: vec4f = vec4f(1.0, 2.0, 3.0, 4.0);  // vec4<f32>

// Signed integer vectors
var offset2: vec2i = vec2i(-1, 2);        // vec2<i32>
var offset3: vec3i = vec3i(0, 1, -1);     // vec3<i32>

// Unsigned integer vectors
var mask2: vec2u = vec2u(1u, 0u);         // vec2<u32>
var mask4: vec4u = vec4u(1u, 1u, 0u, 1u); // vec4<u32>

// Half-precision vectors (requires f16 extension)
enable f16;
var compact: vec3h = vec3h(1.0h, 2.0h, 3.0h);  // vec3<f16>
```

### Vector Constructors

WGSL offers flexible vector construction from various sources:

```wgsl
// Scalar broadcast - all components set to same value
var ones: vec3f = vec3f(1.0);  // (1.0, 1.0, 1.0)

// Individual components
var rgb: vec3f = vec3f(0.8, 0.2, 0.5);

// Mixing scalars and vectors
var xy: vec2f = vec2f(1.0, 2.0);
var xyz: vec3f = vec3f(xy, 3.0);  // (1.0, 2.0, 3.0)
var xyzw: vec4f = vec4f(xy, 3.0, 4.0);  // (1.0, 2.0, 3.0, 4.0)

// Vector composition
var first: vec2f = vec2f(1.0, 2.0);
var second: vec2f = vec2f(3.0, 4.0);
var combined: vec4f = vec4f(first, second);  // (1.0, 2.0, 3.0, 4.0)
```

### Swizzling

Swizzling allows you to rearrange and access vector components using intuitive notation. WGSL supports two naming schemes: `xyzw` (for positions/directions) and `rgba` (for colors):

```wgsl
var position: vec4f = vec4f(1.0, 2.0, 3.0, 4.0);

// Single component access
var x: f32 = position.x;  // 1.0
var y: f32 = position.y;  // 2.0

// Swizzling to create new vectors
var xy: vec2f = position.xy;      // (1.0, 2.0)
var zw: vec2f = position.zw;      // (3.0, 4.0)
var yx: vec2f = position.yx;      // (2.0, 1.0) - reversed!

// Three-component swizzles
var xyz: vec3f = position.xyz;    // (1.0, 2.0, 3.0)
var zyx: vec3f = position.zyx;    // (3.0, 2.0, 1.0)

// Repetition is allowed in read contexts
var xxx: vec3f = position.xxx;    // (1.0, 1.0, 1.0)

// Color swizzling (rgba)
var color: vec4f = vec4f(1.0, 0.5, 0.0, 1.0);
var rgb: vec3f = color.rgb;       // (1.0, 0.5, 0.0)
var bgr: vec3f = color.bgr;       // (0.0, 0.5, 1.0)
var alpha: f32 = color.a;         // 1.0

// Cannot mix xyzw and rgba in same swizzle
// var invalid = position.xga;  // ERROR!
```

### Vector Operations

Vectors support component-wise arithmetic operations:

```wgsl
var a: vec3f = vec3f(1.0, 2.0, 3.0);
var b: vec3f = vec3f(4.0, 5.0, 6.0);

// Component-wise addition
var sum: vec3f = a + b;  // (5.0, 7.0, 9.0)

// Component-wise multiplication
var product: vec3f = a * b;  // (4.0, 10.0, 18.0)

// Scalar multiplication
var scaled: vec3f = a * 2.0;  // (2.0, 4.0, 6.0)

// Component-wise division
var quotient: vec3f = a / b;  // (0.25, 0.4, 0.5)

// Vector negation
var negated: vec3f = -a;  // (-1.0, -2.0, -3.0)

// Built-in functions for vectors
var len: f32 = length(a);           // Vector magnitude
var normalized: vec3f = normalize(a);  // Unit vector
var d: f32 = dot(a, b);             // Dot product
var cross_prod: vec3f = cross(a, b);  // Cross product (vec3 only)
```

## Matrix Types

Matrices in WGSL are used primarily for transformations in graphics programming. Understanding their storage and multiplication order is crucial.

### Matrix Declaration

Matrices are declared with dimensions and element type:

```wgsl
// Common matrix types
var transform2d: mat2x2<f32> = mat2x2<f32>(
    1.0, 0.0,  // First column
    0.0, 1.0   // Second column
);

var transform3d: mat3x3<f32> = mat3x3<f32>(
    1.0, 0.0, 0.0,  // First column
    0.0, 1.0, 0.0,  // Second column
    0.0, 0.0, 1.0   // Third column
);

var projection: mat4x4<f32> = mat4x4<f32>(
    1.0, 0.0, 0.0, 0.0,  // Column 0
    0.0, 1.0, 0.0, 0.0,  // Column 1
    0.0, 0.0, 1.0, 0.0,  // Column 2
    0.0, 0.0, 0.0, 1.0   // Column 3
);

// Shorthand for float matrices
var mat2: mat2x2f = mat2x2f(1.0, 0.0, 0.0, 1.0);
var mat3: mat3x3f = mat3x3f(/* 9 values */);
var mat4: mat4x4f = mat4x4f(/* 16 values */);
```

WGSL also supports non-square matrices:

```wgsl
var mat2x3: mat2x3<f32>;  // 2 columns, 3 rows
var mat3x2: mat3x2<f32>;  // 3 columns, 2 rows
var mat4x3: mat4x3<f32>;  // 4 columns, 3 rows
```

### Column-Major Storage

WGSL uses column-major storage for matrices, which is important when interfacing with CPU code:

```wgsl
var m: mat3x3f = mat3x3f(
    1.0, 2.0, 3.0,  // Column 0 (not row 0!)
    4.0, 5.0, 6.0,  // Column 1
    7.0, 8.0, 9.0   // Column 2
);

// This matrix represents:
// [ 1.0  4.0  7.0 ]
// [ 2.0  5.0  8.0 ]
// [ 3.0  6.0  9.0 ]

// Accessing columns and elements
var col0: vec3f = m[0];     // (1.0, 2.0, 3.0)
var col1: vec3f = m[1];     // (4.0, 5.0, 6.0)
var element: f32 = m[1][2]; // Access column 1, row 2 = 6.0
```

### Matrix Multiplication

Matrix multiplication in WGSL follows mathematical conventions:

```wgsl
var mat_a: mat4x4f = mat4x4f(/* ... */);
var mat_b: mat4x4f = mat4x4f(/* ... */);
var position: vec4f = vec4f(1.0, 2.0, 3.0, 1.0);

// Matrix-matrix multiplication
var combined: mat4x4f = mat_a * mat_b;

// Matrix-vector multiplication (transforms the vector)
var transformed: vec4f = mat_a * position;

// Note: vector * matrix multiplies row vector on left
var row_mult: vec4f = position * mat_a;

// Common transformation matrices
fn translation_matrix(offset: vec3f) -> mat4x4f {
    return mat4x4f(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        offset.x, offset.y, offset.z, 1.0
    );
}

fn scale_matrix(scale: vec3f) -> mat4x4f {
    return mat4x4f(
        scale.x, 0.0, 0.0, 0.0,
        0.0, scale.y, 0.0, 0.0,
        0.0, 0.0, scale.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

// Transpose and other operations
var transposed: mat4x4f = transpose(mat_a);
var determinant: f32 = determinant(mat_a);
```

## Arrays and Structs

Arrays and structs allow you to create complex data structures for GPU processing.

### Fixed-Size Arrays

Fixed-size arrays have a compile-time known length:

```wgsl
// Array declaration with explicit size
var colors: array<vec3f, 4> = array<vec3f, 4>(
    vec3f(1.0, 0.0, 0.0),  // Red
    vec3f(0.0, 1.0, 0.0),  // Green
    vec3f(0.0, 0.0, 1.0),  // Blue
    vec3f(1.0, 1.0, 0.0)   // Yellow
);

// Array indexing
var first_color: vec3f = colors[0];
var second_color: vec3f = colors[1];

// Arrays in functions
fn get_palette_color(index: u32) -> vec3f {
    var palette: array<vec3f, 3> = array<vec3f, 3>(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0)
    );
    return palette[index];
}

// Multi-dimensional arrays
var matrix_data: array<array<f32, 3>, 3>;  // 3x3 array of floats
matrix_data[0][0] = 1.0;
matrix_data[1][2] = 5.0;
```

### Runtime-Sized Arrays

Runtime-sized arrays can only appear as the last member of a struct in storage buffers:

```wgsl
struct ParticleBuffer {
    count: u32,
    particles: array<vec4f>  // Runtime-sized!
}

@group(0) @binding(0) var<storage, read> particle_data: ParticleBuffer;

fn process_particles() {
    let num_particles = particle_data.count;
    for (var i = 0u; i < num_particles; i++) {
        let particle = particle_data.particles[i];
        // Process particle...
    }
}
```

### Struct Definitions

Structs group related data together:

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

struct Material {
    albedo: vec3f,
    metallic: f32,
    roughness: f32,
    ambient_occlusion: f32
}

// Using structs
var vertex: Vertex;
vertex.position = vec3f(0.0, 1.0, 0.0);
vertex.normal = vec3f(0.0, 1.0, 0.0);
vertex.uv = vec2f(0.5, 0.5);
vertex.color = vec4f(1.0, 1.0, 1.0, 1.0);

// Struct constructors
var light: Light = Light(
    vec3f(10.0, 10.0, 10.0),  // position
    vec3f(1.0, 1.0, 1.0),     // color
    100.0,                     // intensity
    50.0                       // radius
);
```

### Member Access

Accessing struct members uses dot notation:

```wgsl
struct Camera {
    view_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    position: vec3f,
    direction: vec3f
}

var camera: Camera;
var view_proj = camera.projection_matrix * camera.view_matrix;
var cam_pos = camera.position;

// Nested structs
struct Transform {
    position: vec3f,
    rotation: vec4f,  // Quaternion
    scale: vec3f
}

struct GameObject {
    transform: Transform,
    color: vec4f,
    active: bool
}

var object: GameObject;
object.transform.position = vec3f(1.0, 2.0, 3.0);
object.transform.scale = vec3f(1.0, 1.0, 1.0);
object.color = vec4f(1.0, 0.5, 0.0, 1.0);
```

## Memory Layout and Alignment

Understanding memory layout is critical for correctly passing data between CPU and GPU.

### Alignment Rules

WGSL enforces strict alignment rules that can surprise developers coming from CPU programming:

```wgsl
// This struct has unexpected size!
struct Problematic {
    a: f32,      // Offset 0, size 4
    b: vec3f,    // Offset 16 (not 4!), size 12
    c: f32       // Offset 32 (not 16!), size 4
}
// Total size: 36 bytes (not 20 as you might expect)

// Why? vec3 has 16-byte alignment despite only being 12 bytes!
```

The alignment rules are:

- `f32`, `i32`, `u32`, `bool`: 4-byte alignment
- `vec2<T>`: 8-byte alignment
- `vec3<T>`: **16-byte alignment** (this is the surprising one!)
- `vec4<T>`: 16-byte alignment
- `mat2x2<f32>`: 8-byte alignment
- `mat3x3<f32>`, `mat4x4<f32>`: 16-byte alignment
- Structs: Alignment of largest member
- Arrays: Element alignment rounded up to multiple of 16

### Vector3 Padding Issue

The `vec3` padding to 16 bytes is the most common source of bugs:

```wgsl
// CPU might send this as 24 bytes (3 floats × 2)
// But GPU expects 32 bytes!
struct TwoVec3s {
    a: vec3f,  // Offset 0, size 12, alignment 16
    b: vec3f   // Offset 16, size 12, alignment 16
}
// Total: 32 bytes on GPU, but often packed as 24 on CPU

// Solution 1: Use vec4 instead
struct BetterLayout {
    a: vec4f,  // Just set w to 0.0 if unused
    b: vec4f
}

// Solution 2: Add explicit padding
struct ExplicitPadding {
    a: vec3f,
    _padding1: f32,  // Explicit pad
    b: vec3f,
    _padding2: f32
}
```

### @size and @align Attributes

You can manually control layout with attributes:

```wgsl
struct CustomLayout {
    @align(16) small_value: f32,  // Force 16-byte alignment
    @size(64) data: vec4f,         // Reserve 64 bytes (wastes space!)
    normal_value: vec4f
}

// Practical example: matching a C++ std140 layout
struct UniformBlock {
    @align(16) time: f32,
    @align(16) resolution: vec2f,
    @align(16) mouse: vec2f,
    view_matrix: mat4x4f
}
```

### Struct Padding

The compiler automatically adds padding to satisfy alignment:

```wgsl
struct AutoPadded {
    a: f32,      // Offset 0
    // 12 bytes padding inserted here
    b: vec4f,    // Offset 16
    c: f32,      // Offset 32
    // 12 bytes padding
    d: vec4f     // Offset 48
}

// To minimize padding, order members by alignment (largest first)
struct Optimized {
    d: vec4f,    // Offset 0
    b: vec4f,    // Offset 16
    a: f32,      // Offset 32
    c: f32       // Offset 36
    // Only 8 bytes padding at end to round to 16-byte multiple
}
```

## Functions

Functions in WGSL follow a clear, Rust-like syntax with explicit type annotations.

### Function Syntax

```wgsl
// Basic function with return value
fn add(a: f32, b: f32) -> f32 {
    return a + b;
}

// Function with no return value (returns void implicitly)
fn print_info(value: f32) {
    // Do something with value
    // No return statement needed
}

// Multiple parameters of different types
fn transform_point(point: vec3f, matrix: mat4x4f, scale: f32) -> vec3f {
    let transformed = matrix * vec4f(point, 1.0);
    return transformed.xyz * scale;
}

// Using structs as parameters
struct Ray {
    origin: vec3f,
    direction: vec3f
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.origin + ray.direction * t;
}
```

### Parameters and Return Types

All parameters are passed by value (copied). For large structures, this can impact performance:

```wgsl
// This copies the entire matrix (64 bytes)
fn expensive(m: mat4x4f) -> f32 {
    return determinant(m);
}

// For read-only large data, consider using pointers to storage
fn efficient(index: u32) -> f32 {
    // Read from storage buffer instead
    let m = matrices.data[index];
    return determinant(m);
}

// Multiple return values via struct
struct IntersectionResult {
    hit: bool,
    distance: f32,
    normal: vec3f
}

fn ray_sphere_intersection(ray: Ray, sphere_center: vec3f, radius: f32) -> IntersectionResult {
    var result: IntersectionResult;

    let oc = ray.origin - sphere_center;
    let a = dot(ray.direction, ray.direction);
    let b = 2.0 * dot(oc, ray.direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        result.hit = false;
        return result;
    }

    result.hit = true;
    result.distance = (-b - sqrt(discriminant)) / (2.0 * a);
    let hit_point = ray_at(ray, result.distance);
    result.normal = normalize(hit_point - sphere_center);

    return result;
}
```

### Entry Points

Entry points are special functions that define shader stages:

```wgsl
// Vertex shader entry point
@vertex
fn vertex_main(@builtin(vertex_index) vertex_idx: u32) -> @builtin(position) vec4f {
    var positions = array<vec2f, 3>(
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5)
    );

    return vec4f(positions[vertex_idx], 0.0, 1.0);
}

// Fragment shader entry point
@fragment
fn fragment_main(@builtin(position) coord: vec4f) -> @location(0) vec4f {
    return vec4f(1.0, 0.5, 0.0, 1.0);  // Orange color
}

// Compute shader entry point
@compute @workgroup_size(8, 8, 1)
fn compute_main(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x + global_id.y * 256u;
    // Process data at index
}
```

## Attributes

Attributes provide metadata about declarations, controlling how data flows through the pipeline.

### @location

Used for shader stage inputs and outputs:

```wgsl
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec4f
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) tex_coords: vec2f,
    @location(2) vertex_color: vec4f
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = vec4f(input.position, 1.0);
    output.world_normal = input.normal;
    output.tex_coords = input.uv;
    output.vertex_color = input.color;
    return output;
}
```

### @binding and @group

Bind resources to the shader:

```wgsl
// Uniform buffer
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Storage buffer (read-only)
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;

// Storage buffer (read-write)
@group(0) @binding(2) var<storage, read_write> output_data: array<vec4f>;

// Texture and sampler
@group(1) @binding(0) var my_texture: texture_2d<f32>;
@group(1) @binding(1) var my_sampler: sampler;

// Multiple bind groups organize related resources
@group(0) @binding(0) var<uniform> camera: Camera;      // Per-frame
@group(1) @binding(0) var<uniform> material: Material;  // Per-material
@group(2) @binding(0) var<uniform> object: Transform;   // Per-object
```

### @builtin

Access built-in values provided by the GPU:

```wgsl
// Vertex shader builtins
@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) inst_idx: u32
) -> @builtin(position) vec4f {
    // Use vertex and instance indices
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

// Fragment shader builtins
@fragment
fn fs_main(
    @builtin(position) frag_coord: vec4f,
    @builtin(front_facing) is_front: bool,
    @builtin(sample_index) sample_idx: u32,
    @location(0) color: vec4f
) -> @location(0) vec4f {
    if (!is_front) {
        return vec4f(0.5, 0.5, 0.5, 1.0);  // Back face color
    }
    return color;
}

// Compute shader builtins
@compute @workgroup_size(16, 16)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(num_workgroups) num_workgroups: vec3u
) {
    // Various ways to compute indices
    let flat_idx = global_id.x + global_id.y * 256u;
}
```

### @workgroup_size

Specifies compute shader workgroup dimensions:

```wgsl
// 1D workgroup (256 threads)
@compute @workgroup_size(256)
fn compute_1d(@builtin(global_invocation_id) id: vec3u) { }

// 2D workgroup (16×16 = 256 threads)
@compute @workgroup_size(16, 16)
fn compute_2d(@builtin(global_invocation_id) id: vec3u) { }

// 3D workgroup (8×8×4 = 256 threads)
@compute @workgroup_size(8, 8, 4)
fn compute_3d(@builtin(global_invocation_id) id: vec3u) { }

// Using overrides for configurable workgroup size
override block_size: u32 = 16;
@compute @workgroup_size(block_size, block_size)
fn configurable_compute(@builtin(global_invocation_id) id: vec3u) { }
```

## Operators and Control Flow

WGSL provides comprehensive operators and control flow structures.

### Arithmetic Operators

```wgsl
var a: f32 = 10.0;
var b: f32 = 3.0;

var sum = a + b;        // 13.0
var difference = a - b; // 7.0
var product = a * b;    // 30.0
var quotient = a / b;   // 3.333...
var remainder = a % b;  // 1.0 (floating-point modulo)

// Unary operators
var negation = -a;      // -10.0
```

### Logical Operators

```wgsl
var x: bool = true;
var y: bool = false;

var and_result = x && y;  // false (logical AND)
var or_result = x || y;   // true (logical OR)
var not_result = !x;      // false (logical NOT)

// Comparison operators
var a: f32 = 5.0;
var b: f32 = 10.0;

var equal = a == b;          // false
var not_equal = a != b;      // true
var less = a < b;            // true
var less_equal = a <= b;     // true
var greater = a > b;         // false
var greater_equal = a >= b;  // false
```

### Bitwise Operators

```wgsl
var a: u32 = 0b1100u;  // 12
var b: u32 = 0b1010u;  // 10

var and_bits = a & b;   // 0b1000 = 8
var or_bits = a | b;    // 0b1110 = 14
var xor_bits = a ^ b;   // 0b0110 = 6
var not_bits = ~a;      // Bitwise NOT
var left_shift = a << 2u;   // 48
var right_shift = a >> 1u;  // 6
```

### if/else Statements

```wgsl
fn classify_value(x: f32) -> i32 {
    if (x > 0.0) {
        return 1;
    } else if (x < 0.0) {
        return -1;
    } else {
        return 0;
    }
}

// Single-line if without braces is NOT allowed
// if (condition) do_something();  // ERROR!

// Always use braces
if (condition) {
    do_something();
}
```

### switch Statements

```wgsl
fn get_color_name(channel: u32) -> vec3f {
    switch (channel) {
        case 0u: {
            return vec3f(1.0, 0.0, 0.0);  // Red
        }
        case 1u: {
            return vec3f(0.0, 1.0, 0.0);  // Green
        }
        case 2u: {
            return vec3f(0.0, 0.0, 1.0);  // Blue
        }
        default: {
            return vec3f(0.0, 0.0, 0.0);  // Black
        }
    }
}

// Multiple cases can share code
fn is_primary_color(channel: u32) -> bool {
    switch (channel) {
        case 0u, 1u, 2u: {
            return true;
        }
        default: {
            return false;
        }
    }
}
```

### loop, for, and while

```wgsl
// Infinite loop with break
fn find_first_negative(data: array<f32, 10>) -> i32 {
    var i = 0;
    loop {
        if (i >= 10) {
            break;
        }
        if (data[i] < 0.0) {
            return i;
        }
        i++;
    }
    return -1;
}

// For loop (C-style)
fn sum_array(data: array<f32, 100>) -> f32 {
    var sum = 0.0;
    for (var i = 0; i < 100; i++) {
        sum += data[i];
    }
    return sum;
}

// While loop
fn count_until_threshold(threshold: f32) -> u32 {
    var count = 0u;
    var value = 0.0;
    while (value < threshold) {
        value += 0.1;
        count++;
    }
    return count;
}

// continue statement
fn sum_positive(data: array<f32, 10>) -> f32 {
    var sum = 0.0;
    for (var i = 0; i < 10; i++) {
        if (data[i] < 0.0) {
            continue;  // Skip negative values
        }
        sum += data[i];
    }
    return sum;
}
```

### select() Function

WGSL provides `select()` as an alternative to ternary operator:

```wgsl
// select(false_value, true_value, condition)
var max_val = select(a, b, b > a);  // Equivalent to: b > a ? b : a
var clamped = select(x, 0.0, x < 0.0);  // Clamp negative to 0

// Works with vectors (component-wise selection)
var a_vec = vec3f(1.0, 2.0, 3.0);
var b_vec = vec3f(4.0, 1.5, 2.0);
var max_vec = select(a_vec, b_vec, b_vec > a_vec);  // (4.0, 2.0, 3.0)
```

### return and discard

```wgsl
// Early return
fn safe_divide(a: f32, b: f32) -> f32 {
    if (b == 0.0) {
        return 0.0;  // Early return to avoid division by zero
    }
    return a / b;
}

// discard in fragment shaders (kills the fragment)
@fragment
fn fragment_with_alpha_test(
    @location(0) tex_coord: vec2f,
    @location(1) tex: texture_2d<f32>,
    @location(2) samp: sampler
) -> @location(0) vec4f {
    let color = textureSample(tex, samp, tex_coord);

    if (color.a < 0.5) {
        discard;  // Don't write this fragment
    }

    return color;
}
```

## Built-in Functions

WGSL provides a rich standard library of built-in functions.

### Math Functions

```wgsl
// Trigonometric
var angle = 3.14159 / 4.0;
var sine = sin(angle);
var cosine = cos(angle);
var tangent = tan(angle);
var arc_sine = asin(0.707);
var arc_cosine = acos(0.707);
var arc_tangent = atan(1.0);
var arc_tangent2 = atan2(y, x);  // Two-argument arctangent

// Exponential and logarithmic
var e_to_x = exp(2.0);
var two_to_x = exp2(3.0);     // 2^3 = 8
var log_e = log(2.718);
var log_2 = log2(8.0);        // 3.0
var power = pow(2.0, 8.0);    // 256.0
var square_root = sqrt(16.0); // 4.0
var inv_sqrt = inverseSqrt(4.0);  // 0.5 (optimized 1/sqrt)

// Common math
var absolute = abs(-5.0);       // 5.0
var sign_val = sign(-3.0);      // -1.0
var floor_val = floor(3.7);     // 3.0
var ceil_val = ceil(3.2);       // 4.0
var round_val = round(3.5);     // 4.0
var fract_val = fract(3.7);     // 0.7 (fractional part)
var min_val = min(5.0, 3.0);    // 3.0
var max_val = max(5.0, 3.0);    // 5.0
var clamped = clamp(x, 0.0, 1.0);  // Clamp x to [0, 1]
var mixed = mix(a, b, 0.5);     // Linear interpolation (lerp)
var stepped = step(0.5, x);     // 0 if x < 0.5, else 1
var smooth = smoothstep(0.0, 1.0, x);  // Smooth interpolation

// Practical example: smoothstep for fade effects
fn fade_by_distance(distance: f32) -> f32 {
    return smoothstep(100.0, 200.0, distance);  // Fade from 100 to 200 units
}
```

### Vector and Matrix Functions

```wgsl
var v1 = vec3f(1.0, 2.0, 3.0);
var v2 = vec3f(4.0, 5.0, 6.0);

// Vector operations
var len = length(v1);              // Vector magnitude
var dist = distance(v1, v2);       // Distance between points
var normalized = normalize(v1);    // Unit vector
var dot_prod = dot(v1, v2);        // Dot product
var cross_prod = cross(v1, v2);    // Cross product (vec3 only)
var reflected = reflect(v1, normalize(v2));  // Reflection
var refracted = refract(v1, normalize(v2), 1.5);  // Refraction

// Per-component operations
var face_forward = faceforward(n, i, nref);  // Orient normal toward viewer

// Matrix operations
var m = mat4x4f(/* ... */);
var transposed = transpose(m);
var determ = determinant(m);

// Practical lighting example
fn lambertian_lighting(normal: vec3f, light_dir: vec3f) -> f32 {
    return max(dot(normalize(normal), normalize(light_dir)), 0.0);
}
```

### Texture Functions

```wgsl
// Texture sampling (fragment shaders only)
@group(0) @binding(0) var my_texture: texture_2d<f32>;
@group(0) @binding(1) var my_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    // Basic texture sampling
    var color = textureSample(my_texture, my_sampler, uv);

    // Bias the mip level
    var biased = textureSampleBias(my_texture, my_sampler, uv, 2.0);

    // Explicit gradient sampling
    var grad = textureSampleGrad(my_texture, my_sampler, uv, dpdx, dpdy);

    // Sample specific mip level
    var level = textureSampleLevel(my_texture, my_sampler, uv, 2.0);

    return color;
}

// Texture loading (any shader stage)
@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let coord = vec2i(i32(id.x), i32(id.y));

    // Load texel without sampling
    var texel = textureLoad(my_texture, coord, 0);  // Mip level 0

    // Get texture dimensions
    var dims = textureDimensions(my_texture);
    var mip_dims = textureDimensions(my_texture, 1);  // Level 1 dimensions

    // Get number of mip levels
    var num_levels = textureNumLevels(my_texture);
}

// Texture writing (storage textures)
@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_write(@builtin(global_invocation_id) id: vec3u) {
    let coord = vec2i(i32(id.x), i32(id.y));
    let color = vec4f(1.0, 0.0, 0.0, 1.0);
    textureStore(output_texture, coord, color);
}
```

### Synchronization Functions

```wgsl
// Workgroup barrier (compute shaders)
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn compute_with_sync(@builtin(local_invocation_index) idx: u32) {
    // Write to shared memory
    shared_data[idx] = f32(idx);

    // Wait for all threads in workgroup
    workgroupBarrier();

    // Now safe to read from shared memory
    let neighbor = shared_data[(idx + 1u) % 256u];

    // Storage barrier for storage buffers
    storageBarrier();
}

// Practical example: parallel reduction
var<workgroup> reduction_buffer: array<f32, 256>;

@compute @workgroup_size(256)
fn parallel_sum(
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(global_invocation_id) global_id: vec3u
) {
    // Load data into shared memory
    reduction_buffer[local_idx] = input_data[global_id.x];
    workgroupBarrier();

    // Reduction tree
    var stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            reduction_buffer[local_idx] += reduction_buffer[local_idx + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes result
    if (local_idx == 0u) {
        output_data[global_id.x / 256u] = reduction_buffer[0];
    }
}
```

### Atomic Operations

```wgsl
// Atomic operations on storage buffers
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

@compute @workgroup_size(64)
fn atomic_operations(@builtin(global_invocation_id) id: vec3u) {
    // Atomic add (returns old value)
    let old_value = atomicAdd(&counter, 1u);

    // Other atomic operations
    atomicSub(&counter, 1u);
    atomicMax(&counter, 100u);
    atomicMin(&counter, 0u);
    atomicAnd(&counter, 0xFFu);
    atomicOr(&counter, 0x01u);
    atomicXor(&counter, 0xFFu);

    // Atomic exchange and compare-exchange
    let prev = atomicExchange(&counter, 42u);
    let swapped = atomicCompareExchangeWeak(&counter, 42u, 100u);

    // Load and store
    let value = atomicLoad(&counter);
    atomicStore(&counter, 0u);
}
```

## Best Practices and Common Pitfalls

### Memory Layout Pitfalls

**Problem:** vec3 padding causes data misalignment

```wgsl
// WRONG: CPU sends 24 bytes, GPU expects 32
struct BadLayout {
    position: vec3f,  // 12 bytes + 4 padding = 16
    velocity: vec3f   // 12 bytes + 4 padding = 16
}  // Total: 32 bytes on GPU, but CPU might pack as 24

// SOLUTION: Use vec4 or add explicit padding
struct GoodLayout {
    position: vec4f,  // 16 bytes (set w=0 if unused)
    velocity: vec4f   // 16 bytes
}  // Total: 32 bytes on both CPU and GPU
```

### Uniformity Analysis

**Problem:** Non-uniform control flow with texture sampling

```wgsl
// WRONG: Texture sampling in non-uniform control flow
@fragment
fn bad_fragment(@location(0) uv: vec2f, @location(1) should_sample: f32) -> @location(0) vec4f {
    if (should_sample > 0.5) {  // Non-uniform condition!
        return textureSample(tex, samp, uv);  // ERROR!
    }
    return vec4f(0.0);
}

// SOLUTION: Use select or ensure uniform control flow
@fragment
fn good_fragment(@location(0) uv: vec2f, @location(1) should_sample: f32) -> @location(0) vec4f {
    let sampled = textureSample(tex, samp, uv);
    let black = vec4f(0.0);
    return select(black, sampled, should_sample > 0.5);
}
```

### Performance Tips

1. **Minimize memory access:** Cache frequently-used values in registers

```wgsl
// BAD: Multiple reads from storage
for (var i = 0u; i < 100u; i++) {
    output[i] = input[i] * uniforms.scale + uniforms.offset;
}

// GOOD: Cache uniform values
let scale = uniforms.scale;
let offset = uniforms.offset;
for (var i = 0u; i < 100u; i++) {
    output[i] = input[i] * scale + offset;
}
```

2. **Use workgroup memory for compute shaders**

```wgsl
var<workgroup> shared_cache: array<vec4f, 256>;

@compute @workgroup_size(256)
fn efficient_compute(@builtin(local_invocation_index) idx: u32) {
    // Load once into shared memory
    shared_cache[idx] = slow_storage_buffer[idx];
    workgroupBarrier();

    // Fast access from workgroup memory
    let value = shared_cache[idx];
}
```

3. **Avoid dynamic indexing when possible**

```wgsl
// SLOWER: Dynamic array indexing
var result = data[dynamic_index];

// FASTER: Unrolled or compile-time known indices
var result = data[0] + data[1] + data[2];
```

### Common Mistakes

**Forgetting swizzle order:**

```wgsl
var rgba = vec4f(1.0, 0.5, 0.0, 1.0);
var bgr = rgba.bgr;  // Correct: (0.0, 0.5, 1.0)
// var invalid = rgba.rbg;  // Error: invalid swizzle
```

**Integer division truncation:**

```wgsl
var a: i32 = 7;
var b: i32 = 2;
var result = a / b;  // 3, not 3.5!

// For float division, cast first
var float_result = f32(a) / f32(b);  // 3.5
```

**Forgetting that WGSL is column-major:**

```wgsl
// This is column 0, then column 1, NOT row 0, then row 1
var m = mat2x2f(
    1.0, 2.0,  // Column 0
    3.0, 4.0   // Column 1
);
// Represents: | 1.0  3.0 |
//             | 2.0  4.0 |
```

### Debugging Tips

1. **Use color output for debugging fragment shaders:**

```wgsl
@fragment
fn debug_fragment(@location(0) value: f32) -> @location(0) vec4f {
    // Visualize scalar as red intensity
    return vec4f(value, 0.0, 0.0, 1.0);

    // Visualize normal vectors
    // return vec4f(normal * 0.5 + 0.5, 1.0);
}
```

2. **Write debug values to storage buffers:**

```wgsl
@group(0) @binding(0) var<storage, read_write> debug_output: array<f32>;

@compute @workgroup_size(64)
fn debug_compute(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    debug_output[index] = some_computed_value;
}
```

3. **Validate shader compilation errors carefully:** WGSL provides detailed error messages about type mismatches, uniformity violations, and invalid memory access patterns.

---

WGSL represents a modern approach to GPU shading languages, prioritizing safety and portability while maintaining high performance. Its Rust-inspired syntax and strict type system help catch errors at compile time, and its explicit memory layout rules ensure consistent behavior across platforms. By understanding these fundamentals and following best practices, you can write efficient, maintainable GPU code for WebGPU applications.
