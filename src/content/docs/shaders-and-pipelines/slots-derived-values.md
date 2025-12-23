---
title: Slots and Derived Values in TypeGPU
sidebar:
  order: 50
---

TypeGPU provides powerful mechanisms for dynamic resource management through **slots** and **derived values**. These features enable runtime resource switching and computed values without requiring shader recompilation, making them essential tools for building flexible, performance-oriented WebGPU applications.

## Overview

In traditional WebGPU programming, resources like textures, samplers, and uniform buffers are typically bound statically to pipelines. Any change to these resources often requires recreating bind groups or even entire pipelines. TypeGPU's **slots** solve this problem by providing placeholder values that can be bound dynamically at runtime, while **derived values** enable reactive computation patterns where values are automatically calculated based on dependencies.

Together, these features provide a type-safe, declarative approach to dynamic resource binding and computed GPU values, allowing developers to build more flexible rendering systems, implement efficient material systems, and create multi-pass rendering pipelines with ease.

## What are Slots?

Slots in TypeGPU are **typed placeholders for GPU resources** that can be bound at runtime without requiring pipeline recreation. They act as indirection layers between your shader code and the actual GPU resources, enabling you to switch resources dynamically between draw calls or compute dispatches.

### Key Characteristics

- **Runtime Resource Switching**: Change textures, buffers, or values between draws without recompiling shaders or recreating pipelines
- **Type Safety**: TypeScript's type system ensures you bind resources of the correct type to each slot
- **Performance**: Avoid expensive pipeline recreation by using slots for resources that change frequently
- **Declarative API**: Define what can change upfront, then bind concrete values when needed

Think of slots as "variables" that exist on the CPU side but are accessible in your GPU shaders. While regular shader uniforms are baked into bind groups, slots provide an additional layer of flexibility that lets you swap entire resources at runtime.

## Creating Slots

The `tgpu.slot()` API creates a new slot with optional type information and default values:

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Create a slot for a scalar value
const timeSlot = tgpu.slot(d.f32);

// Create a slot for a vector
const colorSlot = tgpu.slot(d.vec3f);

// Create a slot with a default value
const scaleSlot = tgpu.slot(d.f32, 1.0);

// Create a slot for a texture
const textureSlot = tgpu.slot(d.texture2d(d.f32));

// Create a slot for a sampler
const samplerSlot = tgpu.slot("filtering");
```

### Slot API Signature

```typescript
tgpu.slot: <T>(schema: DataSchema<T>, defaultValue?: T) => TgpuSlot<T>
```

The slot function accepts:

- **schema**: A data schema defining the type of value the slot will hold (from `typegpu/data`)
- **defaultValue** (optional): An initial value for the slot

### Slot Types and Schemas

Slots can hold various types of GPU resources:

**Primitive Types:**

```typescript
const floatSlot = tgpu.slot(d.f32);
const intSlot = tgpu.slot(d.i32);
const uintSlot = tgpu.slot(d.u32);
```

**Vector Types:**

```typescript
const vec2Slot = tgpu.slot(d.vec2f);
const vec3Slot = tgpu.slot(d.vec3f);
const vec4Slot = tgpu.slot(d.vec4f);
```

**Matrix Types:**

```typescript
const mat3Slot = tgpu.slot(d.mat3x3f);
const mat4Slot = tgpu.slot(d.mat4x4f);
```

**Textures:**

```typescript
const texture2DSlot = tgpu.slot(d.texture2d(d.f32));
const textureCubeSlot = tgpu.slot(d.textureCube(d.f32));
const storageTextureSlot = tgpu.slot(d.storageTexture2d("rgba8unorm"));
```

**Structured Data:**

```typescript
const materialSlot = tgpu.slot(
  d.struct({
    albedo: d.vec3f,
    metallic: d.f32,
    roughness: d.f32,
  }),
);
```

### Default Values

Providing default values is useful for initialization and fallback scenarios:

```typescript
const opacitySlot = tgpu.slot(d.f32, 1.0);
const tintColorSlot = tgpu.slot(d.vec3f, vec3(1.0, 1.0, 1.0));
```

Default values ensure your slots have valid data even before explicit binding, preventing undefined behavior during initial renders.

## Using Slots in Shaders

Slots can be used within TGSL (TypeGPU Shading Language) functions, but they require special handling because they have different representations on the CPU versus the GPU.

### Accessing Slot Values: The `.value` Property

Objects that have different types on the CPU and on the GPU (like buffers, layouts, and slots) need to be accessed via the **`.value` property** in TGSL functions. TypeGPU also provides a `$` property alias as a shorthand.

```typescript
const timeSlot = tgpu.slot(d.f32);

const animate = tgpu.fn(
  [],
  d.vec3f,
)((pos) => {
  "use gpu";

  // Access the slot value using .value property
  const time = timeSlot.value;

  // Or use the $ alias
  const time2 = timeSlot.$;

  return d.vec3f(pos.x + sin(time), pos.y + cos(time), pos.z);
});
```

### External Resources in Shaders

Slots are passed to shader functions as **external resources** using the `$uses` method:

```typescript
const textureSlot = tgpu.slot(d.texture2d(d.f32));
const samplerSlot = tgpu.slot("filtering");

const sampleTexture = tgpu
  .fn([d.vec2f], d.vec4f)
  .$uses({ tex: textureSlot, smp: samplerSlot })((uv) => {
  "use gpu";
  return textureSample(textureSlot.value, samplerSlot.value, uv);
});
```

The `$uses` method declares external dependencies, allowing TypeGPU to properly resolve and bind these resources during shader compilation.

### Complete Shader Example

```typescript
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Define slots
const timeSlot = tgpu.slot(d.f32, 0.0);
const colorSlot = tgpu.slot(d.vec3f, vec3(1, 1, 1));
const textureSlot = tgpu.slot(d.texture2d(d.f32));
const samplerSlot = tgpu.slot("filtering");

// Fragment shader using slots
const fragmentShader = tgpu.fn([d.vec2f], d.vec4f).$uses({
  time: timeSlot,
  tint: colorSlot,
  tex: textureSlot,
  smp: samplerSlot,
})((uv) => {
  "use gpu";

  // Sample texture
  const texColor = textureSample(textureSlot.value, samplerSlot.value, uv);

  // Apply animated tint
  const tintedColor = texColor.rgb * colorSlot.value;
  const pulseAmount = (sin(timeSlot.value) + 1.0) * 0.5;

  return d.vec4f(tintedColor * pulseAmount, texColor.a);
});
```

## Runtime Resource Switching

The power of slots becomes apparent when you bind different resources at runtime without recreating pipelines.

### Binding Resources to Slots

Before executing pipelines, you must bind all utilized resources including slots. This is done using the **`.with()` method**:

```typescript
// Create a pipeline
const renderPipeline = root["~unstable"]
  .withVertex(vertexFn, vertexLayout)
  .withFragment(fragmentFn, { color: format })
  .createPipeline();

// Bind slots with specific values
renderPipeline
  .with(bindGroup)
  .with(timeSlot, currentTime)
  .with(colorSlot, vec3(1.0, 0.5, 0.2))
  .with(textureSlot, myTexture)
  .with(samplerSlot, mySampler)
  .withColorAttachment({ color: colorTarget })
  .draw(vertexCount);
```

### Changing Resources Between Draws

Slots truly shine when you need to render multiple objects with different resources:

```typescript
// First draw call with red tint
renderPipeline
  .with(bindGroup)
  .with(colorSlot, vec3(1.0, 0.0, 0.0))
  .with(textureSlot, texture1)
  .withColorAttachment({ color: colorTarget })
  .draw(vertexCount);

// Second draw call with blue tint and different texture
renderPipeline
  .with(bindGroup)
  .with(colorSlot, vec3(0.0, 0.0, 1.0))
  .with(textureSlot, texture2)
  .withColorAttachment({ color: colorTarget })
  .draw(vertexCount);

// Third draw call with another configuration
renderPipeline
  .with(bindGroup)
  .with(colorSlot, vec3(0.0, 1.0, 0.0))
  .with(textureSlot, texture3)
  .withColorAttachment({ color: colorTarget })
  .draw(vertexCount);
```

### Performance Implications

**When Slots Excel:**

- Resources change frequently between draw calls
- Material parameters vary per object
- Multi-pass rendering with different textures per pass
- LOD systems switching detail levels

**Overhead Considerations:**

- Slot binding has some overhead compared to static bindings
- Each `.with()` call updates internal state
- For resources that never change, prefer static bind group entries

**Optimization Strategy:**

```typescript
// Good: Slots for dynamic resources
const textureSlot = tgpu.slot(d.texture2d(d.f32));

// Good: Static bind group for unchanging resources
const staticBindGroup = root.createBindGroup(layout, {
  camera: cameraBuffer,
  lights: lightBuffer,
});

// Use both together
renderPipeline
  .with(staticBindGroup) // Bound once
  .with(textureSlot, tex1) // Changes per object
  .draw(vertexCount);
```

## Derived Values

### What is `tgpu.derived()`?

The `tgpu.~unstable.derived()` API enables **reactive computed values** that are automatically calculated based on dependencies. While slots hold concrete values that you bind explicitly, derived values compute their results from other slots or values.

**Important Note**: The derived API is currently under the `~unstable` namespace and is generally discouraged except for advanced use cases requiring full WGSL compatibility.

### API Signature

```typescript
tgpu.~unstable.derived: <T>(compute: () => T) => TgpuDerived<T>
```

The derived function accepts:

- **compute**: A function `() => T` that computes and returns the derived value

### Creating Derived Values

```typescript
const widthSlot = tgpu.slot(d.f32, 1920);
const heightSlot = tgpu.slot(d.f32, 1080);

// Derived aspect ratio
const aspectRatio = tgpu["~unstable"].derived(() => {
  return widthSlot.value / heightSlot.value;
});

// Derived viewport size
const viewportSize = tgpu["~unstable"].derived(() => {
  return d.vec2f(widthSlot.value, heightSlot.value);
});

// Derived inverse resolution
const invResolution = tgpu["~unstable"].derived(() => {
  return d.vec2f(1.0 / widthSlot.value, 1.0 / heightSlot.value);
});
```

### Reactive Programming Patterns

Derived values follow reactive programming principles:

```typescript
// Base slots
const timeSlot = tgpu.slot(d.f32);
const speedSlot = tgpu.slot(d.f32, 1.0);

// Derived animated value
const animatedOffset = tgpu["~unstable"].derived(() => {
  return sin(timeSlot.value * speedSlot.value);
});

// Derived color that changes with time
const animatedColor = tgpu["~unstable"].derived(() => {
  const t = timeSlot.value * speedSlot.value;
  return d.vec3f(
    (sin(t) + 1.0) * 0.5,
    (sin(t + 2.094) + 1.0) * 0.5,
    (sin(t + 4.189) + 1.0) * 0.5,
  );
});
```

### Dependency Tracking

TypeGPU automatically tracks which slots a derived value depends on:

```typescript
const slot1 = tgpu.slot(d.f32);
const slot2 = tgpu.slot(d.f32);
const slot3 = tgpu.slot(d.f32);

// This derived value depends on slot1 and slot2
const derived1 = tgpu["~unstable"].derived(() => {
  return slot1.value + slot2.value;
});

// This derived value depends on derived1 and slot3
const derived2 = tgpu["~unstable"].derived(() => {
  return derived1.value * slot3.value;
});
```

**Automatic Invalidation**: When a slot's value changes, all derived values that depend on it are marked for recomputation.

**Recomputation Triggers**: Derived values are recalculated when:

- A dependent slot is bound to a new value
- A dependent derived value is recalculated
- The compute function is explicitly called

**Caching Behavior**: TypeGPU caches derived value results to avoid redundant computation. The cache is invalidated when dependencies change.

## Use Cases

### Dynamic Texture Swapping

Perfect for texture atlases, sprite animation, or dynamic material systems:

```typescript
const textureSlot = tgpu.slot(d.texture2d(d.f32));
const textures = [texture1, texture2, texture3];
let currentFrame = 0;

function animateSprite() {
  renderPipeline
    .with(textureSlot, textures[currentFrame % textures.length])
    .withColorAttachment({ color: output })
    .draw(6);

  currentFrame++;
}

requestAnimationFrame(animateSprite);
```

### Material Systems

Implement configurable shader parameters for different materials:

```typescript
// Define material parameter slots
const albedoSlot = tgpu.slot(d.vec3f);
const metallicSlot = tgpu.slot(d.f32);
const roughnessSlot = tgpu.slot(d.f32);
const normalMapSlot = tgpu.slot(d.texture2d(d.f32));

// Material definitions
const materials = {
  gold: {
    albedo: vec3(1.0, 0.766, 0.336),
    metallic: 1.0,
    roughness: 0.2,
    normalMap: goldNormalMap,
  },
  wood: {
    albedo: vec3(0.4, 0.25, 0.15),
    metallic: 0.0,
    roughness: 0.8,
    normalMap: woodNormalMap,
  },
  plastic: {
    albedo: vec3(0.9, 0.1, 0.1),
    metallic: 0.0,
    roughness: 0.5,
    normalMap: defaultNormalMap,
  },
};

// Render with different materials
function renderObject(material) {
  renderPipeline
    .with(albedoSlot, material.albedo)
    .with(metallicSlot, material.metallic)
    .with(roughnessSlot, material.roughness)
    .with(normalMapSlot, material.normalMap)
    .withColorAttachment({ color: output })
    .draw(vertexCount);
}

renderObject(materials.gold);
renderObject(materials.wood);
renderObject(materials.plastic);
```

### Multi-pass Rendering

Use different resources for each rendering pass:

```typescript
const inputTextureSlot = tgpu.slot(d.texture2d(d.f32));
const outputTextureSlot = tgpu.slot(d.storageTexture2d("rgba8unorm"));

// Blur pass 1: horizontal
blurPipeline
  .with(inputTextureSlot, sceneTexture)
  .with(outputTextureSlot, tempTexture)
  .with(directionSlot, vec2(1, 0))
  .dispatchWorkgroups(workgroupsX, workgroupsY);

// Blur pass 2: vertical
blurPipeline
  .with(inputTextureSlot, tempTexture)
  .with(outputTextureSlot, finalTexture)
  .with(directionSlot, vec2(0, 1))
  .dispatchWorkgroups(workgroupsX, workgroupsY);
```

### LOD Systems

Switch between detail levels based on distance or performance:

```typescript
const modelTextureSlot = tgpu.slot(d.texture2d(d.f32));
const lodTextures = [highResTexture, mediumResTexture, lowResTexture];

function selectLOD(distance) {
  if (distance < 10) return 0;
  if (distance < 50) return 1;
  return 2;
}

function renderWithLOD(object) {
  const lod = selectLOD(object.distanceToCamera);

  renderPipeline
    .with(modelTextureSlot, lodTextures[lod])
    .withColorAttachment({ color: output })
    .draw(object.vertexCount);
}
```

## Performance Implications

### Slot Binding Overhead

Each slot binding operation has a small performance cost:

```typescript
// Less efficient: Many individual bindings
for (let i = 0; i < 1000; i++) {
  pipeline
    .with(slot1, values1[i])
    .with(slot2, values2[i])
    .with(slot3, values3[i])
    .draw(6);
}

// More efficient: Batch similar operations, minimize state changes
for (let material of uniqueMaterials) {
  pipeline.with(materialSlot, material);

  for (let object of objectsWithMaterial[material]) {
    pipeline.with(transformSlot, object.transform).draw(object.vertexCount);
  }
}
```

### When to Use Static vs Dynamic Bindings

**Use Static Bindings (Bind Groups) When:**

- Resources don't change during rendering
- Resources are shared across many draw calls
- You need maximum performance
- Resources are set once at initialization

**Use Slots When:**

- Resources change between draw calls
- You need runtime flexibility
- Different objects require different resources
- Implementing material systems or LOD

### Optimization Strategies

**1. Group Static Resources:**

```typescript
// Static bind group for unchanging resources
const staticBindGroup = root.createBindGroup(layout, {
  camera: cameraBuffer,
  lights: lightBuffer,
  environment: environmentMap,
});

// Slots only for dynamic resources
const albedoSlot = tgpu.slot(d.vec3f);
const textureSlot = tgpu.slot(d.texture2d(d.f32));
```

**2. Minimize Slot Changes:**

```typescript
// Sort objects by material to reduce slot changes
objects.sort((a, b) => a.materialId - b.materialId);

let currentMaterial = null;
for (let obj of objects) {
  if (obj.material !== currentMaterial) {
    pipeline.with(materialSlot, obj.material);
    currentMaterial = obj.material;
  }

  pipeline.with(transformSlot, obj.transform).draw(obj.vertexCount);
}
```

**3. Use Derived Values for Computed Properties:**

```typescript
// Instead of computing on CPU every frame
const invResolution = tgpu["~unstable"].derived(() => {
  return d.vec2f(1.0 / widthSlot.value, 1.0 / heightSlot.value);
});
```

## TypeGPU Integration

### Using Slots in Bind Groups

Slots can be included in bind group layouts:

```typescript
const layout = tgpu.bindGroupLayout({
  viewProjection: { uniform: d.mat4x4f },
  material: { uniform: materialSchema },
  albedoTexture: { texture: d.texture2d(d.f32) },
  sampler: { sampler: "filtering" },
});

// Create slot for dynamic texture
const textureSlot = tgpu.slot(d.texture2d(d.f32));

const bindGroup = root.createBindGroup(layout, {
  viewProjection: cameraBuffer,
  material: materialBuffer,
  albedoTexture: textureSlot, // Slot in bind group
  sampler: defaultSampler,
});
```

### Combining with Pipelines

Complete example integrating slots with pipelines:

```typescript
const root = await tgpu.init();

// Define slots
const timeSlot = tgpu.slot(d.f32, 0);
const colorSlot = tgpu.slot(d.vec3f, vec3(1, 1, 1));

// Create shader using slots
const fragmentFn = tgpu
  .fn([d.vec2f], d.vec4f)
  .$uses({ time: timeSlot, color: colorSlot })((uv) => {
  "use gpu";
  const col = colorSlot.value;
  const t = timeSlot.value;
  return d.vec4f(col * sin(t), 1.0);
});

// Create pipeline
const pipeline = root["~unstable"]
  .withVertex(vertexFn, vertexLayout)
  .withFragment(fragmentFn, { color: "rgba8unorm" })
  .createPipeline();

// Render loop
function render(time) {
  pipeline
    .with(vertexLayout, vertexBuffer)
    .with(timeSlot, time / 1000)
    .with(colorSlot, vec3(1, 0.5, 0.2))
    .withColorAttachment({ color: outputTexture })
    .draw(6);

  requestAnimationFrame(render);
}
```

## Best Practices

### When to Use Slots vs Static Bindings

**Prefer Slots For:**

- Per-object material properties
- Animated parameters that change every frame
- Textures that swap based on game state
- LOD selection
- Multi-pass rendering with different inputs

**Prefer Static Bindings For:**

- Scene-wide uniforms (camera, lighting)
- Resources loaded once at startup
- Shared textures (environment maps, lookup tables)
- Rarely-changing configuration

### Organizing Slot Hierarchies

Structure slots logically by usage:

```typescript
// Per-frame slots (change every frame)
const perFrame = {
  time: tgpu.slot(d.f32),
  deltaTime: tgpu.slot(d.f32),
  frameCount: tgpu.slot(d.u32),
};

// Per-view slots (change per camera)
const perView = {
  viewMatrix: tgpu.slot(d.mat4x4f),
  projectionMatrix: tgpu.slot(d.mat4x4f),
};

// Per-object slots (change per object)
const perObject = {
  modelMatrix: tgpu.slot(d.mat4x4f),
  albedoTexture: tgpu.slot(d.texture2d(d.f32)),
  normalTexture: tgpu.slot(d.texture2d(d.f32)),
};

// Per-material slots
const perMaterial = {
  baseColor: tgpu.slot(d.vec3f),
  metallic: tgpu.slot(d.f32),
  roughness: tgpu.slot(d.f32),
};
```

### Naming Conventions

Use clear, descriptive names for slots:

```typescript
// Good: Clear intent
const albedoTextureSlot = tgpu.slot(d.texture2d(d.f32));
const metallicFactorSlot = tgpu.slot(d.f32);
const modelMatrixSlot = tgpu.slot(d.mat4x4f);

// Avoid: Vague names
const slot1 = tgpu.slot(d.texture2d(d.f32));
const param = tgpu.slot(d.f32);
const data = tgpu.slot(d.mat4x4f);
```

Suffix slot variables with "Slot" to distinguish them from regular values.

## Common Pitfalls

### Overusing Dynamic Bindings

**Problem**: Using slots for everything sacrifices performance:

```typescript
// Bad: Everything is a slot
const cameraSlot = tgpu.slot(d.mat4x4f);
const lightSlot = tgpu.slot(lightSchema);
const envMapSlot = tgpu.slot(d.textureCube(d.f32));
```

**Solution**: Use static bind groups for stable resources:

```typescript
// Good: Static bind group for unchanging data
const sceneBindGroup = root.createBindGroup(layout, {
  camera: cameraBuffer,
  lights: lightBuffer,
  envMap: environmentMap,
});

// Slots only for dynamic data
const modelMatrixSlot = tgpu.slot(d.mat4x4f);
```

### Type Mismatches

**Problem**: Binding values that don't match the slot schema:

```typescript
const vec3Slot = tgpu.slot(d.vec3f);

// Runtime error: wrong type
pipeline.with(vec3Slot, vec4(1, 2, 3, 4));
```

**Solution**: TypeScript will catch these at compile time with proper typing:

```typescript
const vec3Slot = tgpu.slot(d.vec3f);

// Type error caught by TypeScript
pipeline.with(vec3Slot, vec3(1, 2, 3)); // ✓ Correct
```

### Forgetting to Bind Slots

**Problem**: Executing pipeline without binding required slots:

```typescript
const colorSlot = tgpu.slot(d.vec3f);

// Error: colorSlot not bound
pipeline.withColorAttachment({ color: output }).draw(6);
```

**Solution**: Always bind all slots before execution:

```typescript
// Correct: All slots bound
pipeline
  .with(colorSlot, vec3(1, 0, 0))
  .withColorAttachment({ color: output })
  .draw(6);
```

### Accessing Slots Without `.value` in TGSL

**Problem**: Directly accessing slot in shader code:

```typescript
const timeSlot = tgpu.slot(d.f32);

const shader = tgpu.fn(
  [],
  d.f32,
)(() => {
  "use gpu";
  return timeSlot; // ✗ Wrong: missing .value
});
```

**Solution**: Always use `.value` (or `$` alias) in TGSL:

```typescript
const shader = tgpu.fn(
  [],
  d.f32,
)(() => {
  "use gpu";
  return timeSlot.value; // ✓ Correct
  // or: return timeSlot.$;
});
```

---

## Conclusion

TypeGPU's slots and derived values provide powerful abstractions for managing dynamic GPU resources in a type-safe, performant manner. By understanding when to use slots versus static bindings, how to leverage derived values for reactive computation, and following best practices for organization and naming, you can build flexible, maintainable WebGPU applications that make the most of modern GPU capabilities.

Remember:

- **Slots** enable runtime resource switching without pipeline recreation
- **Derived values** provide reactive computation patterns
- Use the **`.value` property** to access slots in TGSL shaders
- Bind slots with the **`.with()` method** before pipeline execution
- Balance flexibility with performance by using static bindings where appropriate
- Organize slots hierarchically by update frequency

For more information, consult the [official TypeGPU documentation](https://docs.swmansion.com/TypeGPU/), explore the [API reference](https://docs.swmansion.com/TypeGPU/api/typegpu/variables/tgpu/), or join the community on [Discord](https://discord.gg/8jpfgDqPcM).

---

## Sources

- [TypeGPU Official Documentation](https://docs.swmansion.com/TypeGPU/)
- [tgpu API Reference](https://docs.swmansion.com/TypeGPU/api/typegpu/variables/tgpu/)
- [TypeGPU GitHub Repository](https://github.com/software-mansion/TypeGPU)
- [TGSL Fundamentals](https://docs.swmansion.com/TypeGPU/fundamentals/tgsl/)
- [Bind Groups Documentation](https://docs.swmansion.com/TypeGPU/fundamentals/bind-groups/)
- [Pipelines Documentation](https://docs.swmansion.com/TypeGPU/fundamentals/pipelines/)
- [Functions Documentation](https://docs.swmansion.com/TypeGPU/fundamentals/functions/)
