---
title: Slots and Derived Values
sidebar:
  order: 50
---

## Overview

TypeGPU provides **slots** for dynamic resource management and **derived values** for reactive computation patterns. These features enable runtime resource switching without shader recompilation.

:::note[Key Concepts]
- **Slots**: Typed placeholders for GPU resources bound at runtime
- **Derived Values**: Computed values that update automatically based on dependencies
:::

## Creating Slots

```typescript title="Slot creation"
import tgpu from "typegpu";
import * as d from "typegpu/data";

// Scalar slots
const timeSlot = tgpu.slot(d.f32);
const scaleSlot = tgpu.slot(d.f32, 1.0);  // With default

// Vector/matrix slots
const colorSlot = tgpu.slot(d.vec3f);
const transformSlot = tgpu.slot(d.mat4x4f);

// Texture/sampler slots
const textureSlot = tgpu.slot(d.texture2d(d.f32));
const samplerSlot = tgpu.slot("filtering");

// Structured data
const materialSlot = tgpu.slot(d.struct({
  albedo: d.vec3f,
  metallic: d.f32,
  roughness: d.f32,
}));
```

## Using Slots in Shaders

Access slots with `.value` (or `$` alias) and declare dependencies with `$uses`:

```typescript title="Slots in TGSL" {7-8,12-13}
const timeSlot = tgpu.slot(d.f32, 0.0);
const colorSlot = tgpu.slot(d.vec3f);
const textureSlot = tgpu.slot(d.texture2d(d.f32));
const samplerSlot = tgpu.slot("filtering");

const fragmentShader = tgpu.fn([d.vec2f], d.vec4f)
  .$uses({ time: timeSlot, tint: colorSlot, tex: textureSlot, smp: samplerSlot })
  ((uv) => {
    "use gpu";
    const texColor = textureSample(textureSlot.value, samplerSlot.value, uv);
    const pulse = (Math.sin(timeSlot.value) + 1.0) * 0.5;
    return d.vec4f(texColor.rgb * colorSlot.value * pulse, texColor.a);
  });
```

## Runtime Resource Binding

Bind slots with `.with()` before pipeline execution:

```typescript title="Binding slots" {4-7}
const renderPipeline = root["~unstable"]
  .withVertex(vertexFn, vertexLayout)
  .withFragment(fragmentFn, { color: format })
  .createPipeline();

renderPipeline
  .with(bindGroup)
  .with(timeSlot, currentTime)
  .with(colorSlot, vec3(1.0, 0.5, 0.2))
  .with(textureSlot, myTexture)
  .with(samplerSlot, mySampler)
  .withColorAttachment({ color: colorTarget })
  .draw(vertexCount);
```

### Switching Resources Between Draws

```typescript title="Dynamic material switching"
// Red material
renderPipeline
  .with(colorSlot, vec3(1.0, 0.0, 0.0))
  .with(textureSlot, texture1)
  .withColorAttachment({ color: output })
  .draw(vertexCount);

// Blue material - no pipeline recreation
renderPipeline
  .with(colorSlot, vec3(0.0, 0.0, 1.0))
  .with(textureSlot, texture2)
  .withColorAttachment({ color: output })
  .draw(vertexCount);
```

## Derived Values

:::caution[Unstable API]
Derived values are under `~unstable` namespace and may change in future versions.
:::

Derived values compute automatically from dependencies:

```typescript title="Reactive computation"
const widthSlot = tgpu.slot(d.f32, 1920);
const heightSlot = tgpu.slot(d.f32, 1080);

// Automatically recalculates when width/height change
const aspectRatio = tgpu["~unstable"].derived(() => {
  return widthSlot.value / heightSlot.value;
});

const invResolution = tgpu["~unstable"].derived(() => {
  return d.vec2f(1.0 / widthSlot.value, 1.0 / heightSlot.value);
});
```

### Dependency Chains

```typescript title="Chained derived values"
const timeSlot = tgpu.slot(d.f32);
const speedSlot = tgpu.slot(d.f32, 1.0);

// Depends on timeSlot and speedSlot
const animatedOffset = tgpu["~unstable"].derived(() => {
  return Math.sin(timeSlot.value * speedSlot.value);
});

// Depends on animatedOffset
const animatedColor = tgpu["~unstable"].derived(() => {
  const t = animatedOffset.value;
  return d.vec3f(t * 0.5 + 0.5, 0.5, 1.0 - t * 0.5);
});
```

## Use Cases

<details>
<summary>**Material Systems**</summary>

```typescript
const albedoSlot = tgpu.slot(d.vec3f);
const metallicSlot = tgpu.slot(d.f32);
const roughnessSlot = tgpu.slot(d.f32);
const normalMapSlot = tgpu.slot(d.texture2d(d.f32));

const materials = {
  gold: { albedo: vec3(1.0, 0.766, 0.336), metallic: 1.0, roughness: 0.2 },
  wood: { albedo: vec3(0.4, 0.25, 0.15), metallic: 0.0, roughness: 0.8 },
};

function renderWithMaterial(mat) {
  renderPipeline
    .with(albedoSlot, mat.albedo)
    .with(metallicSlot, mat.metallic)
    .with(roughnessSlot, mat.roughness)
    .draw(vertexCount);
}
```

</details>

<details>
<summary>**Multi-pass Rendering**</summary>

```typescript
const inputTextureSlot = tgpu.slot(d.texture2d(d.f32));
const directionSlot = tgpu.slot(d.vec2f);

// Horizontal blur pass
blurPipeline
  .with(inputTextureSlot, sceneTexture)
  .with(directionSlot, vec2(1, 0))
  .dispatchWorkgroups(workgroupsX, workgroupsY);

// Vertical blur pass
blurPipeline
  .with(inputTextureSlot, tempTexture)
  .with(directionSlot, vec2(0, 1))
  .dispatchWorkgroups(workgroupsX, workgroupsY);
```

</details>

<details>
<summary>**LOD Systems**</summary>

```typescript
const textureSlot = tgpu.slot(d.texture2d(d.f32));
const lodTextures = [highRes, mediumRes, lowRes];

function selectLOD(distance) {
  if (distance < 10) return 0;
  if (distance < 50) return 1;
  return 2;
}

function renderWithLOD(object) {
  renderPipeline
    .with(textureSlot, lodTextures[selectLOD(object.distance)])
    .draw(object.vertexCount);
}
```

</details>

## Performance Guidelines

:::tip[When to Use Slots]
- Resources change between draw calls
- Material parameters vary per object
- Multi-pass rendering with different textures
- LOD or quality level switching
:::

:::tip[When to Use Static Bind Groups]
- Resources don't change during rendering
- Scene-wide uniforms (camera, lighting)
- Resources loaded once at startup
:::

### Optimization Strategy

```typescript title="Combine static and dynamic bindings"
// Static for unchanging resources
const staticBindGroup = root.createBindGroup(layout, {
  camera: cameraBuffer,
  lights: lightBuffer,
});

// Slots only for dynamic resources
const textureSlot = tgpu.slot(d.texture2d(d.f32));

// Minimize slot changes by sorting objects
objects.sort((a, b) => a.materialId - b.materialId);

let currentMaterial = null;
for (const obj of objects) {
  if (obj.material !== currentMaterial) {
    pipeline.with(materialSlot, obj.material);
    currentMaterial = obj.material;
  }
  pipeline.with(transformSlot, obj.transform).draw(obj.vertexCount);
}
```

## Common Pitfalls

:::danger[Accessing Slots Without .value]
```typescript
// WRONG
const shader = tgpu.fn([], d.f32)(() => {
  "use gpu";
  return timeSlot;  // Missing .value
});

// CORRECT
const shader = tgpu.fn([], d.f32)(() => {
  "use gpu";
  return timeSlot.value;
});
```
:::

:::danger[Forgetting to Bind Slots]
```typescript
// WRONG - colorSlot not bound
pipeline.withColorAttachment({ color: output }).draw(6);

// CORRECT
pipeline
  .with(colorSlot, vec3(1, 0, 0))
  .withColorAttachment({ color: output })
  .draw(6);
```
:::

:::caution[Type Mismatches]
TypeScript catches binding wrong types at compile time:

```typescript
const vec3Slot = tgpu.slot(d.vec3f);

pipeline.with(vec3Slot, vec4(1, 2, 3, 4));  // Type error
pipeline.with(vec3Slot, vec3(1, 2, 3));     // Correct
```
:::

## Resources

:::note[Official Documentation]
- [TypeGPU Documentation](https://docs.swmansion.com/TypeGPU/)
- [tgpu API Reference](https://docs.swmansion.com/TypeGPU/api/typegpu/variables/tgpu/)
- [Bind Groups Guide](https://docs.swmansion.com/TypeGPU/fundamentals/bind-groups/)
:::
