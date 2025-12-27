---
title: TypeGPU Utilities
sidebar:
  order: 26
---

## Overview

TypeGPU provides companion packages for common GPU programming tasks:
- **@typegpu/noise**: Pseudo-random and Perlin noise generation
- **@typegpu/color**: Color space conversions and manipulation

These packages work with both TypeGPU and vanilla WebGPU projects.

## @typegpu/noise

Pseudo-random utilities for procedural generation, visual effects, and simulations.

### Installation

```bash
npm install @typegpu/noise
```

### Random Number Generation

```typescript title="Uniform random [0, 1)"
import { random } from "@typegpu/noise";
import * as d from "typegpu/data";

const computeShader = tgpu.fn([d.u32], d.f32).does`(seed: u32) -> f32 {
  // Returns value in [0, 1)
  return ${random.uniform}(seed);
}`;
```

### Perlin Noise

Smooth, natural-looking variations for terrain, textures, and effects.

```typescript title="2D Perlin noise" {5}
import { perlin2d } from "@typegpu/noise";
import * as d from "typegpu/data";

const generateTerrain = tgpu.fn([d.vec2f], d.f32).does`(pos: vec2f) -> f32 {
  // Returns value in [-1, 1]
  return ${perlin2d}.sample(pos);
}`;
```

```typescript title="3D Perlin noise"
import { perlin3d } from "@typegpu/noise";

const volumetricNoise = tgpu.fn([d.vec3f], d.f32).does`(pos: vec3f) -> f32 {
  return ${perlin3d}.sample(pos);
}`;
```

### Fractal Brownian Motion (FBM)

Layer multiple noise octaves for complex patterns:

```typescript title="FBM pattern"
import { perlin2d } from "@typegpu/noise";

const fbm = tgpu.fn([d.vec2f, d.i32], d.f32).does`(pos: vec2f, octaves: i32) -> f32 {
  var value = 0.0;
  var amplitude = 0.5;
  var frequency = 1.0;
  var p = pos;

  for (var i = 0; i < octaves; i++) {
    value += amplitude * ${perlin2d}.sample(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
  }

  return value;
}`;
```

### Use Cases

| Function | Output Range | Use Case |
|----------|--------------|----------|
| `random.uniform` | [0, 1) | Particle spawning, randomization |
| `perlin2d.sample` | [-1, 1] | Terrain, textures, 2D effects |
| `perlin3d.sample` | [-1, 1] | Volumetric effects, 3D textures |

## @typegpu/color

Color space conversions and manipulation for GPU shaders.

### Installation

```bash
npm install @typegpu/color
```

### Color Spaces

```typescript title="RGB to OKLab conversion"
import { rgbToOklab, oklabToRgb } from "@typegpu/color";
import * as d from "typegpu/data";

const adjustLightness = tgpu.fn([d.vec3f, d.f32], d.vec3f)
  .does`(rgb: vec3f, factor: f32) -> vec3f {
    // Convert to perceptually uniform space
    var lab = ${rgbToOklab}(rgb);

    // Adjust lightness (L channel)
    lab.x *= factor;

    // Convert back to RGB
    return ${oklabToRgb}(lab);
  }`;
```

### Color Blending

```typescript title="Perceptually smooth blending"
import { rgbToOklab, oklabToRgb } from "@typegpu/color";

const blendColors = tgpu.fn([d.vec3f, d.vec3f, d.f32], d.vec3f)
  .does`(a: vec3f, b: vec3f, t: f32) -> vec3f {
    let labA = ${rgbToOklab}(a);
    let labB = ${rgbToOklab}(b);

    // Blend in OKLab for perceptually uniform interpolation
    let blended = mix(labA, labB, t);

    return ${oklabToRgb}(blended);
  }`;
```

:::tip[Why OKLab?]
OKLab is a perceptually uniform color space. Blending in OKLab produces more natural transitions than RGB blending, avoiding muddy mid-tones.
:::

## Zero-Initialized Values

TypeGPU v0.7+ allows creating zero-initialized values from any schema:

```typescript title="Zero initialization"
import * as d from "typegpu/data";

// Define a struct schema
const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  lifetime: d.f32,
});

// Create zero-initialized value
const emptyParticle = Particle();
// { position: [0,0,0], velocity: [0,0,0], lifetime: 0 }

// Works with any type
const zeroVec = d.vec3f();    // [0, 0, 0]
const zeroMat = d.mat4x4f();  // Identity-like zero matrix
const zeroArray = d.arrayOf(d.f32, 10)();  // [0,0,0,0,0,0,0,0,0,0]
```

### Buffer Initialization

```typescript title="Initialize buffer with zeros"
const particleBuffer = root
  .createBuffer(d.arrayOf(Particle, 1000))
  .$usage("storage");

// Write zeros to clear buffer
const zeros = d.arrayOf(Particle, 1000)();
particleBuffer.write(zeros);
```

## Vanilla WebGPU Compatibility

All utility packages work with vanilla WebGPU (no TypeGPU required):

```typescript title="Using @typegpu/noise with vanilla WebGPU"
import { perlin2d } from "@typegpu/noise";
import tgpu from "typegpu";

// Get WGSL code for the function
const wgslCode = tgpu.resolve({
  noise: perlin2d,
});

// Use in your existing WGSL shader
const shaderCode = `
${wgslCode}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = vec2f(id.xy) / 512.0;
  let n = noise_sample(uv * 10.0);  // Generated function name
  // ... use noise value
}
`;
```

## Package Summary

| Package | Features | Version |
|---------|----------|---------|
| `@typegpu/noise` | Random, Perlin 2D/3D | 0.7.0+ |
| `@typegpu/color` | OKLab conversions | 0.7.0+ |

:::note[Ecosystem Packages]
Additional packages like `@typegpu/sdf` (signed distance fields) are also available. Check the [TypeGPU ecosystem](https://docs.swmansion.com/TypeGPU/ecosystem/) for the full list.
:::
