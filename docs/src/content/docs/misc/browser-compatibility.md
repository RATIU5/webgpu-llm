---
title: Browser Compatibility
sidebar:
  order: 10
---

## Overview

WebGPU achieved support across all major browsers in 2025, making desktop-class graphics available on the web. Chrome led with version 113 (April 2023), followed by Safari 26 (June 2025) and Firefox 141 (July 2025).

:::note[Multi-Backend Architecture]
WebGPU translates to native graphics APIs:
- **Windows**: Direct3D 12
- **macOS/iOS**: Metal
- **Linux/Android/ChromeOS**: Vulkan

This can affect performance characteristics and feature availability across platforms.
:::

## Browser Support Status

| Browser | Version | Platforms | Status | Backend |
|---------|---------|-----------|--------|---------|
| Chrome | 113+ | Windows, macOS, ChromeOS | Stable | Dawn (D3D12/Metal/Vulkan) |
| Chrome | 121+ | Android 12+ (Qualcomm/ARM) | Flag required | Dawn (Vulkan) |
| Edge | 113+ | Windows, macOS | Stable | Dawn (D3D12/Metal) |
| Firefox | 141+ | Windows | Stable | wgpu (D3D12) |
| Firefox | 145+ | macOS (Apple Silicon) | Stable | wgpu (Metal) |
| Safari | 26+ | macOS, iOS, iPadOS, visionOS | Stable | WebKit (Metal) |

### Platform-Specific Notes

<details>
<summary>**Chrome/Edge Details**</summary>

- **Windows**: Full support since v113 with D3D12 backend
- **macOS**: Full support on Intel and Apple Silicon
- **Android**: Requires Android 12+, Qualcomm/ARM GPUs, flag enabled via `chrome://flags/#enable-unsafe-webgpu`
- **Linux**: Requires flags: `--enable-unsafe-webgpu --enable-features=Vulkan`

</details>

<details>
<summary>**Firefox Details**</summary>

- **Windows**: Stable since Firefox 141 (July 2025)
- **macOS Apple Silicon**: Stable since Firefox 145
- **macOS Intel/Linux**: Available in Nightly, stable release planned 2026
- Built on the **wgpu** Rust library

</details>

<details>
<summary>**Safari Details**</summary>

- **All Apple platforms**: Safari 26 (June 2025)
- Enabled by default, no flags required
- Includes visionOS for Vision Pro support
- Tight Metal integration for optimal performance

</details>

## Feature Detection

### Basic Support Check

```typescript title="Check WebGPU availability" {1-4,6-11}
// Step 1: Check API exists (requires HTTPS)
if (!navigator.gpu) {
  throw new Error("WebGPU not supported");
}

// Step 2: Request adapter
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No GPU adapter found");
}

// Step 3: Request device
const device = await adapter.requestDevice();
```

:::caution[Secure Context Required]
WebGPU requires HTTPS. Pages served over HTTP won't have access to `navigator.gpu`, even in supporting browsers.
:::

### Optional Feature Detection

```typescript title="Check optional features"
// Query supported features
const hasTimestamps = adapter.features.has("timestamp-query");
const hasBC = adapter.features.has("texture-compression-bc");
const hasF16 = adapter.features.has("shader-f16");

// Request only available features
const availableFeatures = ["timestamp-query", "shader-f16"]
  .filter(f => adapter.features.has(f));

const device = await adapter.requestDevice({
  requiredFeatures: availableFeatures,
});
```

### Common Optional Features

| Feature | Description | Typical Platform |
|---------|-------------|------------------|
| `texture-compression-bc` | BC1-BC7 compression | Desktop |
| `texture-compression-etc2` | ETC2/EAC compression | Mobile |
| `texture-compression-astc` | ASTC compression | Mobile |
| `timestamp-query` | GPU timing queries | Most GPUs |
| `shader-f16` | 16-bit float in shaders | Modern GPUs |

### Query Limits

```typescript title="Check device limits"
const limits = adapter.limits;
console.log(`Max texture size: ${limits.maxTextureDimension2D}`);
console.log(`Max buffer size: ${limits.maxBufferSize}`);

if (limits.maxTextureDimension2D < 4096) {
  console.warn("Limited texture resolution support");
}
```

## Fallback Strategies

### WebGL Fallback

```typescript title="Progressive fallback" {2-8,11-14}
async function initGraphics() {
  // Try WebGPU first
  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const device = await adapter.requestDevice();
      return { type: "webgpu", device };
    }
  }

  // Fall back to WebGL2
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
  if (gl) {
    return { type: "webgl", gl };
  }

  throw new Error("No graphics API available");
}
```

### Library-Based Abstraction

```typescript title="Three.js automatic fallback"
import { WebGPURenderer } from "three/webgpu";
import { WebGLRenderer } from "three";

let renderer;
try {
  renderer = new WebGPURenderer();
  await renderer.init();
} catch {
  renderer = new WebGLRenderer();
}
```

:::tip[Use Preferred Canvas Format]
Always query the optimal format for the platform:

```typescript
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });
```
:::

## Platform Considerations

### Desktop Platforms

| Platform | Backend | Notes |
|----------|---------|-------|
| Windows 10/11 | D3D12 | Best tested, widest hardware support |
| macOS | Metal | Excellent performance, Intel and Apple Silicon |
| Linux | Vulkan | Requires flags, driver quality varies |

### Mobile Platforms

| Platform | Requirements | Notes |
|----------|--------------|-------|
| Android | Android 12+, Qualcomm/ARM GPU | Flag required in Chrome |
| iOS/iPadOS | iOS 26+ | Safari only, Metal backend |

:::danger[Mobile Constraints]
Mobile GPUs have stricter power and thermal limits. Test on multiple device tiers and implement quality presets.
:::

## Testing Recommendations

```typescript title="Comprehensive compatibility test"
async function testWebGPU() {
  if (!navigator.gpu) {
    return { supported: false, reason: "API unavailable" };
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return { supported: false, reason: "No adapter" };
  }

  const device = await adapter.requestDevice();
  return {
    supported: true,
    features: Array.from(adapter.features),
    limits: adapter.limits,
  };
}
```

:::tip[Testing Strategy]
- Test on Chrome, Firefox, and Safari (different implementations)
- Test on Windows, macOS, and Linux (different backends)
- Test on multiple GPU vendors (NVIDIA, AMD, Intel, Apple)
- Test on different device tiers for mobile
:::

## Resources

:::note[Official Documentation]
- [Can I Use WebGPU](https://caniuse.com/webgpu)
- [WebGPU Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
- [MDN WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
:::
