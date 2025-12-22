# Browser Compatibility

## Overview

WebGPU represents a significant advancement in web graphics technology, but as a modern API specification, it is still in the process of achieving universal browser support. Unlike established web APIs that enjoy near-universal availability, WebGPU is currently classified as **not Baseline** by web standards organizations, meaning it does not yet work in all widely-used browsers. This limited availability stems from the API's recent standardization and the significant engineering effort required to implement it across different browser engines and operating systems.

The current state of WebGPU browser support reflects a transitional period in web platform capabilities. Major browser vendors—Google (Chrome), Mozilla (Firefox), and Apple (Safari)—have all committed to WebGPU implementation, but they have reached different stages of rollout as of 2025. Chromium-based browsers led the charge with stable support beginning in early 2023, while Firefox and Safari achieved full support in 2025. This staggered adoption timeline has important implications for web developers: while WebGPU unlocks powerful graphics and compute capabilities, production applications must carefully consider their target audience's browser landscape and implement appropriate fallback strategies.

Understanding browser compatibility goes beyond simply checking version numbers. WebGPU's behavior varies across platforms due to its multi-backend architecture. On Windows, implementations use **Direct3D 12** as the underlying graphics API; macOS implementations leverage **Metal**; and Linux, ChromeOS, and Android implementations rely on **Vulkan**. These backend differences can affect performance characteristics, feature availability, and even bug manifestations. Additionally, hardware requirements vary by platform—some Android devices require specific GPU vendors or Android versions, while desktop platforms have their own driver and hardware prerequisites. This complexity makes comprehensive compatibility testing essential for robust WebGPU applications.

## Current Support Status (2025)

### Chrome and Edge

**Chromium-based browsers** (Google Chrome and Microsoft Edge) were the first to ship stable WebGPU support, making the API available to mainstream users starting with **version 113 in April 2023**. This initial release provided full WebGPU support on Windows, macOS, and ChromeOS, establishing these platforms as the most mature and battle-tested environments for WebGPU development.

Platform-specific support details for Chrome and Edge:

**Windows Support**: Full production support since version 113 using the **Direct3D 12 backend**. Windows represents the best-supported desktop platform due to the maturity of the D3D12 ecosystem and extensive testing. Both x86 and x64 architectures are fully supported on Windows 10 and Windows 11. Windows ARM64 support has been more gradual, with Intel Gen12+ GPU support arriving in version 144 Beta around January 2025.

**macOS Support**: Complete support since version 113 using the **Metal backend**, Apple's native graphics API. The implementation covers both Intel-based Macs and Apple Silicon (M1, M2, M3, and later) with excellent performance characteristics. Metal's modern design aligns well with WebGPU's architecture, resulting in efficient translation and minimal overhead.

**ChromeOS Support**: Production-ready since version 113 using the **Vulkan backend**. ChromeOS devices with compatible GPUs can run WebGPU applications natively, though the diversity of ChromeOS hardware means testing across multiple device tiers is advisable.

**Android Support**: Mobile WebGPU support launched with **Chrome version 121** (released in early 2024), but with specific requirements. Devices must run **Android 12 or later** and feature GPUs from **Qualcomm or ARM** vendors. This vendor restriction reflects the Vulkan driver quality requirements for WebGPU's demanding use cases. Android support currently requires enabling the feature flag via `chrome://flags/#enable-unsafe-webgpu`, indicating it remains in a testing phase despite being available in stable Chrome releases.

**Linux Support**: Linux desktop support exists but requires manual enablement through command-line flags. Users must launch Chrome with `--enable-unsafe-webgpu --ozone-platform=x11 --use-angle=vulkan` to activate WebGPU functionality. This flag requirement reflects ongoing stability and security work on Linux's diverse driver ecosystem. The implementation uses the **Vulkan backend**, leveraging the ANGLE translation layer for additional compatibility.

Chromium's WebGPU implementation is built on the **Dawn** project, a C++ library developed by Google that provides a cross-platform abstraction over native graphics APIs. Dawn serves as the production implementation for all Chromium browsers, ensuring consistency across Chrome, Edge, and other Chromium-based browsers.

### Firefox

**Mozilla Firefox** achieved a significant milestone with WebGPU by shipping stable support starting with **version 141 in July 2025**. Firefox's implementation strategy differed from Chrome's, initially focusing on a single platform before expanding coverage.

**Windows Support**: Full production support launched with Firefox 141 in July 2025. The Windows implementation uses the **Direct3D 12 backend**, similar to Chrome, providing comparable performance and compatibility on Windows 10 and Windows 11 systems.

**macOS Support**: Apple Silicon Mac support arrived with **Firefox 145**, providing native WebGPU functionality on M-series processors. Support for Intel-based Macs is progressing through Firefox Nightly builds, with a stable release anticipated in 2026. This staged rollout reflects the different testing and optimization requirements for Apple's two Mac architectures.

**Linux Support**: Available in Firefox Nightly builds for early adopters and developers, with stable release planned for 2026. Linux support uses the **Vulkan backend**, consistent with other browsers' Linux implementations.

**Android Support**: Currently hidden behind a feature flag in development builds, with broader Android support planned for 2026 as Mozilla continues testing and optimization for mobile platforms.

Firefox's WebGPU implementation is built on **wgpu**, a Rust-based graphics library that serves as both a standalone project and the foundation for Firefox's implementation. The wgpu library is also used by the Servo browser engine, creating a shared ecosystem of WebGPU tooling across multiple Mozilla projects. Rust's memory safety guarantees provide additional security benefits for a complex graphics API implementation.

### Safari

**Apple Safari** achieved comprehensive WebGPU support across its entire platform ecosystem with **version 26**, released in June 2025. Safari's implementation stands out for its breadth of platform coverage, as Apple simultaneously shipped WebGPU support across all its operating systems.

**macOS Support**: Available on **macOS Tahoe 26** and later, providing native WebGPU functionality using the **Metal backend**. Since Metal is Apple's own graphics API, Safari's implementation benefits from tight integration with the underlying platform. WebGPU is **enabled by default** in Safari 26, requiring no user configuration or feature flags.

**iOS and iPadOS Support**: Full support on **iOS 26** and **iPadOS 26**, bringing WebGPU to iPhones and iPads. This mobile support enables sophisticated graphics and compute applications on Apple's mobile devices without the restrictions typically imposed on mobile web browsers.

**visionOS Support**: Uniquely, Safari brings WebGPU to **visionOS 26**, Apple's spatial computing platform powering the Vision Pro headset. This makes WebGPU available in an augmented/virtual reality context, opening possibilities for immersive web experiences using standard web technologies.

Safari's unified release across platforms reflects Apple's integrated hardware and software strategy. The Metal backend provides consistent behavior across all Apple devices, though developers should account for hardware capability differences between, for example, an iPhone and a Mac Studio.

## Feature Detection

Proper feature detection is critical for building robust WebGPU applications that gracefully handle unsupported environments. WebGPU provides a multi-layered detection system that allows developers to check for basic API availability, adapter presence, and specific optional features.

### Checking for WebGPU Support

The first and most fundamental check is whether the browser exposes the WebGPU API at all. This is accomplished by testing for the `navigator.gpu` property:

```typescript
if (!navigator.gpu) {
  console.log('WebGPU is not supported in this browser');
  // Fall back to WebGL or display an error message
  return;
}
```

This check catches browsers that don't implement WebGPU at all, including older browser versions and browsers that have chosen not to implement the specification. The check is synchronous and should be the first gate before attempting any WebGPU initialization.

It's important to note that WebGPU is only available in **secure contexts** (HTTPS). Pages served over HTTP will not have access to `navigator.gpu` even in supporting browsers. During development, localhost is considered a secure context, but production deployments must use HTTPS.

### Adapter and Device Checks

Even if `navigator.gpu` exists, WebGPU may not be fully functional. The next layer of detection involves requesting a **GPUAdapter**, which represents the physical GPU hardware:

```typescript
const adapter = await navigator.gpu.requestAdapter({
  powerPreference: 'high-performance' // or 'low-power'
});

if (!adapter) {
  console.log('No compatible GPU adapter found');
  // The browser supports WebGPU, but no suitable GPU is available
  // This might occur with blacklisted drivers or unsupported hardware
  return;
}
```

The `requestAdapter()` method returns `null` if no suitable adapter is available. This can occur if:
- The user's GPU or driver is blacklisted due to known issues
- The hardware doesn't meet minimum requirements
- Software rendering is disabled and no hardware GPU is present
- The user has disabled GPU acceleration in browser settings

You can optionally request a fallback adapter, which provides software-based WebGPU rendering:

```typescript
const fallbackAdapter = await navigator.gpu.requestAdapter({
  forceFallbackAdapter: true
});
```

Fallback adapters guarantee availability but with significantly reduced performance. They're primarily useful for testing and debugging rather than production use.

After obtaining an adapter, you must request a **GPUDevice**, the central object for all WebGPU operations:

```typescript
const device = await adapter.requestDevice({
  requiredFeatures: [], // Optional features your application needs
  requiredLimits: {}     // Limits your application requires
});

if (!device) {
  console.log('Could not request WebGPU device');
  return;
}
```

The `requestDevice()` call can fail if you request features or limits the adapter doesn't support. This leads to the third layer of feature detection.

### Feature Detection for Optional Capabilities

WebGPU defines a core set of guaranteed features and an extensive set of optional features that may or may not be available depending on the GPU, driver, and browser implementation. Before requesting these features, you should check their availability:

```typescript
// Query available features
console.log('Supported features:', Array.from(adapter.features));

// Check for specific features
if (adapter.features.has('texture-compression-bc')) {
  console.log('BC texture compression is supported');
}

if (adapter.features.has('timestamp-query')) {
  console.log('GPU timestamp queries are supported');
}

if (adapter.features.has('shader-f16')) {
  console.log('16-bit floating point in shaders is supported');
}
```

Common optional features include:
- **texture-compression-bc**: Block compression formats (BC1-BC7) common on desktop GPUs
- **texture-compression-etc2**: ETC2/EAC compression formats common on mobile GPUs
- **texture-compression-astc**: ASTC compression, widely supported on mobile
- **timestamp-query**: High-precision GPU timing for performance measurement
- **shader-f16**: 16-bit floating point operations in shaders for memory and performance optimization
- **depth-clip-control**: Fine-grained control over depth clipping behavior
- **indirect-first-instance**: Support for indirect drawing with first instance parameter

Your application architecture should adapt based on available features, using compressed textures when available or falling back to uncompressed formats when necessary.

Similarly, you can query adapter limits to ensure the device meets your requirements:

```typescript
const limits = adapter.limits;
console.log(`Max texture size: ${limits.maxTextureDimension2D}`);
console.log(`Max buffer size: ${limits.maxBufferSize}`);
console.log(`Max bind groups: ${limits.maxBindGroups}`);

// Check if limits meet your requirements
if (limits.maxTextureDimension2D < 4096) {
  console.warn('This GPU may not support high-resolution textures');
}
```

## Platform Considerations

WebGPU's multi-backend architecture means that behavior and performance characteristics can vary significantly across platforms. Understanding these platform-specific considerations helps developers optimize for each environment and avoid platform-specific pitfalls.

### Windows Platform

**Backend**: Direct3D 12

Windows with Direct3D 12 represents the most mature and broadly tested WebGPU environment. D3D12's widespread adoption and mature tooling ecosystem means driver quality is generally high, and most GPU vendors invest heavily in D3D12 optimization.

**Hardware Requirements**: Windows 10 version 1909 or later, Windows 11, and a GPU with D3D12 feature level 11_0 or higher. Most GPUs from 2013 onwards meet this requirement, though very old or low-end hardware may be blacklisted.

**Driver Considerations**: Windows Update generally keeps GPU drivers current, but enthusiasts often install vendor-specific drivers from NVIDIA, AMD, or Intel for best performance. Driver quality is typically excellent, though new GPU generations may have initial stability issues.

**Best Practices**: Windows is the safest platform for maximum compatibility. Test on both NVIDIA and AMD GPUs if possible, as they constitute the majority of the Windows gaming and professional market.

### macOS Platform

**Backend**: Metal

macOS with Metal provides excellent WebGPU performance due to tight integration between Apple's graphics API and their hardware. Metal's modern design philosophy aligns closely with WebGPU's architecture, resulting in efficient translation layers.

**Hardware Requirements**: macOS 11 Big Sur or later for older Intel Macs; macOS 12 Monterey or later recommended for Apple Silicon. Most Macs from 2012 onwards have Metal-capable GPUs, though very old machines may lack support.

**Apple Silicon vs Intel**: Apple Silicon (M1/M2/M3) offers significantly better GPU performance and power efficiency than Intel integrated graphics. However, higher-end Intel Macs with discrete AMD GPUs can still deliver strong performance.

**Best Practices**: Use `navigator.gpu.getPreferredCanvasFormat()` to get the optimal canvas format for the platform—this is especially important on macOS where the preferred format may differ from Windows. Test on both Intel and Apple Silicon if supporting a broad Mac user base.

### Linux Platform

**Backend**: Vulkan

Linux support is the most complex due to the diversity of the Linux ecosystem—multiple distributions, desktop environments, display servers (X11, Wayland), and varying driver quality across GPU vendors.

**Hardware Requirements**: A Vulkan 1.1-capable GPU with up-to-date drivers. NVIDIA, AMD, and Intel GPUs from roughly 2014 onwards support Vulkan, but driver installation and maintenance varies by distribution.

**Driver Considerations**:
- **NVIDIA**: Proprietary drivers typically required for best performance; open-source nouveau drivers have limited Vulkan support
- **AMD**: Open-source RADV drivers provide excellent Vulkan support and are often preferred over proprietary amdgpu-pro
- **Intel**: Open-source drivers (ANV/Mesa) generally provide good Vulkan support for integrated GPUs

**Current Limitations**: As of 2025, Chrome requires command-line flags, and Firefox support is in Nightly builds. Production use on Linux requires users to manually enable features, limiting mainstream adoption.

**Best Practices**: If targeting Linux users, provide clear documentation on enabling WebGPU. Consider Linux support optional until flag requirements are removed.

### Mobile Platforms

**Android**:
- **Requirements**: Android 12+, Qualcomm (Adreno) or ARM (Mali) GPUs
- **Backend**: Vulkan 1.1 or later
- **Limitations**: Still behind feature flags in Chrome 121+; MediaTek and other GPU vendors not yet supported
- **Considerations**: Mobile GPUs have stricter power and thermal constraints; test on multiple device tiers (flagship, mid-range, budget)

**iOS and iPadOS**:
- **Requirements**: iOS 26, iPadOS 26 or later
- **Backend**: Metal
- **Support**: Fully supported in Safari 26 without flags
- **Considerations**: Apple's tight hardware/software integration provides consistent behavior, but thermal throttling on iPhones may affect sustained performance

Mobile platforms require special attention to power consumption, thermal management, and reduced memory availability compared to desktop systems. Optimize asset sizes, reduce compute intensity, and implement quality presets for different device tiers.

## Fallback Strategies

Given WebGPU's limited availability, production web applications must implement fallback strategies to support users on older browsers or unsupported platforms.

### WebGL Fallback

The most common fallback is to WebGL 2.0 (or WebGL 1.0 for maximum compatibility). This requires maintaining parallel rendering paths or using a library that abstracts the differences.

#### Manual Detection and Switching

```typescript
async function initGraphics() {
  // Try WebGPU first
  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const device = await adapter.requestDevice();
      return initWebGPURenderer(device);
    }
  }

  // Fall back to WebGL2
  const canvas = document.getElementById('canvas');
  const gl = canvas.getContext('webgl2');
  if (gl) {
    return initWebGL2Renderer(gl);
  }

  // Fall back to WebGL1
  const gl1 = canvas.getContext('webgl');
  if (gl1) {
    return initWebGL1Renderer(gl1);
  }

  // No GPU acceleration available
  throw new Error('No supported graphics API available');
}
```

This approach requires implementing and maintaining separate rendering backends, which significantly increases code complexity.

#### Library-Based Abstraction

Modern graphics libraries can handle fallback automatically:

**Three.js**: The popular 3D library added WebGPU support while maintaining WebGL compatibility. Developers can use `WebGPURenderer` for WebGPU or automatically fall back:

```typescript
import { WebGPURenderer } from 'three/webgpu';
import { WebGLRenderer } from 'three';

let renderer;
try {
  renderer = new WebGPURenderer();
  await renderer.init();
} catch (error) {
  console.log('WebGPU not available, falling back to WebGL');
  renderer = new WebGLRenderer();
}
```

**Babylon.js**: Another major 3D engine with WebGPU support and automatic fallback capabilities.

Using an abstraction library reduces maintenance burden but adds dependency weight and may not expose all WebGPU-specific features.

### Progressive Enhancement

Rather than full fallback, consider progressive enhancement where the application works everywhere but uses WebGPU for enhanced capabilities:

```typescript
class Renderer {
  private useWebGPU: boolean;

  async initialize() {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        this.device = await adapter.requestDevice();
        this.useWebGPU = true;
        return;
      }
    }

    // Initialize WebGL
    this.gl = canvas.getContext('webgl2');
    this.useWebGPU = false;
  }

  render(scene) {
    if (this.useWebGPU) {
      this.renderAdvanced(scene); // Higher quality, more features
    } else {
      this.renderBasic(scene); // Simplified rendering
    }
  }
}
```

This approach allows you to leverage WebGPU's compute capabilities for physics simulation, AI inference, or other enhancements while maintaining a functional baseline experience.

### Polyfills and Their Limitations

Currently, no production-ready WebGPU polyfill exists that implements the full API over WebGL. The architectural differences between WebGPU and WebGL make a complete polyfill impractical:

- WebGPU's explicit synchronization model differs fundamentally from WebGL's implicit synchronization
- WebGPU's compute shaders have no direct WebGL equivalent
- WGSL (WebGPU Shading Language) cannot be directly translated to GLSL in all cases
- WebGPU's resource binding model (bind groups) differs from WebGL's binding points

Some partial compatibility layers exist for specific use cases, but they cannot provide full WebGPU functionality over WebGL. The practical fallback path is maintaining separate WebGL and WebGPU implementations or using a library that abstracts both.

## Optional Features and Extensions

WebGPU's feature system allows the specification to evolve while maintaining a stable core. Applications can request optional features that enable advanced functionality when available.

### Common Optional Features

#### Texture Compression

Compressed textures significantly reduce memory usage and bandwidth:

```typescript
const adapter = await navigator.gpu.requestAdapter();

// Check platform-appropriate compression
let compressionFeature = null;
if (adapter.features.has('texture-compression-bc')) {
  compressionFeature = 'texture-compression-bc'; // Common on desktop (Windows/PC)
} else if (adapter.features.has('texture-compression-astc')) {
  compressionFeature = 'texture-compression-astc'; // Common on mobile
} else if (adapter.features.has('texture-compression-etc2')) {
  compressionFeature = 'texture-compression-etc2'; // Mobile fallback
}

const device = await adapter.requestDevice({
  requiredFeatures: compressionFeature ? [compressionFeature] : []
});
```

#### Timestamp Queries

Enable precise performance measurement:

```typescript
if (adapter.features.has('timestamp-query')) {
  const device = await adapter.requestDevice({
    requiredFeatures: ['timestamp-query']
  });

  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: 2
  });

  // Use in command encoding to measure GPU time
}
```

#### Shader F16

16-bit floating point operations reduce memory and increase performance:

```typescript
if (adapter.features.has('shader-f16')) {
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16']
  });

  // Use f16 types in WGSL shaders
}
```

### Requesting Features Safely

Always check feature availability before requesting:

```typescript
const desiredFeatures = [
  'timestamp-query',
  'shader-f16',
  'texture-compression-bc'
];

const availableFeatures = desiredFeatures.filter(
  feature => adapter.features.has(feature)
);

const device = await adapter.requestDevice({
  requiredFeatures: availableFeatures
});
```

If you request a feature the adapter doesn't support, `requestDevice()` will fail, so defensive checking is essential.

## Future Outlook

WebGPU continues to evolve with new features and broader browser support on the horizon.

### Expected Browser Improvements

**Firefox**: Full Linux and Intel Mac support expected in stable releases throughout 2026, with Android support also planned for 2026. This will significantly expand WebGPU's reach on non-Chromium browsers.

**Chrome/Edge**: Removal of command-line flags for Linux support anticipated in 2025-2026. Android support expansion to additional GPU vendors (MediaTek, etc.) expected as driver quality improves.

**Safari**: Already feature-complete across Apple platforms; future updates likely to focus on performance optimization and new optional features.

### Roadmap Features

The WebGPU specification continues to develop new capabilities:

**Subgroups**: GPU subgroup operations (also called wave/warp operations) enable advanced optimization techniques by allowing shader threads to communicate within hardware execution groups. Subgroup support would bring WebGPU closer to native API capabilities for high-performance compute.

**Advanced Texture Formats**: Additional formats for specialized use cases, including higher-precision formats and video texture integration.

**Direct Storage Integration**: Potential future integration with emerging storage APIs for faster asset loading directly to GPU memory.

**Ray Tracing**: While not currently in the specification, ray tracing hardware has become commonplace. Future WebGPU versions may expose ray tracing capabilities similar to DXR, Metal ray tracing, and Vulkan ray tracing extensions.

### Adoption Trajectory

Based on current trends, WebGPU is expected to achieve "Baseline" status (available in all major browsers) by late 2026 or 2027, once Firefox completes its platform rollout and flag requirements are removed. This timeline assumes no major setbacks and continued prioritization by browser vendors.

## Testing Across Browsers

Comprehensive testing is essential for WebGPU applications due to platform and browser variations.

### Testing Strategy

**Multi-Browser Testing**: Test on Chrome, Firefox, and Safari to catch browser-specific issues. Each uses different WebGPU implementations (Dawn, wgpu, WebKit) that may expose different bugs or performance characteristics.

**Multi-Platform Testing**: Test on Windows, macOS, and Linux if possible. The different backends (D3D12, Metal, Vulkan) can produce different results for the same WebGPU code.

**Hardware Diversity**: Test on multiple GPU vendors (NVIDIA, AMD, Intel, Apple Silicon) and different performance tiers. Budget GPUs may hit limit thresholds that high-end GPUs don't, revealing portability issues.

**Mobile Testing**: If targeting mobile, test on both Android (multiple device tiers, multiple GPU vendors) and iOS (different iPhone/iPad models).

### Automated Testing

WebGPU testing can be integrated into CI/CD pipelines:

```typescript
// Example test structure
describe('WebGPU Compatibility', () => {
  it('should detect WebGPU support', () => {
    expect(navigator.gpu).toBeDefined();
  });

  it('should request adapter', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    expect(adapter).not.toBeNull();
  });

  it('should create device', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    expect(device).toBeDefined();
  });
});
```

Consider using headless browser testing with browsers that support WebGPU in headless mode, though GPU testing in CI environments can be challenging due to lack of GPU hardware in typical CI runners.

### Browser Developer Tools

All major browsers provide WebGPU debugging capabilities:

**Chrome DevTools**: Inspect GPU resources, capture command buffers, view validation errors, and profile GPU performance.

**Safari Web Inspector**: Debug Metal backend performance and resource usage.

**Firefox Developer Tools**: Profile and debug wgpu-based WebGPU implementation.

## Best Practices

### Always Perform Feature Detection

Never assume WebGPU is available. Always check `navigator.gpu`, request an adapter, and handle null returns gracefully:

```typescript
async function safeWebGPUInit() {
  if (!navigator.gpu) {
    return { supported: false, reason: 'WebGPU not supported' };
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return { supported: false, reason: 'No GPU adapter available' };
  }

  const device = await adapter.requestDevice();
  return { supported: true, device };
}
```

### Provide User-Friendly Fallbacks

When WebGPU is unavailable, inform users clearly:

```typescript
const result = await safeWebGPUInit();
if (!result.supported) {
  displayMessage(
    'WebGPU is not available in your browser. ' +
    'Please update to the latest version of Chrome, Firefox, or Safari. ' +
    'The application will use WebGL instead with reduced features.'
  );
  initWebGLFallback();
}
```

### Use Preferred Canvas Format

Always query the preferred canvas format rather than hardcoding:

```typescript
const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device,
  format: preferredFormat,
  // other configuration
});
```

This ensures optimal performance and compatibility across platforms.

### Test Broadly

Don't develop exclusively on one browser or platform. WebGPU behavior varies enough that multi-platform testing is essential for production applications.

### Monitor Browser Support Status

WebGPU support is evolving. Regularly check resources like [caniuse.com/webgpu](https://caniuse.com/webgpu) and the [WebGPU implementation status wiki](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) for updates.

## Common Pitfalls

### Assuming WebGPU Availability

**Pitfall**: Writing code that assumes `navigator.gpu` exists without checking.

**Solution**: Always feature-detect and handle unsupported environments gracefully.

### Forgetting Secure Context Requirement

**Pitfall**: Testing locally over HTTP or deploying without HTTPS, causing WebGPU to be unavailable.

**Solution**: Always use HTTPS in production. During development, use localhost or enable browser flags for insecure origins (testing only).

### Ignoring Mobile Constraints

**Pitfall**: Developing on desktop and assuming the same assets and performance will work on mobile.

**Solution**: Test on actual mobile devices, reduce asset sizes, implement quality settings, and monitor thermal throttling.

### Not Testing Cross-Platform

**Pitfall**: Developing exclusively on one OS (often macOS or Windows) and discovering platform-specific bugs late.

**Solution**: Test on all target platforms throughout development, not just before release.

### Requesting Unavailable Features

**Pitfall**: Requesting features without checking availability, causing `requestDevice()` to fail.

**Solution**: Check `adapter.features.has()` before adding features to `requiredFeatures`.

### Hardcoding Texture Formats

**Pitfall**: Hardcoding canvas format as `'bgra8unorm'` or `'rgba8unorm'` instead of querying the preferred format.

**Solution**: Use `navigator.gpu.getPreferredCanvasFormat()` to get the platform-optimal format.

### Ignoring Adapter Limits

**Pitfall**: Creating resources that exceed adapter limits, causing device loss or initialization failures.

**Solution**: Query `adapter.limits` and ensure your application respects hardware constraints, or request higher limits through `requestDevice()`.

---

## Summary Table: Browser Support Overview

| Browser | Version | Platforms | Status | Backend |
|---------|---------|-----------|--------|---------|
| Chrome | 113+ | Windows, macOS, ChromeOS | Stable | Dawn (D3D12/Metal/Vulkan) |
| Chrome | 121+ | Android 12+ (Qualcomm/ARM) | Flag required | Dawn (Vulkan) |
| Chrome | - | Linux | Flags required | Dawn (Vulkan) |
| Edge | 113+ | Windows, macOS | Stable | Dawn (D3D12/Metal) |
| Firefox | 141+ | Windows | Stable | wgpu (D3D12) |
| Firefox | 145+ | macOS (Apple Silicon) | Stable | wgpu (Metal) |
| Firefox | Nightly | macOS (Intel), Linux | Development | wgpu (Metal/Vulkan) |
| Safari | 26+ | macOS Tahoe 26+ | Stable | WebKit (Metal) |
| Safari | 26+ | iOS, iPadOS, visionOS 26+ | Stable | WebKit (Metal) |

This comprehensive guide should help you navigate the complex landscape of WebGPU browser compatibility and make informed decisions about when and how to deploy WebGPU applications to production.
