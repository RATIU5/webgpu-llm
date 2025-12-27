# Documentation Enhancement Plan

Based on research of [TypeGPU GitHub](https://github.com/software-mansion/TypeGPU), [WebGPU spec](https://www.w3.org/TR/webgpu/), [WGSL spec](https://www.w3.org/TR/WGSL/), and [WebGPU Fundamentals](https://webgpufundamentals.org/), here are the identified gaps and proposed additions.

---

## Priority 1: Missing Core WebGPU Features

### 1.1 New Document: `shaders-and-pipelines/instancing-indirect-drawing.md`

**Why missing matters:** Instancing is fundamental for rendering many objects efficiently. Indirect drawing enables GPU-driven rendering.

**Content:**
- Instanced rendering with `draw(vertexCount, instanceCount)`
- `@builtin(instance_index)` usage
- Instance step mode in vertex buffers
- Indirect buffers (`GPUBufferUsage.INDIRECT`)
- `drawIndirect()` and `drawIndexedIndirect()`
- GPU-driven culling pattern

---

### 1.2 New Document: `shaders-and-pipelines/render-bundles.md`

**Why missing matters:** Render bundles reduce CPU overhead for repeated draw sequences—critical for VR and CPU-bound apps.

**Content:**
- `GPURenderBundleEncoder` creation
- Recording commands into bundles
- Executing via `executeBundles()`
- When bundles help (CPU-bound) vs. don't (GPU-bound)
- Combining with indirect draws for dynamic content

---

### 1.3 New Document: `resources-and-binding/storage-textures.md`

**Why missing matters:** Storage textures allow direct writes from compute shaders—essential for image processing and procedural generation.

**Content:**
- Write-only vs. read-write storage textures
- Supported formats (`r32float`, `rgba8unorm`, etc.)
- WGSL declaration: `texture_storage_2d<format, access>`
- `textureStore()` function
- Use cases: post-processing, compute-generated images

---

### 1.4 New Document: `resources-and-binding/multisampling-msaa.md`

**Why missing matters:** MSAA is the standard anti-aliasing approach. Currently not documented.

**Content:**
- Sample counts (1 or 4 in WebGPU v1)
- Creating multisampled textures
- Multisampled render attachments
- Resolve targets
- Pipeline `multisample` configuration
- `texture_multisampled_2d` in WGSL

---

### 1.5 Addition to `resources-and-binding/textures-samplers.md`

**Missing content to add:**
- **Mipmap generation** (not built-in—compute shader approach)
- **Depth textures** (`depth24plus`, `depth32float`)
- **Comparison samplers** for shadow mapping
- `textureSampleCompare()` function

---

## Priority 2: Missing TypeGPU Features

### 2.1 New Document: `fundamentals/typegpu-tooling.md`

**Why missing matters:** `tgpu-gen` and `unplugin-typegpu` are official tools not documented.

**Content:**
- **tgpu-gen CLI**: Automatic code generation
- **unplugin-typegpu**: Vite/Webpack/Rollup plugin
- `"use gpu"` directive for shell-less functions
- Build configuration examples

---

### 2.2 New Document: `fundamentals/typegpu-utilities.md`

**Why missing matters:** Helper packages exist but aren't documented.

**Content:**
- **typegpu-color**: Color manipulation functions
- **typegpu-noise**: Perlin, simplex, and random utilities
- Zero-initialized schema values (new in v0.7)
- Usage patterns with TGSL functions

---

### 2.3 Addition to `advanced/advanced-typegpu-patterns.md`

**Missing content to add:**
- Automatic mipmap generation (new texture API)
- TypeGPU + Three.js integration pattern
- TypeGPU + Babylon.js integration pattern

---

### 2.4 New Document: `advanced/react-integration.md`

**Why missing matters:** React is dominant in web dev; patterns for hooks/components aren't documented.

**Content:**
- `useWebGPU()` hook pattern
- Managing device lifecycle with React
- Cleanup with `useEffect`
- Rendering to canvas refs
- State management for GPU resources
- React Native with `react-native-wgpu`

---

## Priority 3: Missing WGSL Reference Content

### 3.1 New Document: `fundamentals/wgsl-builtin-functions.md`

**Why missing matters:** No quick reference for WGSL functions exists in the docs.

**Content:**

| Category | Functions |
|----------|-----------|
| Math | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `pow`, `exp`, `log`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `fract`, `sign` |
| Vector | `dot`, `cross`, `length`, `distance`, `normalize`, `reflect`, `refract`, `faceForward` |
| Interpolation | `mix`, `smoothstep`, `step`, `clamp`, `saturate`, `min`, `max` |
| Texture | `textureSample`, `textureSampleLevel`, `textureSampleGrad`, `textureSampleCompare`, `textureLoad`, `textureStore`, `textureDimensions` |
| Derivatives | `dpdx`, `dpdy`, `dpdxCoarse`, `dpdyCoarse`, `dpdxFine`, `dpdyFine`, `fwidth` |
| Pack/Unpack | `pack4x8snorm`, `unpack4x8snorm`, `pack2x16float`, etc. |

---

### 3.2 Addition to `fundamentals/wgsl-shading-language.md`

**Missing content to add:**
- **Control flow**: `if/else`, `for`, `while`, `loop`, `switch`, `break`, `continue`, `continuing`
- **Pointers and references**: `ptr<address_space, T>`, `&` operator
- **Const assertions**: `const_assert`
- **Override declarations**: Pipeline-overridable constants

---

## Priority 4: Practical Topics

### 4.1 New Document: `data-and-buffers/loading-images-textures.md`

**Why missing matters:** Loading images is a fundamental task not explicitly covered.

**Content:**
- Loading with `createImageBitmap()`
- `copyExternalImageToTexture()`
- Handling image orientation (flipY)
- Compressed texture loading (BC, ETC2, ASTC)
- Video frame capture with `importExternalTexture()`

---

### 4.2 New Document: `misc/camera-view-matrices.md`

**Why missing matters:** Camera math is essential for 3D but not covered. Coordinate systems doc exists but lacks camera patterns.

**Content:**
- Look-at matrix construction
- Orbit camera pattern
- First-person camera pattern
- Projection matrix types (perspective, orthographic)
- Using `wgpu-matrix` for camera math

---

## Priority 5: Enhancement to Existing Documents

### 5.1 `shaders-and-pipelines/compute-pipelines.md`

**Add:**
- Workgroup shared memory patterns (already in workgroup-variables.md, but cross-reference)
- Compute shader dispatching strategies
- Synchronization with `storageBarrier()`, `workgroupBarrier()`

---

### 5.2 `misc/timestamp-queries-profiling.md`

**Add:**
- Occlusion queries (`type: "occlusion"`)
- `beginOcclusionQuery()` / `endOcclusionQuery()`
- Visibility testing patterns

---

### 5.3 `fundamentals/error-handling-validation.md`

**Add:**
- Device lost recovery patterns (expand beyond mention)
- Memory pressure handling
- Graceful degradation strategies

---

## Summary Table

| Priority | Type | Document/Section | Effort |
|----------|------|------------------|--------|
| P1 | New | instancing-indirect-drawing.md | Medium |
| P1 | New | render-bundles.md | Medium |
| P1 | New | storage-textures.md | Small |
| P1 | New | multisampling-msaa.md | Medium |
| P1 | Add | textures-samplers.md (mipmaps, depth) | Small |
| P2 | New | typegpu-tooling.md | Medium |
| P2 | New | typegpu-utilities.md | Small |
| P2 | Add | advanced-typegpu-patterns.md | Small |
| P2 | New | react-integration.md | Medium |
| P3 | New | wgsl-builtin-functions.md | Medium |
| P3 | Add | wgsl-shading-language.md (control flow, pointers) | Small |
| P4 | New | loading-images-textures.md | Medium |
| P4 | New | camera-view-matrices.md | Medium |
| P5 | Add | compute-pipelines.md | Small |
| P5 | Add | timestamp-queries-profiling.md (occlusion) | Small |
| P5 | Add | error-handling-validation.md | Small |

---

## Recommended Implementation Order

1. **Phase A** (Core WebGPU gaps): P1 items—these are standard WebGPU features missing entirely
2. **Phase B** (TypeGPU completeness): P2 items—tools and utilities users need
3. **Phase C** (WGSL reference): P3 items—essential reference material
4. **Phase D** (Practical topics): P4 items—common real-world tasks
5. **Phase E** (Enhancements): P5 items—improve existing docs

---

## Sources

- [TypeGPU GitHub](https://github.com/software-mansion/TypeGPU)
- [TypeGPU Releases](https://github.com/software-mansion/TypeGPU/releases)
- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [MDN WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)
- [Toji's WebGPU Best Practices](https://toji.dev/webgpu-best-practices/)
- [Can I Use WebGPU](https://caniuse.com/webgpu)
