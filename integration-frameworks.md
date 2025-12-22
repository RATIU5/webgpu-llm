# Integration with Three.js and Babylon.js

## Overview

As WebGPU gains adoption across modern browsers, many developers seek to combine the type safety and developer experience of TypeGPU with the established ecosystems of Three.js and Babylon.js. These popular 3D frameworks offer comprehensive scene management, optimized renderers, and extensive tooling, while TypeGPU provides type-safe compute shader capabilities and TGSL (TypeGPU Shading Language) for writing GPU code in TypeScript.

Integrating TypeGPU with Three.js or Babylon.js unlocks powerful hybrid architectures where you leverage the strengths of each library. Use established frameworks for rendering, camera controls, lighting, and scene management, while employing TypeGPU for custom compute operations, physics simulations, procedural generation, and type-safe shader development. This approach provides the best of both worlds: mature rendering systems with cutting-edge GPU compute capabilities.

This integration pattern is particularly valuable for applications requiring:

- **Custom physics engines**: GPU-accelerated particle systems, fluid dynamics, or cloth simulation rendered through framework pipelines
- **Procedural generation**: Compute-based terrain generation, mesh deformation, or texture synthesis
- **Data visualization**: Large-scale scientific visualization where compute shaders process data and frameworks handle rendering
- **Advanced post-processing**: Custom effects pipelines that operate on framework-rendered scenes

The key challenge is managing shared GPU resources—particularly the GPUDevice instance and associated buffers, textures, and command queues. Both Three.js and Babylon.js provide WebGPU renderers that create and manage a device internally. TypeGPU needs access to this same device to create compute pipelines and execute custom shaders. Proper integration requires understanding device sharing, resource interoperability, and command submission order.

Throughout this guide, we'll explore practical integration patterns, complete working examples, and best practices for combining TypeGPU with these popular frameworks. Whether you're building a game, scientific visualization, or interactive experience, these techniques will help you harness the full power of WebGPU across multiple libraries.

## Why Integrate?

### Leveraging Existing Ecosystems

Three.js and Babylon.js represent years of development effort, battle-testing, and community contributions. These frameworks provide:

**Mature Rendering Pipelines**: Optimized renderers supporting advanced lighting models (PBR, IBL), shadow mapping, post-processing effects, and multi-pass rendering strategies. Reimplementing these features from scratch would require months or years of work.

**Scene Management**: Hierarchical scene graphs with automatic frustum culling, level-of-detail systems, and efficient spatial indexing. These systems handle thousands of objects efficiently, managing visibility determination and render batching automatically.

**Asset Loading and Processing**: Comprehensive loaders for GLTF, FBX, OBJ, and numerous other formats. Frameworks handle texture compression, animation retargeting, and material conversion automatically.

**Cross-Platform Abstractions**: While focusing on WebGPU integration here, both frameworks support fallback renderers (WebGL, WebGL2), enabling broader device compatibility without rewriting application logic.

**Tooling and Ecosystem**: Inspector tools, editor integrations (Blender exporters, Unity converters), extensive documentation, and large communities provide invaluable support during development.

By integrating TypeGPU, you add type-safe compute capabilities and custom shader authoring to this foundation rather than building everything from scratch.

### Custom Compute Operations

Three.js and Babylon.js excel at rendering but provide limited support for arbitrary compute operations. TypeGPU fills this gap:

**Physics Simulations**: Implement custom physics solvers on the GPU. Calculate forces, integrate motion, and handle collisions for thousands of objects simultaneously. Results feed directly into framework rendering without CPU roundtrips.

**Particle Systems**: While frameworks include basic particle systems, custom implementations offer unlimited flexibility. Simulate millions of particles with complex behaviors (flocking, fluid dynamics, electromagnetic interactions) using TypeGPU compute shaders.

**Mesh Processing**: Perform skinning, morphing, or procedural deformation in compute shaders. Generate or modify mesh geometry on the GPU, then pass results to framework rendering.

**Data Analysis**: Process scientific datasets, perform statistical calculations, or execute machine learning inference on the GPU while visualizing results through framework rendering capabilities.

### Type-Safe Shader Development

Traditional shader development involves writing WGSL or GLSL in string templates, losing TypeScript's type checking, autocomplete, and refactoring capabilities. TGSL (TypeGPU Shading Language) addresses this:

**IDE Integration**: Full autocomplete, go-to-definition, and type checking within shader code. Refactor shader functions with confidence that TypeScript will catch all affected references.

**Compile-Time Validation**: Type mismatches between CPU and GPU code are caught during compilation rather than at runtime. This prevents entire classes of bugs common in traditional shader development.

**Code Reuse**: Share utility functions, mathematical operations, and data structures between CPU and GPU code. Define complex types once and use them consistently across your application.

**Gradual Adoption**: Mix TGSL and WGSL freely. Start with existing shaders and incrementally convert critical portions to TypeScript for improved maintainability.

### Performance Optimizations

Combining frameworks with TypeGPU enables optimization strategies impossible with either alone:

**Hybrid Rendering**: Use framework renderers for standard objects while implementing custom rendering kernels for specialized cases (massive instancing, volume rendering, ray marching).

**Asynchronous Compute**: While the framework renders one frame, use TypeGPU to compute physics or procedural content for the next frame on separate compute queues.

**Memory Management**: Share buffers between compute and render pipelines, eliminating expensive CPU-GPU transfers. Update vertex buffers directly in compute shaders consumed by framework rendering.

**Workload Balancing**: Offload CPU-intensive tasks to GPU compute shaders, freeing CPU resources for game logic, AI, or other systems while frameworks handle rendering.

## Three.js Integration

### Three.js WebGPU Renderer

Three.js has been actively developing WebGPU support through its experimental WebGPURenderer. This modern renderer leverages WebGPU's capabilities while maintaining Three.js's familiar API.

**Current Status**: As of Three.js r160+, WebGPURenderer is available as an experimental feature. It supports most core Three.js functionality including PBR materials, shadows, post-processing, and the Three.js Shading Language (TSL) node-based material system. While not yet feature-complete compared to the WebGL renderer, it's production-ready for many use cases.

**Creating a WebGPU Renderer**:

```typescript
import { WebGPURenderer } from 'three/webgpu';
import { Scene, PerspectiveCamera, Mesh, BoxGeometry, MeshStandardNodeMaterial } from 'three';

async function initThreeJS() {
  // Create renderer - automatically initializes WebGPU
  const renderer = new WebGPURenderer({ antialias: true });
  await renderer.init(); // Asynchronous initialization

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);

  // Setup scene
  const scene = new Scene();
  const camera = new PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 5;

  // Create a simple mesh with node material
  const geometry = new BoxGeometry(1, 1, 1);
  const material = new MeshStandardNodeMaterial({ color: 0x00ff00 });
  const cube = new Mesh(geometry, material);
  scene.add(cube);

  return { renderer, scene, camera, cube };
}
```

**Accessing the GPUDevice**: The WebGPURenderer creates and manages a GPUDevice internally. Access it for TypeGPU integration:

```typescript
const renderer = new WebGPURenderer();
await renderer.init();

// Access the backend which contains the device
const backend = renderer.backend;
const device = backend.device; // GPUDevice instance

console.log('Device limits:', device.limits);
console.log('Device features:', device.features);
```

This device instance is what you'll share with TypeGPU for compute operations and custom shader development.

### Using TypeGPU Compute with Three.js

The most common integration pattern uses Three.js for rendering and TypeGPU for compute operations. Here's a complete example implementing GPU-based particle physics:

```typescript
import { WebGPURenderer } from 'three/webgpu';
import { Scene, PerspectiveCamera, BufferGeometry, BufferAttribute, Points, PointsMaterial } from 'three';
import tgpu from 'typegpu';
import { arrayOf, vec3f, vec4f, f32, struct } from 'typegpu/data';

// Define particle structure
const Particle = struct({
  position: vec3f,
  velocity: vec3f,
  color: vec4f,
  lifetime: f32,
  age: f32,
});

const SimParams = struct({
  deltaTime: f32,
  particleCount: f32,
  gravity: vec3f,
  damping: f32,
});

async function createParticleSystem() {
  // Initialize Three.js
  const renderer = new WebGPURenderer({ antialias: true });
  await renderer.init();

  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const scene = new Scene();
  const camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 5;

  // Get shared GPUDevice
  const device = renderer.backend.device;

  // Initialize TypeGPU with existing device
  const root = tgpu.initFromDevice(device);

  // Create particle data
  const particleCount = 10000;
  const initialParticles = new Float32Array(particleCount * 12); // 12 floats per particle

  for (let i = 0; i < particleCount; i++) {
    const offset = i * 12;
    // Position
    initialParticles[offset + 0] = (Math.random() - 0.5) * 4;
    initialParticles[offset + 1] = (Math.random() - 0.5) * 4;
    initialParticles[offset + 2] = (Math.random() - 0.5) * 4;
    // Velocity
    initialParticles[offset + 3] = (Math.random() - 0.5) * 2;
    initialParticles[offset + 4] = (Math.random() - 0.5) * 2;
    initialParticles[offset + 5] = (Math.random() - 0.5) * 2;
    // Color
    initialParticles[offset + 6] = Math.random();
    initialParticles[offset + 7] = Math.random();
    initialParticles[offset + 8] = Math.random();
    initialParticles[offset + 9] = 1.0;
    // Lifetime and age
    initialParticles[offset + 10] = 5.0 + Math.random() * 3.0;
    initialParticles[offset + 11] = Math.random() * 5.0;
  }

  // Create GPU buffers for double buffering
  const bufferA = device.createBuffer({
    size: initialParticles.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX,
    label: 'Particle Buffer A',
  });

  const bufferB = device.createBuffer({
    size: initialParticles.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX,
    label: 'Particle Buffer B',
  });

  device.queue.writeBuffer(bufferA, 0, initialParticles);
  device.queue.writeBuffer(bufferB, 0, initialParticles);

  // Create simulation parameters buffer
  const paramsBuffer = root
    .createBuffer(SimParams, {
      deltaTime: 0.016,
      particleCount: particleCount,
      gravity: [0, -1.0, 0],
      damping: 0.99,
    })
    .$usage('uniform');

  // Create TypeGPU compute shader
  const updateParticles = tgpu
    .fn([arrayOf(Particle), arrayOf(Particle), SimParams])
    .does(`
      const idx = workgroupId.x * 64 + localInvocationId.x;

      if (idx >= params.particleCount) return;

      let particle = particlesIn[idx];

      // Update age
      particle.age += params.deltaTime;

      // Respawn if too old
      if (particle.age >= particle.lifetime) {
        particle.age = 0.0;
        particle.position = vec3f(
          (fract(sin(f32(idx) * 12.9898)) - 0.5) * 4.0,
          (fract(sin(f32(idx) * 78.233)) - 0.5) * 4.0,
          (fract(sin(f32(idx) * 45.164)) - 0.5) * 4.0
        );
      } else {
        // Apply physics
        particle.velocity += params.gravity * params.deltaTime;
        particle.velocity *= params.damping;
        particle.position += particle.velocity * params.deltaTime;
      }

      particlesOut[idx] = particle;
    `)
    .$uses({ particlesIn: bufferA, particlesOut: bufferB, params: paramsBuffer })
    .$name('updateParticles');

  const computePipeline = root
    .makeComputePipeline(updateParticles)
    .$workgroupSize(64);

  // Create Three.js points for rendering
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('color', new BufferAttribute(colors, 3));

  const material = new PointsMaterial({ size: 0.05, vertexColors: true });
  const points = new Points(geometry, material);
  scene.add(points);

  // Animation loop
  let currentBuffer = 0;
  let lastTime = performance.now();

  function animate() {
    const now = performance.now();
    const deltaTime = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    // Update simulation parameters
    paramsBuffer.write({ deltaTime });

    // Run compute shader
    const inputBuffer = currentBuffer === 0 ? bufferA : bufferB;
    const outputBuffer = currentBuffer === 0 ? bufferB : bufferA;

    root
      .createCommandEncoder()
      .beginComputePass()
      .setPipeline(computePipeline)
      .setBindGroup(0, updateParticles.createBindGroup([inputBuffer, outputBuffer, paramsBuffer]))
      .dispatchWorkgroups(Math.ceil(particleCount / 64))
      .end()
      .submit();

    // Copy GPU buffer data back to Three.js geometry
    // Note: This is a simplified example - production code should use
    // GPU-to-GPU copies or shared buffers for better performance
    device.queue.onSubmittedWorkDone().then(() => {
      // Read positions from outputBuffer and update geometry
      // In practice, you'd use a more efficient method
    });

    currentBuffer = 1 - currentBuffer;

    // Render with Three.js
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  animate();

  return { renderer, scene, camera, points };
}

createParticleSystem().catch(console.error);
```

**Key Integration Points**:

1. **Device Sharing**: Use `tgpu.initFromDevice(device)` to wrap Three.js's GPUDevice
2. **Buffer Usage Flags**: Combine `STORAGE` (for compute), `VERTEX` (for rendering), and `COPY_SRC`/`COPY_DST` (for data transfer)
3. **Command Ordering**: Submit TypeGPU compute commands before Three.js render commands to ensure data is ready
4. **Resource Lifecycle**: Let Three.js manage the device lifecycle; TypeGPU uses it without taking ownership

### Custom Materials with TGSL

While Three.js provides a node-based material system (TSL), you can also use TGSL for maximum type safety. Here's how to create custom shader materials:

```typescript
import { WebGPURenderer } from 'three/webgpu';
import { ShaderMaterial, Mesh, PlaneGeometry } from 'three';
import tgpu from 'typegpu';

async function createCustomMaterial() {
  const renderer = new WebGPURenderer();
  await renderer.init();

  const device = renderer.backend.device;
  const root = tgpu.initFromDevice(device);

  // Define uniforms with TGSL
  const Uniforms = tgpu.struct({
    time: tgpu.f32,
    resolution: tgpu.vec2f,
    color: tgpu.vec3f,
  });

  // Create vertex shader with TGSL
  const vertexShader = tgpu
    .vertexFn([
      tgpu.param('position', tgpu.vec3f, { location: 0 }),
      tgpu.param('uv', tgpu.vec2f, { location: 1 }),
      tgpu.param('uniforms', Uniforms),
    ])
    .outputs({
      position: tgpu.builtin.position,
      vUv: tgpu.vec2f,
    })
    .implement((position, uv, uniforms) => {
      return {
        position: tgpu.vec4f(position, 1.0),
        vUv: uv,
      };
    });

  // Create fragment shader with TGSL
  const fragmentShader = tgpu
    .fragmentFn([
      tgpu.param('vUv', tgpu.vec2f),
      tgpu.param('uniforms', Uniforms),
    ])
    .outputs({
      color: tgpu.vec4f,
    })
    .implement((vUv, uniforms) => {
      // Animated gradient effect
      const wave = tgpu.sin(vUv.x * 10.0 + uniforms.time) * 0.5 + 0.5;
      const color = tgpu.mix(
        tgpu.vec3f(1.0, 0.0, 0.0),
        uniforms.color,
        wave
      );

      return {
        color: tgpu.vec4f(color, 1.0),
      };
    });

  // Convert to WGSL and create Three.js ShaderMaterial
  const vertexWGSL = vertexShader.toWGSL();
  const fragmentWGSL = fragmentShader.toWGSL();

  const material = new ShaderMaterial({
    vertexShader: vertexWGSL,
    fragmentShader: fragmentWGSL,
    uniforms: {
      time: { value: 0.0 },
      resolution: { value: [window.innerWidth, window.innerHeight] },
      color: { value: [0.0, 1.0, 0.0] },
    },
  });

  // Create mesh
  const geometry = new PlaneGeometry(2, 2);
  const mesh = new Mesh(geometry, material);

  // Update uniforms in animation loop
  function updateMaterial(time: number) {
    material.uniforms.time.value = time;
  }

  return { material, mesh, updateMaterial };
}
```

**Integration Pattern**:
- Define shader logic in type-safe TGSL
- Convert to WGSL for Three.js consumption
- Use Three.js's ShaderMaterial as the interface
- Benefit from TypeScript's type checking during development

### Sharing Buffers and Textures

Efficient integration requires sharing GPU resources between TypeGPU and Three.js without redundant copies:

```typescript
import { WebGPURenderer } from 'three/webgpu';
import { DataTexture, FloatType, RGBAFormat } from 'three';
import tgpu from 'typegpu';

async function sharedTextureExample() {
  const renderer = new WebGPURenderer();
  await renderer.init();

  const device = renderer.backend.device;
  const root = tgpu.initFromDevice(device);

  const width = 512;
  const height = 512;

  // Create GPU texture
  const gpuTexture = device.createTexture({
    size: { width, height },
    format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING |
           GPUTextureUsage.TEXTURE_BINDING |
           GPUTextureUsage.COPY_SRC |
           GPUTextureUsage.RENDER_ATTACHMENT,
    label: 'Shared Texture',
  });

  // Wrap in TypeGPU texture
  const tgpuTexture = root.wrapTexture(gpuTexture, {
    format: 'rgba32float',
    dimension: '2d',
  });

  // Create compute shader to generate texture data
  const generateTexture = tgpu
    .computeFn([tgpuTexture])
    .workgroupSize(8, 8)
    .does(`
      const coord = vec2u(
        workgroupId.x * 8u + localInvocationId.x,
        workgroupId.y * 8u + localInvocationId.y
      );

      if (coord.x >= 512u || coord.y >= 512u) return;

      // Generate procedural pattern
      let color = vec4f(
        f32(coord.x) / 512.0,
        f32(coord.y) / 512.0,
        0.5,
        1.0
      );

      textureStore(texture, coord, color);
    `)
    .$uses({ texture: tgpuTexture });

  // Execute compute shader
  const pipeline = root.makeComputePipeline(generateTexture);

  root
    .createCommandEncoder()
    .beginComputePass()
    .setPipeline(pipeline)
    .dispatchWorkgroups(64, 64) // 512 / 8 = 64
    .end()
    .submit();

  // Create Three.js texture from same GPU texture
  // Note: Three.js doesn't directly support wrapping external GPUTextures,
  // so you'd typically use a staging buffer to transfer data or use
  // backend-specific APIs

  const threeTexture = new DataTexture(
    new Float32Array(width * height * 4),
    width,
    height,
    RGBAFormat,
    FloatType
  );

  // In practice, you'd set up a render target that shares the GPU texture
  // or use backend APIs to reference the same underlying resource

  return { gpuTexture, tgpuTexture, threeTexture };
}
```

**Resource Management Considerations**:

1. **Buffer Usage Flags**: Ensure buffers have all necessary usage flags for both compute and rendering
2. **Format Compatibility**: Match texture formats between TypeGPU and Three.js (e.g., 'rgba8unorm', 'rgba32float')
3. **Ownership**: Three.js typically manages resource lifecycles; avoid destroying shared resources prematurely
4. **Synchronization**: Use command encoder ordering to ensure compute completes before rendering uses results

### Complete Example: Particle Simulation with Three.js Rendering

Here's a production-ready particle system combining TypeGPU compute with Three.js rendering:

```typescript
import { WebGPURenderer } from 'three/webgpu';
import {
  Scene,
  PerspectiveCamera,
  BufferGeometry,
  Points,
  ShaderMaterial,
  FloatType,
  RGBAFormat,
  DataTexture,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import tgpu from 'typegpu';
import { struct, vec3f, vec4f, f32, arrayOf } from 'typegpu/data';

const Particle = struct({
  position: vec3f,
  velocity: vec3f,
  color: vec4f,
  lifetime: f32,
});

class GPUParticleSystem {
  private device: GPUDevice;
  private root: any;
  private renderer: WebGPURenderer;
  private scene: Scene;
  private camera: PerspectiveCamera;
  private controls: OrbitControls;
  private particleCount: number;
  private bufferA: GPUBuffer;
  private bufferB: GPUBuffer;
  private currentBuffer: number = 0;
  private computePipeline: any;
  private points: Points;

  constructor(particleCount: number = 50000) {
    this.particleCount = particleCount;
  }

  async init() {
    // Initialize Three.js renderer
    this.renderer = new WebGPURenderer({ antialias: true });
    await this.renderer.init();

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(this.renderer.domElement);

    // Setup scene
    this.scene = new Scene();
    this.camera = new PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.z = 10;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);

    // Get device and initialize TypeGPU
    this.device = this.renderer.backend.device;
    this.root = tgpu.initFromDevice(this.device);

    // Initialize particle buffers
    await this.initParticleBuffers();

    // Create compute pipeline
    this.createComputePipeline();

    // Create rendering
    this.createParticleRendering();

    // Handle resize
    window.addEventListener('resize', () => this.onResize());

    return this;
  }

  private async initParticleBuffers() {
    const particleSize = 13 * 4; // 13 floats * 4 bytes
    const bufferSize = particleSize * this.particleCount;

    // Initialize particle data
    const initialData = new Float32Array(this.particleCount * 13);
    for (let i = 0; i < this.particleCount; i++) {
      const offset = i * 13;
      // Position
      initialData[offset + 0] = (Math.random() - 0.5) * 10;
      initialData[offset + 1] = (Math.random() - 0.5) * 10;
      initialData[offset + 2] = (Math.random() - 0.5) * 10;
      // Velocity
      initialData[offset + 3] = (Math.random() - 0.5) * 1;
      initialData[offset + 4] = (Math.random() - 0.5) * 1;
      initialData[offset + 5] = (Math.random() - 0.5) * 1;
      // Color
      initialData[offset + 6] = Math.random();
      initialData[offset + 7] = Math.random();
      initialData[offset + 8] = Math.random();
      initialData[offset + 9] = 1.0;
      // Lifetime
      initialData[offset + 10] = 5.0 + Math.random() * 5.0;
    }

    this.bufferA = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'Particle Buffer A',
    });

    this.bufferB = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'Particle Buffer B',
    });

    this.device.queue.writeBuffer(this.bufferA, 0, initialData);
    this.device.queue.writeBuffer(this.bufferB, 0, initialData);
  }

  private createComputePipeline() {
    const ParticleBuffer = arrayOf(Particle, this.particleCount);

    const SimParams = struct({
      deltaTime: f32,
      time: f32,
    });

    const paramsBuffer = this.root.createBuffer(SimParams).$usage('uniform');

    this.computePipeline = this.root
      .makeComputePipeline(
        tgpu
          .fn([ParticleBuffer, ParticleBuffer, SimParams])
          .does(`
            const idx = workgroupId.x * 256 + localInvocationId.x;
            if (idx >= ${this.particleCount}) return;

            let p = particlesIn[idx];

            // Simple gravity and damping
            p.velocity.y -= 0.5 * params.deltaTime;
            p.velocity *= 0.99;
            p.position += p.velocity * params.deltaTime;

            // Bounds check - wrap around
            if (abs(p.position.x) > 10.0) p.position.x *= -1.0;
            if (abs(p.position.y) > 10.0) p.position.y *= -1.0;
            if (abs(p.position.z) > 10.0) p.position.z *= -1.0;

            particlesOut[idx] = p;
          `)
          .$uses({
            particlesIn: this.bufferA,
            particlesOut: this.bufferB,
            params: paramsBuffer,
          })
      )
      .$workgroupSize(256);
  }

  private createParticleRendering() {
    const geometry = new BufferGeometry();

    // We'll update these from GPU buffer each frame
    const positions = new Float32Array(this.particleCount * 3);
    const colors = new Float32Array(this.particleCount * 3);

    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));

    const material = new ShaderMaterial({
      vertexShader: `
        attribute vec3 position;
        attribute vec3 color;
        varying vec3 vColor;
        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;

        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          gl_PointSize = 300.0 / -mvPosition.z;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;

        void main() {
          vec2 center = gl_PointCoord - vec2(0.5);
          float dist = length(center);
          if (dist > 0.5) discard;

          float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
    });

    this.points = new Points(geometry, material);
    this.scene.add(this.points);
  }

  private updateParticleGeometry() {
    // Read data from current output buffer
    // In production, use more efficient GPU-to-GPU transfer
    const outputBuffer = this.currentBuffer === 0 ? this.bufferB : this.bufferA;

    // Create staging buffer for readback
    const stagingBuffer = this.device.createBuffer({
      size: outputBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputBuffer.size);
    this.device.queue.submit([encoder.finish()]);

    // Map and read data
    stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(stagingBuffer.getMappedRange());

      const positions = this.points.geometry.attributes.position.array as Float32Array;
      const colors = this.points.geometry.attributes.color.array as Float32Array;

      for (let i = 0; i < this.particleCount; i++) {
        const offset = i * 13;
        positions[i * 3 + 0] = data[offset + 0];
        positions[i * 3 + 1] = data[offset + 1];
        positions[i * 3 + 2] = data[offset + 2];
        colors[i * 3 + 0] = data[offset + 6];
        colors[i * 3 + 1] = data[offset + 7];
        colors[i * 3 + 2] = data[offset + 8];
      }

      this.points.geometry.attributes.position.needsUpdate = true;
      this.points.geometry.attributes.color.needsUpdate = true;

      stagingBuffer.unmap();
      stagingBuffer.destroy();
    });
  }

  update(deltaTime: number, time: number) {
    // Run compute shader
    const inputBuffer = this.currentBuffer === 0 ? this.bufferA : this.bufferB;
    const outputBuffer = this.currentBuffer === 0 ? this.bufferB : this.bufferA;

    this.root
      .createCommandEncoder()
      .beginComputePass()
      .setPipeline(this.computePipeline)
      .dispatchWorkgroups(Math.ceil(this.particleCount / 256))
      .end()
      .submit();

    this.currentBuffer = 1 - this.currentBuffer;

    // Update geometry (async)
    this.updateParticleGeometry();

    // Update controls
    this.controls.update();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  private onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  dispose() {
    this.bufferA.destroy();
    this.bufferB.destroy();
    this.points.geometry.dispose();
    (this.points.material as ShaderMaterial).dispose();
    this.renderer.dispose();
  }
}

// Usage
async function main() {
  const particleSystem = new GPUParticleSystem(50000);
  await particleSystem.init();

  let lastTime = performance.now();

  function animate() {
    const now = performance.now();
    const deltaTime = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    particleSystem.update(deltaTime, now / 1000);
    particleSystem.render();

    requestAnimationFrame(animate);
  }

  animate();
}

main().catch(console.error);
```

This example demonstrates a complete integration with proper resource management, double buffering, and efficient rendering.

## Babylon.js Integration

### Babylon.js WebGPU Engine

Babylon.js provides comprehensive WebGPU support through its WebGPUEngine class, offering a powerful alternative to the traditional WebGL engine with improved performance and modern features.

**Engine Initialization**:

```typescript
import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder } from '@babylonjs/core';

async function initBabylonWebGPU() {
  const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;

  // Create WebGPU engine
  const engine = new Engine(canvas, true);
  await engine.initAsync(); // Initialize WebGPU support

  // Ensure we're using WebGPU
  if (!engine.isWebGPU) {
    throw new Error('WebGPU not available, falling back to WebGL');
  }

  console.log('WebGPU engine initialized successfully');

  // Create scene
  const scene = new Scene(engine);

  // Setup camera
  const camera = new ArcRotateCamera(
    'camera',
    -Math.PI / 2,
    Math.PI / 2.5,
    10,
    Vector3.Zero(),
    scene
  );
  camera.attachControl(canvas, true);

  // Add light
  const light = new HemisphericLight('light', new Vector3(0, 1, 0), scene);
  light.intensity = 0.7;

  // Create simple geometry
  const sphere = MeshBuilder.CreateSphere('sphere', { diameter: 2 }, scene);

  // Run render loop
  engine.runRenderLoop(() => {
    scene.render();
  });

  // Handle resize
  window.addEventListener('resize', () => {
    engine.resize();
  });

  return { engine, scene, camera };
}
```

**Accessing the GPUDevice**: Babylon.js's WebGPU engine provides access to the underlying device:

```typescript
async function getWebGPUDevice() {
  const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
  const engine = new Engine(canvas, true);
  await engine.initAsync();

  // Access the WebGPU device through engine internals
  const device = (engine as any)._device as GPUDevice;

  if (!device) {
    throw new Error('WebGPU device not available');
  }

  console.log('Device limits:', device.limits);
  console.log('Device features:', device.features);

  return device;
}
```

**Capabilities**: The Babylon.js WebGPU engine supports:
- Advanced PBR materials with full physically-based rendering
- Compute shaders through the ComputeShader class
- Multiple render targets and complex post-processing
- Shadow mapping, screen-space reflections, and ambient occlusion
- Particle systems with GPU acceleration
- WGSL shader support through the ShaderMaterial system

### TypeGPU Compute in Babylon Scenes

Integrating TypeGPU compute shaders with Babylon.js scenes enables custom simulation and data processing:

```typescript
import { Engine, Scene, MeshBuilder, StandardMaterial, Color3 } from '@babylonjs/core';
import tgpu from 'typegpu';
import { struct, vec3f, f32, arrayOf } from 'typegpu/data';

// Define mesh vertex structure
const MeshVertex = struct({
  position: vec3f,
  normal: vec3f,
  displacement: f32,
});

async function proceduralMeshDeformation() {
  // Initialize Babylon.js
  const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
  const engine = new Engine(canvas, true);
  await engine.initAsync();

  const scene = new Scene(engine);

  // Setup camera
  const camera = new ArcRotateCamera('camera', 0, Math.PI / 3, 10, Vector3.Zero(), scene);
  camera.attachControl(canvas, true);

  // Get WebGPU device
  const device = (engine as any)._device as GPUDevice;
  const root = tgpu.initFromDevice(device);

  // Create mesh
  const subdivisions = 128;
  const mesh = MeshBuilder.CreateGround(
    'ground',
    { width: 10, height: 10, subdivisions },
    scene
  );

  const material = new StandardMaterial('material', scene);
  material.wireframe = true;
  material.diffuseColor = new Color3(0.2, 0.8, 0.4);
  mesh.material = material;

  // Get mesh vertex data
  const positions = mesh.getVerticesData('position');
  const normals = mesh.getVerticesData('normal');
  const vertexCount = positions.length / 3;

  // Create GPU buffer for vertex data
  const vertexData = new Float32Array(vertexCount * 7); // 3 pos + 3 normal + 1 displacement

  for (let i = 0; i < vertexCount; i++) {
    vertexData[i * 7 + 0] = positions[i * 3 + 0];
    vertexData[i * 7 + 1] = positions[i * 3 + 1];
    vertexData[i * 7 + 2] = positions[i * 3 + 2];
    vertexData[i * 7 + 3] = normals[i * 3 + 0];
    vertexData[i * 7 + 4] = normals[i * 3 + 1];
    vertexData[i * 7 + 5] = normals[i * 3 + 2];
    vertexData[i * 7 + 6] = 0.0; // Initial displacement
  }

  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'Vertex Buffer',
  });

  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // Create compute shader for wave deformation
  const SimParams = struct({
    time: f32,
    amplitude: f32,
    frequency: f32,
    vertexCount: f32,
  });

  const paramsBuffer = root
    .createBuffer(SimParams, {
      time: 0,
      amplitude: 0.5,
      frequency: 2.0,
      vertexCount: vertexCount,
    })
    .$usage('uniform');

  const deformMesh = tgpu
    .fn([arrayOf(MeshVertex), SimParams])
    .does(`
      const idx = workgroupId.x * 64 + localInvocationId.x;
      if (idx >= u32(params.vertexCount)) return;

      let vertex = vertices[idx];

      // Calculate wave displacement
      let wave = sin(vertex.position.x * params.frequency + params.time) *
                 cos(vertex.position.z * params.frequency + params.time);

      vertex.displacement = wave * params.amplitude;

      vertices[idx] = vertex;
    `)
    .$uses({ vertices: vertexBuffer, params: paramsBuffer })
    .$name('deformMesh');

  const computePipeline = root
    .makeComputePipeline(deformMesh)
    .$workgroupSize(64);

  // Animation loop
  let time = 0;
  engine.runRenderLoop(() => {
    time += 0.016;

    // Update parameters
    paramsBuffer.write({ time });

    // Run compute shader
    root
      .createCommandEncoder()
      .beginComputePass()
      .setPipeline(computePipeline)
      .dispatchWorkgroups(Math.ceil(vertexCount / 64))
      .end()
      .submit();

    // Read back data and update Babylon.js mesh
    // Note: In production, this should be optimized with async readback
    const stagingBuffer = device.createBuffer({
      size: vertexBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(vertexBuffer, 0, stagingBuffer, 0, vertexBuffer.size);
    device.queue.submit([encoder.finish()]);

    stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(stagingBuffer.getMappedRange());

      // Update Babylon.js mesh positions
      for (let i = 0; i < vertexCount; i++) {
        const offset = i * 7;
        const displacement = data[offset + 6];
        positions[i * 3 + 1] = vertexData[offset + 1] + displacement;
      }

      mesh.updateVerticesData('position', positions);
      stagingBuffer.unmap();
      stagingBuffer.destroy();
    });

    scene.render();
  });

  return { engine, scene, mesh };
}
```

This example shows how to:
- Initialize Babylon.js with WebGPU
- Share the GPUDevice with TypeGPU
- Run compute shaders to modify mesh geometry
- Update Babylon.js meshes with computed results

### Custom Shader Materials

Babylon.js supports custom shaders through ShaderMaterial and PBRCustomMaterial. Here's how to integrate TGSL:

```typescript
import { ShaderMaterial, Effect } from '@babylonjs/core';
import tgpu from 'typegpu';

function createTGSLMaterial(scene: Scene) {
  const device = (scene.getEngine() as any)._device as GPUDevice;
  const root = tgpu.initFromDevice(device);

  // Define shader in TGSL
  const vertexShader = tgpu
    .vertexFn([
      tgpu.param('position', tgpu.vec3f, { location: 0 }),
      tgpu.param('normal', tgpu.vec3f, { location: 1 }),
      tgpu.param('uv', tgpu.vec2f, { location: 2 }),
    ])
    .outputs({
      position: tgpu.builtin.position,
      vNormal: tgpu.vec3f,
      vUV: tgpu.vec2f,
    })
    .implement((position, normal, uv) => {
      // Simple vertex transformation
      return {
        position: tgpu.vec4f(position, 1.0),
        vNormal: normal,
        vUV: uv,
      };
    });

  const fragmentShader = tgpu
    .fragmentFn([
      tgpu.param('vNormal', tgpu.vec3f),
      tgpu.param('vUV', tgpu.vec2f),
    ])
    .outputs({
      color: tgpu.vec4f,
    })
    .implement((vNormal, vUV) => {
      // Procedural coloring based on normal
      const color = vNormal * 0.5 + 0.5;
      return {
        color: tgpu.vec4f(color, 1.0),
      };
    });

  // Convert to WGSL
  const vertexWGSL = vertexShader.toWGSL();
  const fragmentWGSL = fragmentShader.toWGSL();

  // Register shader with Babylon.js
  Effect.ShadersStore['customVertexShader'] = vertexWGSL;
  Effect.ShadersStore['customFragmentShader'] = fragmentWGSL;

  // Create material
  const material = new ShaderMaterial(
    'customMaterial',
    scene,
    {
      vertex: 'custom',
      fragment: 'custom',
    },
    {
      attributes: ['position', 'normal', 'uv'],
      uniforms: ['worldViewProjection'],
    }
  );

  return material;
}
```

### Resource Sharing

Efficient resource sharing between TypeGPU and Babylon.js:

```typescript
async function sharedResourceExample() {
  const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
  const engine = new Engine(canvas, true);
  await engine.initAsync();

  const scene = new Scene(engine);
  const device = (engine as any)._device as GPUDevice;
  const root = tgpu.initFromDevice(device);

  // Create shared buffer for particle data
  const particleCount = 10000;
  const bufferSize = particleCount * 16 * 4; // vec4 * 4 floats

  const sharedBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE |
           GPUBufferUsage.VERTEX |
           GPUBufferUsage.COPY_DST |
           GPUBufferUsage.COPY_SRC,
    label: 'Shared Particle Buffer',
  });

  // Use in TypeGPU compute
  const computeShader = tgpu
    .fn([arrayOf(tgpu.vec4f, particleCount)])
    .does(`
      const idx = workgroupId.x * 64 + localInvocationId.x;
      if (idx >= ${particleCount}) return;

      // Modify particle data
      particles[idx].y += 0.01;
    `)
    .$uses({ particles: sharedBuffer })
    .$name('updateParticles');

  // Use in Babylon.js rendering
  // (Babylon.js can consume the buffer through its vertex buffer system)

  return { engine, scene, sharedBuffer };
}
```

### Complete Example: Procedural Terrain with Compute Shader

A full example demonstrating compute-based terrain generation:

```typescript
import {
  Engine,
  Scene,
  ArcRotateCamera,
  Vector3,
  HemisphericLight,
  Mesh,
  VertexData,
  StandardMaterial,
  Color3,
} from '@babylonjs/core';
import tgpu from 'typegpu';
import { struct, vec3f, vec2f, f32, arrayOf } from 'typegpu/data';

const TerrainVertex = struct({
  position: vec3f,
  normal: vec3f,
  uv: vec2f,
  height: f32,
});

const TerrainParams = struct({
  seed: f32,
  scale: f32,
  octaves: f32,
  persistence: f32,
  lacunarity: f32,
  time: f32,
});

class ProceduralTerrain {
  private engine: Engine;
  private scene: Scene;
  private device: GPUDevice;
  private root: any;
  private mesh: Mesh;
  private resolution: number;
  private vertexBuffer: GPUBuffer;
  private computePipeline: any;

  constructor(resolution: number = 256) {
    this.resolution = resolution;
  }

  async init() {
    // Initialize Babylon.js
    const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
    this.engine = new Engine(canvas, true);
    await this.engine.initAsync();

    this.scene = new Scene(this.engine);

    // Setup camera
    const camera = new ArcRotateCamera(
      'camera',
      -Math.PI / 2,
      Math.PI / 3,
      50,
      Vector3.Zero(),
      this.scene
    );
    camera.attachControl(canvas, true);

    // Add light
    const light = new HemisphericLight('light', new Vector3(0, 1, 0), this.scene);

    // Get WebGPU device
    this.device = (this.engine as any)._device as GPUDevice;
    this.root = tgpu.initFromDevice(this.device);

    // Create terrain mesh
    await this.createTerrainMesh();

    // Create compute pipeline
    this.createComputePipeline();

    // Generate initial terrain
    await this.generateTerrain();

    // Run render loop
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });

    window.addEventListener('resize', () => this.engine.resize());

    return this;
  }

  private async createTerrainMesh() {
    const vertexCount = this.resolution * this.resolution;
    const indexCount = (this.resolution - 1) * (this.resolution - 1) * 6;

    // Create vertex data
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const indices = new Uint32Array(indexCount);

    // Initialize grid
    for (let z = 0; z < this.resolution; z++) {
      for (let x = 0; x < this.resolution; x++) {
        const i = z * this.resolution + x;

        positions[i * 3 + 0] = (x / (this.resolution - 1) - 0.5) * 20;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = (z / (this.resolution - 1) - 0.5) * 20;

        normals[i * 3 + 0] = 0;
        normals[i * 3 + 1] = 1;
        normals[i * 3 + 2] = 0;

        uvs[i * 2 + 0] = x / (this.resolution - 1);
        uvs[i * 2 + 1] = z / (this.resolution - 1);
      }
    }

    // Create indices
    let idx = 0;
    for (let z = 0; z < this.resolution - 1; z++) {
      for (let x = 0; x < this.resolution - 1; x++) {
        const topLeft = z * this.resolution + x;
        const topRight = topLeft + 1;
        const bottomLeft = (z + 1) * this.resolution + x;
        const bottomRight = bottomLeft + 1;

        indices[idx++] = topLeft;
        indices[idx++] = bottomLeft;
        indices[idx++] = topRight;

        indices[idx++] = topRight;
        indices[idx++] = bottomLeft;
        indices[idx++] = bottomRight;
      }
    }

    // Create Babylon.js mesh
    this.mesh = new Mesh('terrain', this.scene);

    const vertexData = new VertexData();
    vertexData.positions = positions;
    vertexData.normals = normals;
    vertexData.uvs = uvs;
    vertexData.indices = indices;

    vertexData.applyToMesh(this.mesh);

    // Create material
    const material = new StandardMaterial('terrainMaterial', this.scene);
    material.diffuseColor = new Color3(0.3, 0.6, 0.3);
    material.specularColor = new Color3(0.1, 0.1, 0.1);
    material.wireframe = false;
    this.mesh.material = material;

    // Create GPU buffer for vertex positions
    const vertexBufferData = new Float32Array(vertexCount * 10); // Extended format
    for (let i = 0; i < vertexCount; i++) {
      vertexBufferData[i * 10 + 0] = positions[i * 3 + 0];
      vertexBufferData[i * 10 + 1] = positions[i * 3 + 1];
      vertexBufferData[i * 10 + 2] = positions[i * 3 + 2];
      vertexBufferData[i * 10 + 3] = normals[i * 3 + 0];
      vertexBufferData[i * 10 + 4] = normals[i * 3 + 1];
      vertexBufferData[i * 10 + 5] = normals[i * 3 + 2];
      vertexBufferData[i * 10 + 6] = uvs[i * 2 + 0];
      vertexBufferData[i * 10 + 7] = uvs[i * 2 + 1];
      vertexBufferData[i * 10 + 8] = 0; // height
      vertexBufferData[i * 10 + 9] = 0; // padding
    }

    this.vertexBuffer = this.device.createBuffer({
      size: vertexBufferData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'Terrain Vertex Buffer',
    });

    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertexBufferData);
  }

  private createComputePipeline() {
    const vertexCount = this.resolution * this.resolution;

    const paramsBuffer = this.root
      .createBuffer(TerrainParams, {
        seed: Math.random() * 1000,
        scale: 10.0,
        octaves: 4.0,
        persistence: 0.5,
        lacunarity: 2.0,
        time: 0,
      })
      .$usage('uniform');

    const generateTerrain = tgpu
      .fn([arrayOf(TerrainVertex), TerrainParams])
      .does(`
        const idx = workgroupId.x * 64 + localInvocationId.x;
        if (idx >= ${vertexCount}) return;

        let vertex = vertices[idx];

        // Multi-octave noise (simplified Perlin-like noise)
        let amplitude = 1.0;
        let frequency = 1.0;
        let height = 0.0;

        for (var i = 0; i < i32(params.octaves); i++) {
          let sampleX = vertex.uv.x * frequency * params.scale + params.seed;
          let sampleZ = vertex.uv.y * frequency * params.scale + params.seed;

          // Simple noise function using sin
          let noise = sin(sampleX * 1.23) * cos(sampleZ * 2.34) +
                     sin(sampleX * 2.34 + params.time * 0.1) * cos(sampleZ * 1.23);

          height += noise * amplitude;

          amplitude *= params.persistence;
          frequency *= params.lacunarity;
        }

        vertex.height = height * 2.0;
        vertex.position.y = vertex.height;

        vertices[idx] = vertex;
      `)
      .$uses({ vertices: this.vertexBuffer, params: paramsBuffer })
      .$name('generateTerrain');

    this.computePipeline = this.root
      .makeComputePipeline(generateTerrain)
      .$workgroupSize(64);
  }

  async generateTerrain() {
    const vertexCount = this.resolution * this.resolution;

    // Execute compute shader
    this.root
      .createCommandEncoder()
      .beginComputePass()
      .setPipeline(this.computePipeline)
      .dispatchWorkgroups(Math.ceil(vertexCount / 64))
      .end()
      .submit();

    // Read back and update Babylon.js mesh
    const stagingBuffer = this.device.createBuffer({
      size: this.vertexBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.vertexBuffer, 0, stagingBuffer, 0, this.vertexBuffer.size);
    this.device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());

    const positions = this.mesh.getVerticesData('position');

    for (let i = 0; i < vertexCount; i++) {
      const offset = i * 10;
      positions[i * 3 + 1] = data[offset + 8]; // height
    }

    this.mesh.updateVerticesData('position', positions);
    this.mesh.createNormals(true); // Recalculate normals

    stagingBuffer.unmap();
    stagingBuffer.destroy();
  }

  dispose() {
    this.vertexBuffer.destroy();
    this.mesh.dispose();
    this.scene.dispose();
    this.engine.dispose();
  }
}

// Usage
async function main() {
  const terrain = new ProceduralTerrain(256);
  await terrain.init();

  // Regenerate terrain every 5 seconds
  setInterval(() => {
    terrain.generateTerrain();
  }, 5000);
}

main().catch(console.error);
```

This complete example demonstrates advanced integration patterns including procedural generation, compute-based mesh modification, and efficient resource sharing between TypeGPU and Babylon.js.

## Resource Sharing Strategies

### Device Sharing

The foundation of integration is sharing a single GPUDevice instance across libraries. This avoids resource conflicts and enables efficient interoperability.

**Single GPUDevice Pattern**:

```typescript
// Initialize framework first
const renderer = new WebGPURenderer(); // Three.js
// OR
const engine = new Engine(canvas, true); // Babylon.js

await renderer.init(); // or engine.initAsync()

// Get the device
const device = renderer.backend.device; // Three.js
// OR
const device = (engine as any)._device; // Babylon.js

// Initialize TypeGPU with existing device
const root = tgpu.initFromDevice(device);
```

**Initialization Order**: Always initialize the framework first, then wrap its device with TypeGPU. This ensures the framework properly configures the device with required features and limits.

**Lifecycle Management**: The framework owns the device and manages its lifecycle. When disposing resources:

```typescript
// Dispose TypeGPU resources first
root.destroy();

// Then dispose framework resources
renderer.dispose(); // Three.js
// OR
engine.dispose(); // Babylon.js
```

### Buffer Interoperability

Buffers can be shared between compute and render pipelines if created with compatible usage flags:

```typescript
// Create buffer usable by both TypeGPU compute and framework rendering
const sharedBuffer = device.createBuffer({
  size: dataSize,
  usage: GPUBufferUsage.STORAGE |      // TypeGPU compute read/write
         GPUBufferUsage.VERTEX |        // Framework vertex data
         GPUBufferUsage.COPY_DST |      // Can write from CPU
         GPUBufferUsage.COPY_SRC |      // Can read to CPU
         GPUBufferUsage.INDEX,          // Optional: if used as index buffer
  label: 'Shared Compute/Render Buffer',
});
```

**Compatible Buffer Usage Flags**:
- `STORAGE`: Required for compute shader read/write access
- `VERTEX`: Required for use as vertex buffer in rendering
- `INDEX`: Required for use as index buffer
- `UNIFORM`: For read-only constant data
- `COPY_DST`: Allows writing data from CPU with `writeBuffer()`
- `COPY_SRC`: Allows reading data back to CPU

**Data Format Matching**: Ensure data layouts match between TypeGPU schemas and framework expectations:

```typescript
// TypeGPU schema
const Vertex = struct({
  position: vec3f,  // 12 bytes
  normal: vec3f,    // 12 bytes
  uv: vec2f,        // 8 bytes
  // Total: 32 bytes per vertex
});

// Three.js expects:
// position: 3 floats (12 bytes)
// normal: 3 floats (12 bytes)
// uv: 2 floats (8 bytes)
// Matches perfectly!
```

Pay attention to alignment requirements. GPU buffers require 16-byte alignment for some operations.

### Texture Interoperability

Textures can be shared similarly to buffers:

```typescript
const sharedTexture = device.createTexture({
  size: { width: 1024, height: 1024 },
  format: 'rgba8unorm',
  usage: GPUTextureUsage.STORAGE_BINDING |    // Compute shader write
         GPUTextureUsage.TEXTURE_BINDING |    // Shader sampling
         GPUTextureUsage.RENDER_ATTACHMENT |  // Render target
         GPUTextureUsage.COPY_SRC |           // Can copy from
         GPUTextureUsage.COPY_DST,            // Can copy to
  label: 'Shared Compute/Render Texture',
});
```

**Format Considerations**:

Common compatible formats:
- `rgba8unorm`: Standard 8-bit RGBA (most compatible)
- `rgba16float`: HDR rendering
- `rgba32float`: High precision compute
- `depth24plus`: Depth textures
- `depth32float`: High precision depth

Match formats between TypeGPU and framework usage:

```typescript
// TypeGPU compute writes rgba8unorm
const computeTexture = root.wrapTexture(sharedTexture, {
  format: 'rgba8unorm',
  dimension: '2d',
});

// Three.js uses as regular texture (automatically handles format)
const threeTexture = renderer.backend.createTextureFromSource(sharedTexture);
```

**View Creation**: Different usage scenarios may require different texture views:

```typescript
// Storage view for compute shader writes
const storageView = sharedTexture.createView({
  format: 'rgba8unorm',
  dimension: '2d',
  aspect: 'all',
});

// Sampling view for fragment shader reads
const samplingView = sharedTexture.createView({
  format: 'rgba8unorm',
  dimension: '2d',
  aspect: 'all',
  baseMipLevel: 0,
  mipLevelCount: 1,
});
```

### Synchronization Considerations

Proper command ordering ensures compute operations complete before rendering uses results:

**Sequential Execution Pattern**:

```typescript
function frame() {
  // 1. Run TypeGPU compute
  root
    .createCommandEncoder()
    .beginComputePass()
    .setPipeline(computePipeline)
    .dispatchWorkgroups(workgroupCount)
    .end()
    .submit();

  // 2. Then render with framework
  // Framework automatically waits for compute to complete
  // before using shared resources
  renderer.render(scene, camera); // Three.js
  // OR
  scene.render(); // Babylon.js
}
```

WebGPU handles synchronization automatically when commands are submitted sequentially to the same queue. Shared resources are synchronized through automatic barriers.

**Advanced: Async Compute Queues**:

For maximum performance, use separate compute queues:

```typescript
// Get queue for async compute (if supported)
const computeQueue = device.queue; // Default queue
// Note: WebGPU currently only exposes one queue

// Future-proof pattern for when multiple queues are available:
function frameAdvanced() {
  // Submit compute work
  const computeCommands = createComputeCommands();
  computeQueue.submit([computeCommands]);

  // Submit render work (automatically synchronized)
  const renderCommands = createRenderCommands();
  device.queue.submit([renderCommands]);
}
```

**Queue Submission Order**:

```typescript
// CORRECT: Compute before render
const encoder1 = device.createCommandEncoder();
// ... record compute pass ...
device.queue.submit([encoder1.finish()]);

const encoder2 = device.createCommandEncoder();
// ... record render pass ...
device.queue.submit([encoder2.finish()]);

// INCORRECT: May cause race conditions
// (submitting in wrong order or simultaneously)
```

## Use Cases

### Physics Simulations

GPU physics with TypeGPU compute, rendered through frameworks:

```typescript
class GPUPhysicsEngine {
  private root: any;
  private bodyBuffer: GPUBuffer;
  private physicsShader: any;

  async init(device: GPUDevice) {
    this.root = tgpu.initFromDevice(device);

    const RigidBody = struct({
      position: vec3f,
      velocity: vec3f,
      force: vec3f,
      mass: f32,
      radius: f32,
    });

    // Create physics simulation shader
    this.physicsShader = tgpu
      .fn([arrayOf(RigidBody), SimParams])
      .does(`
        const idx = workgroupId.x * 64 + localInvocationId.x;
        if (idx >= params.bodyCount) return;

        let body = bodies[idx];

        // Apply forces (gravity, collisions, etc.)
        body.force += vec3f(0.0, -9.8 * body.mass, 0.0);

        // Integrate velocity
        body.velocity += (body.force / body.mass) * params.deltaTime;
        body.force = vec3f(0.0, 0.0, 0.0);

        // Integrate position
        body.position += body.velocity * params.deltaTime;

        // Ground collision
        if (body.position.y - body.radius < 0.0) {
          body.position.y = body.radius;
          body.velocity.y = -body.velocity.y * 0.8;
        }

        bodies[idx] = body;
      `)
      .$uses({ bodies: this.bodyBuffer, params: paramsBuffer })
      .$name('physicsStep');
  }

  update(deltaTime: number) {
    // Run physics simulation
    // Update framework objects with results
  }
}
```

Use cases:
- Rigid body dynamics for games
- Soft body simulation (cloth, jelly)
- Fluid dynamics (SPH, grid-based)
- Destruction and fracturing

### Procedural Generation

Compute-based generation with real-time framework updates:

```typescript
class ProceduralGenerator {
  async generateCityMesh() {
    // Compute shader generates building data
    const generateBuildings = tgpu
      .fn([arrayOf(Building), GridParams])
      .does(`
        // Generate building positions, heights, types
        // Based on noise functions and rules
      `);

    // Execute compute
    this.runCompute(generateBuildings);

    // Read results and create Three.js/Babylon.js meshes
    const buildings = await this.readBuildingData();
    buildings.forEach(building => {
      const mesh = createBuildingMesh(building);
      scene.add(mesh);
    });
  }
}
```

Applications:
- Terrain generation with LOD
- City/dungeon generation
- Vegetation placement
- Texture synthesis

### Post-Processing Effects

Custom effects pipelines with type-safe authoring:

```typescript
class CustomPostProcessing {
  private edgeDetectShader: any;
  private blurShader: any;

  createEffectPipeline(device: GPUDevice) {
    const root = tgpu.initFromDevice(device);

    // Edge detection with TGSL
    this.edgeDetectShader = tgpu
      .fragmentFn([sampler, colorTexture, depthTexture])
      .does(`
        // Sobel edge detection
        let edges = detectEdges(uv, colorTexture);
        return vec4f(edges, edges, edges, 1.0);
      `);

    // Chain with framework's post-processing
    // Apply to rendered scene
  }
}
```

Effects examples:
- Custom bloom/glow
- Advanced anti-aliasing
- Screen-space ambient occlusion
- Temporal effects (motion blur, TAA)

### Data Visualization

Large dataset processing with framework rendering:

```typescript
class ScientificVisualization {
  async visualizeDataset(data: Float32Array) {
    // Compute shader processes millions of data points
    const processData = tgpu
      .fn([arrayOf(DataPoint)])
      .does(`
        // Statistical analysis
        // Clustering
        // Dimensionality reduction
      `);

    // Generate visualization geometry
    const visualizationData = await this.computeVisualization(data);

    // Render with Three.js/Babylon.js
    // (scatter plots, volume rendering, etc.)
  }
}
```

Applications:
- Scientific data visualization
- Financial market analysis
- Medical imaging (MRI, CT scans)
- Network graphs

## Performance Considerations

### Overhead of Mixing Libraries

Combining TypeGPU with frameworks introduces some overhead:

**CPU-GPU Synchronization**: Reading data from GPU to CPU to update framework objects is expensive. Minimize with:
- Direct GPU-to-GPU buffer sharing
- Async readback with double buffering
- Batch updates instead of per-frame transfers

**Command Encoder Overhead**: Creating multiple command encoders per frame has minor overhead. Optimize by:
- Batching compute dispatches
- Reusing bind groups
- Minimizing encoder creation

**Memory Overhead**: Duplicate buffers for double buffering increase memory usage. Acceptable for real-time performance but watch total memory consumption.

### Optimization Strategies

**1. Minimize CPU-GPU Transfers**:

```typescript
// BAD: Read every frame
function badFrame() {
  runCompute();
  const data = await readBuffer(); // Expensive!
  updateFramework(data);
  render();
}

// GOOD: Update framework objects via shared buffers
function goodFrame() {
  runCompute();
  render(); // Framework uses shared buffer directly
}
```

**2. Async Readback Pattern**:

```typescript
let readbackInProgress = false;
let pendingReadback: Promise<ArrayBuffer> | null = null;

function frameWithAsyncReadback() {
  // Start new readback if none in progress
  if (!readbackInProgress) {
    readbackInProgress = true;
    pendingReadback = readBufferAsync().then(data => {
      updateFrameworkAsync(data);
      readbackInProgress = false;
    });
  }

  // Continue rendering with previous data
  runCompute();
  render();
}
```

**3. Workgroup Size Tuning**:

```typescript
// Test different workgroup sizes
const sizes = [32, 64, 128, 256];
sizes.forEach(size => {
  const pipeline = root
    .makeComputePipeline(shader)
    .$workgroupSize(size);

  const time = benchmark(pipeline);
  console.log(`Size ${size}: ${time}ms`);
});
```

**4. Resource Pooling**:

```typescript
class BufferPool {
  private pool: GPUBuffer[] = [];

  acquire(size: number): GPUBuffer {
    return this.pool.pop() || this.device.createBuffer({
      size,
      usage: this.usage,
    });
  }

  release(buffer: GPUBuffer) {
    this.pool.push(buffer);
  }
}
```

### When to Use TypeGPU vs Native

**Use TypeGPU when**:
- Type safety is critical (complex shaders, large teams)
- Sharing code between CPU and GPU
- Building reusable compute kernels
- Prototyping and experimentation

**Use Native WebGPU/Framework APIs when**:
- Maximum performance is essential
- Simple, one-off compute operations
- Leveraging framework-specific optimizations
- Debugging low-level GPU issues

**Hybrid Approach**:
Most production applications benefit from mixing:
- Critical path: Native/framework APIs
- Custom compute: TypeGPU for type safety
- Utilities: TypeGPU for code reuse

## Best Practices

### Clear Ownership

Establish clear ownership of GPU resources:

```typescript
class ResourceManager {
  // Framework owns these
  private frameworkBuffers: Set<GPUBuffer> = new Set();

  // TypeGPU owns these
  private computeBuffers: Set<GPUBuffer> = new Set();

  // Shared resources (neither can destroy)
  private sharedBuffers: Set<GPUBuffer> = new Set();

  createSharedBuffer(size: number): GPUBuffer {
    const buffer = device.createBuffer({ size, usage });
    this.sharedBuffers.add(buffer);
    return buffer;
  }

  dispose() {
    // Only destroy owned resources
    this.computeBuffers.forEach(b => b.destroy());
    // Don't destroy framework or shared resources
  }
}
```

### Consistent Conventions

Adopt naming conventions for clarity:

```typescript
// Prefix shared resources
const shared_particleBuffer = device.createBuffer({...});
const compute_tempBuffer = device.createBuffer({...});
const render_vertexBuffer = device.createBuffer({...});

// Document usage in labels
const buffer = device.createBuffer({
  size,
  usage,
  label: 'Shared: Particle positions (compute write, vertex read)',
});
```

### Documentation

Document integration points:

```typescript
/**
 * Particle System Integration
 *
 * Architecture:
 * - TypeGPU: Physics simulation (compute shaders)
 * - Three.js: Rendering (points with custom shader)
 *
 * Shared Resources:
 * - particleBuffer: Particle positions and colors
 *   - Compute: Write particle data
 *   - Render: Read as vertex buffer
 *
 * Synchronization:
 * - Compute runs before render each frame
 * - No CPU-GPU transfer (direct buffer sharing)
 *
 * Performance:
 * - 50K particles at 60fps on mid-range GPU
 * - Scales to 200K on high-end hardware
 */
class IntegratedParticleSystem {
  // ...
}
```

## Common Pitfalls

### Resource Ownership Conflicts

**Problem**: Both libraries try to destroy shared resources.

```typescript
// BAD
root.destroy(); // Destroys device
renderer.dispose(); // Tries to use destroyed device - crash!
```

**Solution**: Framework owns device, TypeGPU just wraps it:

```typescript
// GOOD
root.destroy(); // Only destroys TypeGPU-created resources
renderer.dispose(); // Destroys device and framework resources
```

### Format Incompatibilities

**Problem**: TypeGPU and framework expect different data formats.

```typescript
// TypeGPU writes vec4f (16 bytes)
// Three.js expects vec3f (12 bytes)
// Misalignment causes corruption
```

**Solution**: Align data structures:

```typescript
// Add padding to match layouts
const Particle = struct({
  position: vec3f,  // 12 bytes
  _pad1: f32,       // 4 bytes padding
  velocity: vec3f,  // 12 bytes
  _pad2: f32,       // 4 bytes padding
  // Now 16-byte aligned
});
```

### Synchronization Issues

**Problem**: Reading compute results before compute finishes.

```typescript
// BAD
runCompute();
const data = readBufferImmediate(); // May read old data!
```

**Solution**: Proper synchronization:

```typescript
// GOOD
runCompute();
device.queue.onSubmittedWorkDone().then(() => {
  const data = readBuffer(); // Guaranteed fresh data
});
```

### Buffer Usage Flag Errors

**Problem**: Missing usage flags for intended operations.

```typescript
// BAD: Created with only STORAGE
const buffer = device.createBuffer({
  size,
  usage: GPUBufferUsage.STORAGE,
});

// Later: Try to use as vertex buffer - error!
setVertexBuffer(buffer);
```

**Solution**: Include all required flags:

```typescript
// GOOD
const buffer = device.createBuffer({
  size,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
});
```

## Conclusion

Integrating TypeGPU with Three.js and Babylon.js unlocks powerful hybrid architectures combining type-safe GPU compute with mature rendering frameworks. By sharing GPUDevice instances, managing resources carefully, and following best practices outlined in this guide, you can build sophisticated applications that leverage the strengths of each library.

Key takeaways:

- **Device Sharing**: Use `tgpu.initFromDevice()` to wrap framework devices
- **Resource Management**: Create buffers/textures with compatible usage flags
- **Synchronization**: Submit compute commands before render commands
- **Performance**: Minimize CPU-GPU transfers through direct resource sharing
- **Type Safety**: Use TGSL for maintainable, refactorable shader code
- **Best Practices**: Clear ownership, consistent conventions, thorough documentation

Whether building games, scientific visualizations, or interactive experiences, these integration patterns provide a solid foundation for WebGPU development. Start with simple examples, gradually adopt more advanced techniques, and leverage both TypeGPU's type safety and framework ecosystems for production-ready applications.

The future of WebGPU is bright, and combining modern tools like TypeGPU with established frameworks positions your projects at the cutting edge of web graphics and compute.

---

*This guide covers TypeGPU integration patterns as of 2025. Always refer to official documentation for the latest API changes and best practices.*
