# Building a Particle System with TypeGPU

## Overview

In this comprehensive tutorial, we'll build an interactive particle simulation using TypeGPU and WebGPU compute shaders. We'll create a system capable of simulating thousands of particles with physics-based behavior, interactive mouse controls, and beautiful visual effects—all accelerated by the GPU.

Particle systems are fundamental to creating visual effects like fire, smoke, explosions, and more. Traditional CPU-based particle systems struggle with large particle counts, but by leveraging GPU compute shaders, we can simulate hundreds of thousands of particles at 60fps. TypeGPU provides a type-safe TypeScript wrapper around WebGPU, making it easier to write and maintain GPU code while catching errors at compile time.

By the end of this tutorial, you'll have a fully functional particle system that responds to mouse input, implements physics simulation, handles particle lifetimes, and renders beautiful particles with alpha blending. The techniques you'll learn form the foundation for more advanced GPU-driven simulations.

## Project Setup

### Dependencies

First, let's set up our project. Create a new directory and initialize it with npm:

```bash
mkdir particle-system
cd particle-system
npm init -y
npm install typegpu
npm install --save-dev vite typescript
```

TypeGPU is the core library that provides type-safe WebGPU abstractions. We'll use Vite as our development server and bundler for its fast hot-module replacement and TypeScript support out of the box.

### Bundler Configuration (Vite Example)

Create a `vite.config.ts` file in your project root:

```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  optimizeDeps: {
    exclude: ['typegpu'],
  },
});
```

These CORS headers are important for SharedArrayBuffer support and certain WebGPU features. The `optimizeDeps.exclude` prevents Vite from pre-bundling TypeGPU, which can sometimes cause issues with WebGPU initialization.

Create a `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "lib": ["ES2022", "DOM"],
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"]
}
```

### Project Structure

Organize your project with the following structure:

```
particle-system/
├── src/
│   ├── main.ts          # Entry point and render loop
│   ├── particle.ts      # Particle data structures
│   ├── compute.ts       # Compute shader logic
│   └── render.ts        # Render pipeline setup
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

Create a simple `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particle System - TypeGPU</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #000;
    }
    canvas {
      display: block;
      width: 100vw;
      height: 100vh;
    }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <script type="module" src="/src/main.ts"></script>
</body>
</html>
```

## Data Structures

### Particle Schema

TypeGPU uses a schema-based approach to define data structures that can be shared between CPU and GPU. Let's create our particle structure in `src/particle.ts`:

```typescript
import tgpu from 'typegpu';

// Define the particle structure
export const Particle = tgpu.struct({
  position: tgpu.vec3f,      // 3D position (x, y, z)
  velocity: tgpu.vec3f,      // 3D velocity vector
  color: tgpu.vec4f,         // RGBA color
  lifetime: tgpu.f32,        // Total lifetime in seconds
  age: tgpu.f32,             // Current age in seconds
});

// Type alias for TypeScript
export type ParticleData = tgpu.Infer<typeof Particle>;

// Helper function to create a random particle
export function createRandomParticle(): ParticleData {
  const angle = Math.random() * Math.PI * 2;
  const speed = 0.5 + Math.random() * 1.5;

  return {
    position: [
      (Math.random() - 0.5) * 2,
      (Math.random() - 0.5) * 2,
      0
    ],
    velocity: [
      Math.cos(angle) * speed,
      Math.sin(angle) * speed,
      (Math.random() - 0.5) * 0.5
    ],
    color: [
      Math.random(),
      Math.random(),
      Math.random(),
      1.0
    ],
    lifetime: 3.0 + Math.random() * 2.0,
    age: 0.0
  };
}
```

Each particle contains:
- **Position**: The particle's location in 3D space
- **Velocity**: How fast and in what direction it's moving
- **Color**: RGBA values for rendering
- **Lifetime**: How long the particle should live before respawning
- **Age**: Current age, used to fade out particles over time

### Simulation Parameters

Create a uniforms structure for physics constants and simulation parameters:

```typescript
export const SimulationParams = tgpu.struct({
  deltaTime: tgpu.f32,           // Time since last frame
  mousePos: tgpu.vec2f,          // Mouse position in normalized coordinates
  mouseActive: tgpu.u32,         // Is mouse being pressed?
  mouseForce: tgpu.f32,          // Strength of mouse interaction
  gravity: tgpu.vec3f,           // Gravity vector
  damping: tgpu.f32,             // Velocity damping factor
  particleCount: tgpu.u32,       // Total number of particles
  bounds: tgpu.vec3f,            // Simulation boundary size
});

export type SimulationParamsData = tgpu.Infer<typeof SimulationParams>;

export const defaultSimParams: SimulationParamsData = {
  deltaTime: 0.016,
  mousePos: [0, 0],
  mouseActive: 0,
  mouseForce: 5.0,
  gravity: [0, -0.5, 0],
  damping: 0.98,
  particleCount: 10000,
  bounds: [2, 2, 1],
};
```

These uniforms control the simulation's behavior and can be updated each frame from the CPU.

## Initializing Particles

### Creating the Particle Buffer

Now let's set up the particle buffers. We'll need two buffers for double buffering (explained later). Create `src/compute.ts`:

```typescript
import tgpu from 'typegpu';
import { Particle, ParticleData, createRandomParticle, SimulationParams } from './particle';

export function initializeParticleBuffers(
  device: GPUDevice,
  particleCount: number
): { bufferA: GPUBuffer; bufferB: GPUBuffer; initialData: ParticleData[] } {

  // Generate initial particle data
  const initialData: ParticleData[] = [];
  for (let i = 0; i < particleCount; i++) {
    initialData.push(createRandomParticle());
  }

  // Calculate buffer size
  const particleSize = Particle.size; // TypeGPU calculates struct size
  const bufferSize = particleSize * particleCount;

  // Create two buffers for ping-pong pattern
  const bufferA = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'Particle Buffer A',
  });

  const bufferB = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'Particle Buffer B',
  });

  // Write initial data to buffer A
  const dataArray = new Float32Array(particleCount * (particleSize / 4));
  initialData.forEach((particle, i) => {
    const offset = i * (particleSize / 4);
    dataArray.set([...particle.position], offset);
    dataArray.set([...particle.velocity], offset + 3);
    dataArray.set([...particle.color], offset + 6);
    dataArray[offset + 10] = particle.lifetime;
    dataArray[offset + 11] = particle.age;
  });

  device.queue.writeBuffer(bufferA, 0, dataArray);
  device.queue.writeBuffer(bufferB, 0, dataArray);

  return { bufferA, bufferB, initialData };
}
```

We create two identical buffers because compute shaders can't safely read and write to the same buffer simultaneously. The ping-pong pattern alternates between reading from one and writing to the other.

## The Compute Shader

### Physics Update Kernel

The compute shader is where the magic happens. This is where we update particle positions, apply forces, and handle lifetimes. Add this to `src/compute.ts`:

```typescript
export function createComputePipeline(device: GPUDevice) {
  const root = tgpu.init();

  // Define our compute shader
  const updateParticles = tgpu
    .computeFn([
      tgpu.param('particlesIn', tgpu.arrayOf(Particle)),
      tgpu.param('particlesOut', tgpu.arrayOf(Particle)),
      tgpu.param('params', SimulationParams),
    ])
    .workgroupSize(64)
    .implement((particlesIn, particlesOut, params) => {
      const idx = tgpu.workgroupId.x * 64 + tgpu.localInvocationId.x;

      // Bounds check
      tgpu.if(idx >= params.particleCount, () => {
        return;
      });

      // Read current particle
      const p = particlesIn[idx];
      const newParticle = tgpu.struct(Particle);

      // Update age
      const newAge = p.age + params.deltaTime;

      // Check if particle should respawn
      tgpu.if(newAge >= p.lifetime, () => {
        // Respawn at random position
        const random1 = tgpu.fract(
          tgpu.sin(idx * 12.9898 + params.deltaTime * 78.233) * 43758.5453
        );
        const random2 = tgpu.fract(
          tgpu.sin(idx * 93.9898 + params.deltaTime * 47.233) * 43758.5453
        );
        const angle = random1 * 6.28318; // 2 * PI
        const speed = 0.5 + random2;

        newParticle.position = tgpu.vec3f(
          (random1 - 0.5) * 2.0,
          (random2 - 0.5) * 2.0,
          0.0
        );
        newParticle.velocity = tgpu.vec3f(
          tgpu.cos(angle) * speed,
          tgpu.sin(angle) * speed,
          (random1 - 0.5) * 0.5
        );
        newParticle.age = 0.0;
        newParticle.lifetime = p.lifetime;
        newParticle.color = p.color;
      })
      .else(() => {
        // Update existing particle
        let vel = p.velocity;

        // Apply gravity
        vel = vel + params.gravity * params.deltaTime;

        // Mouse interaction
        tgpu.if(params.mouseActive > 0, () => {
          const mousePos3D = tgpu.vec3f(params.mousePos.x, params.mousePos.y, 0.0);
          const toMouse = mousePos3D - p.position;
          const dist = tgpu.length(toMouse);

          tgpu.if(dist > 0.01 && dist < 1.5, () => {
            const force = (toMouse / dist) * params.mouseForce * params.deltaTime;
            vel = vel + force;
          });
        });

        // Apply damping
        vel = vel * params.damping;

        // Update position
        let pos = p.position + vel * params.deltaTime;

        // Boundary handling - bouncing
        tgpu.if(tgpu.abs(pos.x) > params.bounds.x, () => {
          pos = tgpu.vec3f(
            tgpu.sign(pos.x) * params.bounds.x,
            pos.y,
            pos.z
          );
          vel = tgpu.vec3f(-vel.x * 0.8, vel.y, vel.z);
        });

        tgpu.if(tgpu.abs(pos.y) > params.bounds.y, () => {
          pos = tgpu.vec3f(
            pos.x,
            tgpu.sign(pos.y) * params.bounds.y,
            pos.z
          );
          vel = tgpu.vec3f(vel.x, -vel.y * 0.8, vel.z);
        });

        newParticle.position = pos;
        newParticle.velocity = vel;
        newParticle.age = newAge;
        newParticle.lifetime = p.lifetime;
        newParticle.color = p.color;
      });

      // Write updated particle
      particlesOut[idx] = newParticle;
    });

  return { updateParticles, root };
}
```

### Force Integration

The compute shader applies several forces to each particle:

1. **Gravity**: A constant downward force defined in simulation parameters
2. **Mouse attraction**: When the mouse is active, particles are pulled toward the cursor
3. **Damping**: Gradually reduces velocity to prevent infinite acceleration

The mouse force calculation computes the distance from each particle to the mouse position, normalizes the direction vector, and applies a force proportional to the `mouseForce` parameter.

### Boundary Handling

We implement bouncing boundaries by detecting when particles exceed the bounds and:
1. Clamping their position to the boundary
2. Reversing the velocity component perpendicular to the boundary
3. Applying a coefficient of restitution (0.8) to simulate energy loss

This creates a realistic bouncing effect. Alternatively, you could implement wrapping boundaries where particles teleport to the opposite side.

### Lifetime and Respawning

Each particle tracks its age, which increments by `deltaTime` each frame. When `age >= lifetime`, the particle is respawned with:
- A new random position
- A new random velocity
- Reset age to 0

We use a simple pseudo-random number generator based on sine functions and the particle index. While not cryptographically secure, it's sufficient for visual randomness and runs efficiently on the GPU.

## The Render Pipeline

### Vertex Shader

Now let's set up the rendering pipeline in `src/render.ts`. We'll use point sprites for simplicity:

```typescript
import tgpu from 'typegpu';
import { Particle } from './particle';

export function createRenderPipeline(device: GPUDevice, format: GPUTextureFormat) {
  const root = tgpu.init();

  // Vertex shader - processes each particle
  const vertexShader = tgpu
    .vertexFn([
      tgpu.param('particle', Particle),
      tgpu.param('vertexIndex', tgpu.builtin.vertexIndex),
    ])
    .outputs({
      position: tgpu.builtin.position,
      color: tgpu.vec4f,
      pointSize: tgpu.builtin.pointSize,
      age: tgpu.f32,
    })
    .implement((particle, vertexIndex) => {
      const lifetimeRatio = particle.age / particle.lifetime;
      const alpha = 1.0 - lifetimeRatio;

      return {
        position: tgpu.vec4f(
          particle.position.x,
          particle.position.y,
          particle.position.z,
          1.0
        ),
        color: tgpu.vec4f(
          particle.color.r,
          particle.color.g,
          particle.color.b,
          alpha
        ),
        pointSize: tgpu.mix(20.0, 5.0, lifetimeRatio),
        age: lifetimeRatio,
      };
    });

  // Fragment shader - renders each particle
  const fragmentShader = tgpu
    .fragmentFn([
      tgpu.param('color', tgpu.vec4f),
      tgpu.param('age', tgpu.f32),
      tgpu.param('pointCoord', tgpu.builtin.pointCoord),
    ])
    .outputs({
      color: tgpu.vec4f,
    })
    .implement((color, age, pointCoord) => {
      // Create circular particles
      const center = tgpu.vec2f(0.5, 0.5);
      const dist = tgpu.distance(pointCoord, center);

      // Soft edge falloff
      const edge = 1.0 - tgpu.smoothstep(0.3, 0.5, dist);

      return {
        color: tgpu.vec4f(
          color.r,
          color.g,
          color.b,
          color.a * edge
        ),
      };
    });

  // Create the render pipeline
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: vertexShader.asShaderModule(device),
      entryPoint: 'main',
      buffers: [],
    },
    fragment: {
      module: fragmentShader.asShaderModule(device),
      entryPoint: 'main',
      targets: [{
        format: format,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one',
            operation: 'add',
          },
        },
      }],
    },
    primitive: {
      topology: 'point-list',
    },
  });

  return { pipeline, root };
}
```

### Fragment Shader

The fragment shader creates circular particles with soft edges:

1. **Distance calculation**: Measures how far the current pixel is from the particle center
2. **Smoothstep falloff**: Creates a smooth gradient from opaque center to transparent edge
3. **Alpha blending**: Multiplies the particle's age-based alpha with the edge falloff

This creates particles that appear as soft, glowing circles that fade out as they age.

### Blending Configuration

We use additive blending for a glowing effect:
- **Color blend**: `srcAlpha + one` - New particles add to existing colors
- **Alpha blend**: `one + one` - Overlapping particles intensify

This creates beautiful layered effects where particles overlap, perfect for fire, sparks, or magical effects. For more subtle particles, you could use standard alpha blending with `srcAlpha + oneMinusSrcAlpha`.

## Double Buffering

### Ping-Pong Pattern

Double buffering solves a critical problem in GPU computing: you can't safely read from and write to the same memory location simultaneously. Imagine updating particle positions - if you're reading particle A's position while also writing to it, you might get corrupted data.

The ping-pong pattern uses two buffers:
1. **Frame 1**: Read from Buffer A, write to Buffer B
2. **Frame 2**: Read from Buffer B, write to Buffer A
3. **Frame 3**: Read from Buffer A, write to Buffer B
4. And so on...

This ensures we always read from a complete, stable dataset while writing to a separate buffer. After each compute pass, we swap which buffer is "current" and which is "next."

```typescript
let currentBuffer = 0; // 0 or 1

function swapBuffers() {
  currentBuffer = 1 - currentBuffer;
}

function getCurrentBuffer() {
  return currentBuffer === 0 ? bufferA : bufferB;
}

function getNextBuffer() {
  return currentBuffer === 0 ? bufferB : bufferA;
}
```

## Interactivity

### Mouse Input

Let's add mouse tracking to make our particle system interactive. Add this to `src/main.ts`:

```typescript
interface MouseState {
  x: number;
  y: number;
  active: boolean;
}

function setupMouseTracking(canvas: HTMLCanvasElement): MouseState {
  const mouse: MouseState = { x: 0, y: 0, active: false };

  const updateMousePosition = (event: MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    // Convert to normalized device coordinates (-1 to 1)
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1); // Flip Y
  };

  canvas.addEventListener('mousemove', updateMousePosition);

  canvas.addEventListener('mousedown', () => {
    mouse.active = true;
  });

  canvas.addEventListener('mouseup', () => {
    mouse.active = false;
  });

  canvas.addEventListener('mouseleave', () => {
    mouse.active = false;
  });

  // Touch support
  canvas.addEventListener('touchmove', (event) => {
    event.preventDefault();
    const touch = event.touches[0];
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -(((touch.clientY - rect.top) / rect.height) * 2 - 1);
    mouse.active = true;
  });

  canvas.addEventListener('touchend', () => {
    mouse.active = false;
  });

  return mouse;
}
```

This code:
1. Tracks mouse position in normalized coordinates (-1 to 1)
2. Flips the Y-axis to match WebGPU's coordinate system
3. Tracks whether the mouse button is pressed
4. Includes touch support for mobile devices

The mouse state is passed to the compute shader through the uniform buffer each frame.

## The Render Loop

### Complete Frame Update

Now let's tie everything together with the main render loop. Here's the complete `src/main.ts`:

```typescript
import tgpu from 'typegpu';
import {
  Particle,
  SimulationParams,
  defaultSimParams,
  SimulationParamsData,
} from './particle';
import { initializeParticleBuffers, createComputePipeline } from './compute';
import { createRenderPipeline } from './render';

async function main() {
  // Get canvas and check WebGPU support
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('Canvas not found');

  if (!navigator.gpu) {
    throw new Error('WebGPU not supported on this browser');
  }

  // Initialize WebGPU
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found');

  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not get WebGPU context');

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  // Configure canvas
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  // Setup mouse tracking
  const mouse = setupMouseTracking(canvas);

  // Initialize particle buffers
  const particleCount = 10000;
  const { bufferA, bufferB } = initializeParticleBuffers(device, particleCount);

  // Create compute pipeline
  const { updateParticles, root: computeRoot } = createComputePipeline(device);

  // Create render pipeline
  const { pipeline: renderPipeline, root: renderRoot } = createRenderPipeline(
    device,
    presentationFormat
  );

  // Create uniform buffer for simulation parameters
  const paramsBuffer = device.createBuffer({
    size: SimulationParams.size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'Simulation Parameters',
  });

  // Ping-pong buffer tracking
  let currentBufferIndex = 0;
  const getInputBuffer = () => (currentBufferIndex === 0 ? bufferA : bufferB);
  const getOutputBuffer = () => (currentBufferIndex === 0 ? bufferB : bufferA);

  // Timing
  let lastTime = performance.now();

  // Render loop
  function frame() {
    const now = performance.now();
    const deltaTime = Math.min((now - lastTime) / 1000, 0.1); // Cap at 100ms
    lastTime = now;

    // Update simulation parameters
    const params: SimulationParamsData = {
      ...defaultSimParams,
      deltaTime,
      mousePos: [mouse.x, mouse.y],
      mouseActive: mouse.active ? 1 : 0,
      particleCount,
    };

    // Write parameters to GPU
    const paramsData = new Float32Array([
      params.deltaTime,
      params.mousePos[0],
      params.mousePos[1],
      params.mouseActive,
      params.mouseForce,
      params.gravity[0],
      params.gravity[1],
      params.gravity[2],
      params.damping,
      params.particleCount,
      params.bounds[0],
      params.bounds[1],
      params.bounds[2],
    ]);
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Create command encoder
    const commandEncoder = device.createCommandEncoder();

    // Compute pass - update particles
    const computePass = commandEncoder.beginComputePass();

    const computeBindGroup = device.createBindGroup({
      layout: updateParticles.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getInputBuffer() } },
        { binding: 1, resource: { buffer: getOutputBuffer() } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    computePass.setPipeline(updateParticles.createPipeline(device));
    computePass.setBindGroup(0, computeBindGroup);

    const workgroupCount = Math.ceil(particleCount / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // Render pass - draw particles
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    const renderBindGroup = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getOutputBuffer() } },
      ],
    });

    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(particleCount);
    renderPass.end();

    // Submit commands
    device.queue.submit([commandEncoder.finish()]);

    // Swap buffers for next frame
    currentBufferIndex = 1 - currentBufferIndex;

    requestAnimationFrame(frame);
  }

  // Start render loop
  requestAnimationFrame(frame);

  // Handle window resize
  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });
  });
}

main().catch(console.error);
```

Each frame:
1. Calculates delta time for frame-rate independent physics
2. Updates simulation parameters with current mouse state
3. Dispatches the compute shader to update all particles
4. Renders particles using the updated buffer
5. Swaps buffers for the next frame

## Complete Code

The complete working example is spread across the files we've created:

- `src/particle.ts`: Data structures and schemas
- `src/compute.ts`: Compute shader and buffer initialization
- `src/render.ts`: Render pipeline setup
- `src/main.ts`: Main application and render loop

To run the project:

```bash
npm install
npm run dev
```

Open your browser to the URL shown (typically `http://localhost:5173`) and you should see thousands of particles bouncing around. Click and drag your mouse to attract particles!

## Performance Considerations

### Particle Count Limits

The number of particles you can simulate depends on your GPU. Modern GPUs can handle 100,000+ particles at 60fps, but integrated GPUs may struggle beyond 20,000. Key factors:

- **Memory bandwidth**: More particles = more data to transfer
- **Compute capacity**: Complex physics per particle increases GPU load
- **Fill rate**: Overlapping particles with alpha blending stress the fragment shader

Start with 10,000 particles and increase until you notice frame drops.

### Workgroup Size

We use a workgroup size of 64, which is a good balance for most GPUs. Workgroup sizes should be:
- Powers of 2 (16, 32, 64, 128, 256)
- Match your GPU's warp/wavefront size (typically 32-64)
- Not too large (limits occupancy) or too small (increases overhead)

Different GPUs have different optimal sizes. NVIDIA tends to prefer 32 or 64, AMD often works well with 64 or 128.

### Memory Access Patterns

Our particle buffer uses a structure-of-arrays (SoA) layout where each particle's data is contiguous. This provides good cache locality when the compute shader processes particles sequentially.

For even better performance with massive particle counts, consider:
- **Spatial partitioning**: Only compute forces between nearby particles
- **Level of detail**: Update distant particles less frequently
- **Frustum culling**: Don't render off-screen particles

## Extensions

### Particle Trails

Add a trail effect by rendering previous particle positions with decreasing alpha:

```typescript
const trailBuffer = device.createBuffer({
  size: particleSize * particleCount * TRAIL_LENGTH,
  usage: GPUBufferUsage.STORAGE,
});
```

In the compute shader, shift old positions and add the current position.

### Different Particle Types

Create multiple particle buffers with different behaviors:
- Heavy particles affected more by gravity
- Light particles that float upward
- Charged particles that repel each other

```typescript
const ParticleType = tgpu.struct({
  ...Particle,
  mass: tgpu.f32,
  charge: tgpu.f32,
});
```

### Collision Detection

Implement particle-particle collisions using spatial hashing:

1. Divide space into a grid
2. Assign each particle to a grid cell
3. Only check collisions within the same or neighboring cells

```typescript
const gridSize = 32;
const cellCount = gridSize * gridSize * gridSize;
const gridBuffer = device.createBuffer({
  size: cellCount * maxParticlesPerCell * 4,
  usage: GPUBufferUsage.STORAGE,
});
```

### 3D Camera

Add a perspective camera for a 3D view:

```typescript
const Camera = tgpu.struct({
  viewMatrix: tgpu.mat4x4f,
  projectionMatrix: tgpu.mat4x4f,
});
```

In the vertex shader:

```typescript
const clipPos = params.projectionMatrix * params.viewMatrix *
                tgpu.vec4f(particle.position, 1.0);
```

Implement camera controls for orbiting, zooming, and panning.

## Conclusion

You've now built a complete GPU-accelerated particle system using TypeGPU! You've learned:

- Setting up a TypeGPU project with proper tooling
- Defining type-safe data structures for GPU and CPU
- Writing compute shaders for physics simulation
- Implementing render pipelines with custom shaders
- Double buffering with the ping-pong pattern
- Adding interactivity with mouse input
- Optimizing for performance

This foundation enables you to create sophisticated visual effects, from realistic fire and smoke to abstract visualizations and game effects. The patterns you've learned—compute-based simulation, efficient GPU memory management, and type-safe shader development—apply to many other GPU computing tasks.

Experiment with the parameters, try the suggested extensions, and create your own unique particle effects. The GPU's parallel processing power combined with TypeGPU's type safety makes it easy to iterate and build complex systems confidently.

Happy coding!
