# TypeGPU Getting Started

## Overview

TypeGPU is a modular and open-ended toolkit for WebGPU that brings type safety and developer-friendly abstractions to GPU programming in JavaScript and TypeScript. Developed by Software Mansion, TypeGPU provides advanced type inference capabilities and enables developers to write GPU shaders directly in TypeScript, eliminating the context-switching between JavaScript/TypeScript application code and WGSL (WebGPU Shading Language) shader code.

At its core, TypeGPU is built on a philosophy of **type-safe WebGPU development**. It mirrors WGSL syntax in TypeScript while providing compile-time type checking, autocomplete support, and static analysis capabilities that are impossible with traditional string-based shader code. The toolkit is designed to be modular and non-opinionated, allowing developers to adopt it incrementally and "granularly eject into vanilla WebGPU at any point," minimizing concerns about vendor lock-in.

TypeGPU serves three primary use cases:

1. **Foundation for New Projects**: Solves common WebGPU challenges like data serialization, buffer management, and shader composition while maintaining flexibility to drop down to vanilla WebGPU when needed.

2. **Integration with Existing Code**: Type-safe APIs can be used independently, enabling partial adoption into established applications regardless of their complexity or current WebGPU implementation.

3. **Library Interoperability**: Functions as an interoperability layer between use-case specific libraries, allowing different WebGPU tools to share typed data without expensive CPU memory transfers.

The low-level nature of TypeGPU and its mirroring of WGSL syntax means that learning TypeGPU helps developers learn WebGPU itself, reducing frustrations commonly associated with GPU programming while maintaining performance characteristics identical to hand-written WebGPU code.

## Why TypeGPU?

TypeGPU addresses several pain points in traditional WebGPU development through a combination of innovative features and thoughtful design decisions.

### Type-Safe Data Schemas with Automatic Validation

One of the most significant challenges in WebGPU development is managing data transfer between CPU and GPU. Data must be serialized into binary buffers following strict alignment and layout rules defined by WGSL. Mistakes in this process lead to subtle bugs that are difficult to diagnose.

TypeGPU solves this problem through **composable data schemas**. Every WGSL data type can be represented as JavaScript schemas imported from `typegpu/data`, including primitive types, structs, and arrays. These schemas provide:

- **Automatic serialization and deserialization**: Complex data structures are automatically converted to/from GPU-compatible binary formats
- **Compile-time validation**: TypeScript catches type mismatches before runtime
- **Runtime validation**: Ensures data conforms to expected schemas when crossing CPU-GPU boundaries
- **Self-documenting code**: Schema definitions serve as clear documentation of data structures

For example, a complex struct with nested arrays and multiple fields can be defined once and used consistently across both CPU and GPU code, with TypeScript ensuring that all access patterns are valid.

### TGSL: Writing Shaders in TypeScript

TGSL (TypeGPU Shading Language) is perhaps the most distinctive feature of TypeGPU. It's a subset of JavaScript/TypeScript that can be transpiled to WGSL, enabling developers to write GPU shader code in the same language as their application code.

Key advantages of TGSL include:

- **Unified Language**: Write both host and kernel code in TypeScript, reducing cognitive load and context switching
- **Full IDE Support**: Get syntax highlighting, autocomplete, type checking, and refactoring tools that work seamlessly with shader code
- **Code Reuse**: Define utility functions once and use them in both CPU and GPU contexts
- **Type Safety**: Shader function signatures are fully visible to TypeScript, enabling static checks and preventing common errors
- **Gradual Adoption**: Mix and match TGSL and WGSL freely - you're never locked into one approach

TGSL works by transpiling JavaScript into a compact AST format called "tinyest," which is then used to generate equivalent WGSL code. This transformation happens at build time via the `unplugin-typegpu` bundler plugin, ensuring zero runtime overhead.

It's important to note that JavaScript doesn't support operator overloading, so while you can use standard operators for numbers, operations involving vectors and matrices require supplementary functions from `typegpu/std` (like `add`, `mul`, `eq`, `lt`, `ge`). This is a trade-off for the benefits of writing shaders in TypeScript, though for complex matrix operations, developers can always drop down to WGSL for improved readability.

### Automatic WGSL Generation

TypeGPU automatically generates WGSL shader code from TypeScript function definitions marked with the `'use gpu'` directive. This generation process:

- Maintains semantic equivalence to the original TypeScript code
- Produces readable, debuggable WGSL output
- Handles all necessary type conversions and memory layout specifications
- Optimizes for GPU execution patterns

Developers can inspect the generated WGSL at any time, making it easy to understand exactly what's running on the GPU and to debug performance issues.

### Bindless Resources with Named Keys

Traditional WebGPU requires explicit binding of resources (buffers, textures, samplers) to specific bind group slots using numeric indices. This approach is error-prone and makes code difficult to maintain.

TypeGPU introduces **bindless resources** where resources are referenced by descriptive string keys rather than numeric indices. This provides:

- **Readable Code**: `resources.particlePositions` is clearer than `binding(0)`
- **Flexibility**: Add or reorder resources without updating all binding indices
- **Type Safety**: TypeScript ensures referenced resources actually exist
- **Automatic Naming**: The build plugin can automatically name resources based on variable names

This abstraction significantly reduces boilerplate code and makes shader interfaces self-documenting.

### Better Developer Experience

Beyond specific features, TypeGPU is designed from the ground up for developer experience:

- **Incremental Adoption**: Start using TypeGPU for just buffer management or just shader authoring, then expand usage over time
- **Minimal Boilerplate**: Common patterns are abstracted away without hiding important details
- **Clear Error Messages**: Type errors and runtime validation failures provide actionable feedback
- **Excellent Documentation**: Comprehensive guides, API documentation, and examples
- **Active Community**: Join the Software Mansion Discord for help and discussions
- **Cross-Platform Support**: Works on web and React Native thanks to the react-native-wgpu library

## Installation

Installing TypeGPU is straightforward using your preferred package manager:

```bash
npm install typegpu
```

Or with Yarn:

```bash
yarn add typegpu
```

Or with pnpm:

```bash
pnpm add typegpu
```

For full TGSL functionality (writing shaders in TypeScript), you'll also need to install the bundler plugin:

```bash
npm install --save-dev unplugin-typegpu
```

The latest stable version is 0.8.2 as of the time of writing. TypeGPU follows semantic versioning, so check the [GitHub releases page](https://github.com/software-mansion/TypeGPU/releases) for the most recent version and changelog.

### Additional Dependencies

If you're using TypeScript (recommended), ensure you have the WebGPU type definitions:

```bash
npm install --save-dev @webgpu/types
```

This provides TypeScript definitions for the WebGPU API itself, which TypeGPU builds upon.

## Bundler Configuration

TypeGPU's shader transpilation features rely on `unplugin-typegpu`, a build-time plugin that hooks into your bundler of choice. The plugin enables TGSL-to-WGSL transpilation, automatic resource naming, and various optimizations.

The plugin is built using [unplugin](https://github.com/unjs/unplugin), which provides a unified plugin API across multiple bundlers. This means the same plugin package works with Vite, Webpack, Rollup, esbuild, Rspack, Rolldown, and Farm.

**Important Note**: The Vite/Rollup and Babel implementations receive the most active maintenance and testing from the TypeGPU team. While other bundlers are supported, their stability may vary.

### Vite Setup

Vite is the recommended bundler for TypeGPU projects due to its fast development server, excellent TypeScript support, and robust plugin ecosystem.

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import typegpuPlugin from 'unplugin-typegpu/vite';

export default defineConfig({
  plugins: [
    typegpuPlugin({
      // Enable automatic naming of TypeGPU resources
      autoNamingEnabled: true,

      // Skip files without TypeGPU references for faster builds
      earlyPruning: true,

      // Customize which files are processed (optional)
      include: /\.m?[jt]sx?$/,

      // Exclude certain files (optional)
      exclude: undefined,
    }),
  ],
});
```

**Configuration Options**:

- `autoNamingEnabled` (default: `true`): Automatically names TypeGPU resources based on variable names, reducing the need for manual `$name()` calls
- `earlyPruning` (default: `true`): Skips processing files that don't import TypeGPU, significantly improving build performance
- `include` (default: `/\.m?[jt]sx?$/`): File patterns to process, using regular expressions or glob patterns
- `exclude` (default: `undefined`): Patterns to skip during processing
- `forceTgpuAlias`: Accommodates custom import names when automatic detection fails

### Webpack Setup

For projects using Webpack, the configuration is slightly different:

```javascript
// webpack.config.js
const TypeGPUPlugin = require('unplugin-typegpu/webpack');

module.exports = {
  // ... other webpack configuration

  plugins: [
    TypeGPUPlugin({
      autoNamingEnabled: true,
      earlyPruning: true,
    }),
  ],

  // Ensure TypeScript files are handled properly
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },

  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
};
```

Make sure you have `ts-loader` or another TypeScript loader configured to handle `.ts` and `.tsx` files before the TypeGPU plugin processes them.

### Rollup Setup

For standalone Rollup configurations (not through Vite):

```javascript
// rollup.config.js
import typescript from '@rollup/plugin-typescript';
import typegpu from 'unplugin-typegpu/rollup';

export default {
  input: 'src/index.ts',
  output: {
    dir: 'dist',
    format: 'es',
  },
  plugins: [
    typescript(),
    typegpu({
      autoNamingEnabled: true,
      earlyPruning: true,
    }),
  ],
};
```

The TypeGPU plugin should generally come after TypeScript compilation but before any minification or bundling steps.

### Babel Plugin (Alternative)

For projects not using a modern bundler, or for React Native projects, TypeGPU provides a Babel plugin:

```javascript
// babel.config.js
module.exports = (api) => {
  api.cache(true);

  return {
    presets: [
      // Your existing presets
      '@babel/preset-env',
      '@babel/preset-typescript',
      // For React Native: 'babel-preset-expo'
    ],
    plugins: [
      // Your existing plugins
      'unplugin-typegpu/babel',
    ],
  };
};
```

The Babel plugin is particularly useful for React Native projects using Expo, where traditional bundler plugins may not be available.

## First TypeGPU Program

Let's create a simple but complete TypeGPU program that demonstrates core concepts. This example performs a basic computation: squaring an array of numbers on the GPU.

```typescript
// main.ts
import tgpu from 'typegpu';
import { arrayOf, f32 } from 'typegpu/data';

async function main() {
  // Initialize TypeGPU and request a GPU device
  const root = await tgpu.init();

  // Define our input data
  const inputData = [1, 2, 3, 4, 5];

  // Create an input buffer with our data
  // arrayOf(f32, 5) creates a schema for an array of 5 floats
  const inputBuffer = root
    .createBuffer(arrayOf(f32, 5), inputData)
    .$usage('storage');

  // Create an output buffer (initially empty)
  const outputBuffer = root
    .createBuffer(arrayOf(f32, 5))
    .$usage('storage');

  // Define a compute shader using TGSL
  const squareNumbers = tgpu
    .fn([inputBuffer, outputBuffer])
    .does(() => {
      'use gpu'; // This marks the function for GPU execution

      // Get the current invocation index
      const idx = builtin.globalInvocationId.x;

      // Read from input, square it, write to output
      outputBuffer[idx] = inputBuffer[idx] * inputBuffer[idx];
    })
    .$name('squareNumbers');

  // Create a compute pipeline
  const pipeline = root
    .makeComputePipeline(squareNumbers)
    .$workgroupSize(1); // One thread per workgroup

  // Execute the shader
  // Dispatch 5 workgroups (one for each element)
  root
    .createCommandEncoder()
    .beginComputePass()
    .setPipeline(pipeline)
    .dispatchWorkgroups(5)
    .end()
    .submit();

  // Read back the results
  const results = await outputBuffer.read();

  console.log('Input:', inputData);
  console.log('Output (squared):', Array.from(results));
  // Expected output: [1, 4, 9, 16, 25]

  // Cleanup
  root.destroy();
}

main().catch(console.error);
```

### Understanding the Example

Let's break down each part of this program:

**1. Initialization**:
```typescript
const root = await tgpu.init();
```
This requests a GPU device from the browser and creates a "root" object that serves as the entry point for all TypeGPU operations. The root manages the GPU device lifecycle and provides factory methods for creating buffers, pipelines, and other resources.

**2. Data Schemas**:
```typescript
arrayOf(f32, 5)
```
This creates a type schema describing an array of 5 32-bit floating-point numbers. TypeGPU uses these schemas to:
- Determine buffer sizes and memory layouts
- Validate data when writing to buffers
- Properly deserialize data when reading from buffers
- Provide TypeScript type information

**3. Buffer Creation**:
```typescript
const inputBuffer = root
  .createBuffer(arrayOf(f32, 5), inputData)
  .$usage('storage');
```
Creates a GPU buffer with storage usage, allowing it to be read and written by shaders. The `$usage()` method is chainable and specifies how the buffer will be used. Common usage flags include:
- `'storage'`: Read/write access from shaders
- `'uniform'`: Read-only constant data
- `'vertex'`: Vertex data for rendering
- `'copy-src'` / `'copy-dst'`: For buffer transfers

**4. TGSL Shader Function**:
```typescript
const squareNumbers = tgpu
  .fn([inputBuffer, outputBuffer])
  .does(() => {
    'use gpu';
    // Shader code here
  })
  .$name('squareNumbers');
```
The `'use gpu'` directive marks this function for TGSL transpilation. The function receives the buffers as dependencies, making them accessible within the shader. The `$name()` method provides a debug-friendly name for the shader.

**5. Pipeline Creation and Execution**:
```typescript
const pipeline = root
  .makeComputePipeline(squareNumbers)
  .$workgroupSize(1);
```
Creates a compute pipeline from the shader. The workgroup size determines how many threads execute together as a group. For this simple example, we use 1, but real-world shaders typically use larger sizes (64, 128, or 256) for better GPU utilization.

**6. Command Encoding and Submission**:
```typescript
root
  .createCommandEncoder()
  .beginComputePass()
  .setPipeline(pipeline)
  .dispatchWorkgroups(5)
  .end()
  .submit();
```
WebGPU uses a command-based API where you record commands into a command buffer, then submit them to the GPU queue. This example dispatches 5 workgroups, one for each array element.

**7. Reading Results**:
```typescript
const results = await outputBuffer.read();
```
Asynchronously reads data back from the GPU to CPU memory. This operation may involve GPU-to-CPU transfer latency, so it's often best to batch multiple computations before reading results.

## The tgpu.init() Pattern

TypeGPU provides flexible initialization patterns to accommodate different use cases and integration scenarios.

### Basic Initialization

```typescript
const root = await tgpu.init();
```

This is the simplest form, which:
1. Requests a GPU adapter from the navigator
2. Requests a device from the adapter with default settings
3. Creates and returns a root object wrapping the device

### Custom Adapter and Device Options

For fine-grained control over GPU device selection and capabilities:

```typescript
const root = await tgpu.init({
  adapter: {
    powerPreference: 'high-performance', // or 'low-power'
  },
  device: {
    requiredFeatures: ['timestamp-query'],
    requiredLimits: {
      maxStorageBufferBindingSize: 1024 * 1024 * 1024, // 1GB
    },
  },
});
```

**Adapter Options**:
- `powerPreference`: Choose between 'low-power' (integrated GPU), 'high-performance' (discrete GPU), or undefined (let browser decide)

**Device Options**:
- `requiredFeatures`: Array of WebGPU features your application needs (e.g., 'timestamp-query', 'texture-compression-bc')
- `requiredLimits`: Override default resource limits (buffer sizes, texture dimensions, etc.)
- `label`: Debug label for the device

### Integrating with Existing WebGPU Code

If you already have a WebGPU device (perhaps from another library), use `tgpu.initFromDevice()`:

```typescript
// Existing WebGPU setup
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: ['timestamp-query'],
});

// Wrap with TypeGPU
const root = tgpu.initFromDevice(device);
```

This pattern is essential for:
- Integrating TypeGPU into existing WebGPU applications
- Using TypeGPU alongside other WebGPU libraries
- Sharing a single GPU device across multiple subsystems

### The Root Object

The root object returned by `tgpu.init()` is your primary interface to TypeGPU. It provides:

**Resource Creation**:
- `createBuffer(schema, data?)`: Create GPU buffers with type schemas
- `createTexture()`: Create typed textures
- `createSampler()`: Create texture samplers

**Pipeline Creation**:
- `makeComputePipeline(fn)`: Create compute pipelines from TGSL functions
- `makeRenderPipeline()`: Create render pipelines for graphics

**Command Encoding**:
- `createCommandEncoder()`: Start recording GPU commands
- `immediatePass()`: Helper for simple single-pass operations

**Device Management**:
- `device`: Access the underlying WebGPU device
- `destroy()`: Clean up all resources and lose the device

## Project Structure Recommendations

Organizing TypeGPU projects effectively improves maintainability and code reuse. Here's a recommended structure:

```
project-root/
├── src/
│   ├── gpu/
│   │   ├── schemas/           # Data type definitions
│   │   │   ├── particle.ts
│   │   │   └── simulation.ts
│   │   ├── shaders/          # TGSL shader functions
│   │   │   ├── compute/
│   │   │   │   ├── physics.ts
│   │   │   │   └── sorting.ts
│   │   │   └── render/
│   │   │       ├── vertex.ts
│   │   │       └── fragment.ts
│   │   ├── pipelines/        # Pipeline configurations
│   │   │   ├── simulation.ts
│   │   │   └── rendering.ts
│   │   └── resources/        # Buffer and texture management
│   │       └── buffers.ts
│   ├── utils/                # CPU-side utilities
│   └── main.ts               # Application entry point
├── vite.config.ts            # Build configuration
├── tsconfig.json             # TypeScript configuration
└── package.json
```

**Key Principles**:

1. **Separate GPU and CPU Code**: Keep GPU-related code in a dedicated directory for clarity
2. **Group by Functionality**: Organize shaders by their purpose (compute vs. render, physics vs. graphics)
3. **Centralize Schemas**: Define data schemas in one place and import them wherever needed
4. **Reusable Shaders**: Write shader functions that can be composed and reused across pipelines

## TypeScript Configuration

TypeGPU requires specific TypeScript settings for optimal functionality:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],

    // Essential for TypeGPU
    "types": ["@webgpu/types"],

    // Recommended for type safety
    "strict": true,
    "strictNullChecks": true,
    "noImplicitAny": true,

    // For better development experience
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,

    // Output configuration
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Critical Settings**:

- `types: ["@webgpu/types"]`: Includes WebGPU type definitions
- `moduleResolution: "bundler"`: Required for modern bundlers like Vite
- `strict: true`: Enables all strict type checking options, catching potential bugs early

For Vite projects, also create or update `src/vite-env.d.ts`:

```typescript
/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />
```

This ensures TypeScript recognizes both Vite-specific globals and WebGPU APIs.

## Best Practices

### Naming Conventions with $name

Use the `$name()` method on all TypeGPU resources for better debugging:

```typescript
const particleBuffer = root
  .createBuffer(particleSchema)
  .$usage('storage')
  .$name('particlePositions'); // Shows up in GPU debugging tools

const updateShader = tgpu
  .fn([particleBuffer])
  .does(() => { 'use gpu'; /* ... */ })
  .$name('updateParticles'); // Clear shader names in profilers
```

With `autoNamingEnabled` in the bundler plugin, many names are added automatically, but explicit naming is still valuable for dynamically created resources.

### Resource Organization

Group related resources into objects for easier management:

```typescript
const resources = {
  buffers: {
    particles: root.createBuffer(particleSchema).$usage('storage'),
    forces: root.createBuffer(forceSchema).$usage('storage'),
  },

  pipelines: {
    physics: root.makeComputePipeline(physicsShader),
    render: root.makeRenderPipeline(renderShader),
  },
};
```

### Type Definitions

Leverage TypeScript's type inference with TypeGPU schemas:

```typescript
import { struct, f32, vec3f } from 'typegpu/data';

// Define schema
const Particle = struct({
  position: vec3f,
  velocity: vec3f,
  mass: f32,
});

// Extract TypeScript type from schema
type ParticleData = typeof Particle.infer;

// Now you have full type safety:
const particle: ParticleData = {
  position: [0, 0, 0],
  velocity: [1, 0, 0],
  mass: 1.0,
};
```

### Performance Considerations

- **Minimize CPU-GPU Transfers**: Batch operations and avoid reading from buffers in tight loops
- **Use Appropriate Workgroup Sizes**: Typically powers of 2 (64, 128, 256) for optimal GPU utilization
- **Reuse Pipelines**: Pipeline creation is expensive; create once and reuse
- **Profile Regularly**: Use browser DevTools and WebGPU-specific profiling tools

## Common Pitfalls

### Plugin Configuration Issues

**Problem**: TGSL functions don't transpile, or build fails silently.

**Solution**: Ensure `unplugin-typegpu` is properly configured:
- Check that the plugin is in the correct position in your bundler's plugin array
- Verify `include` patterns match your source files
- Enable verbose logging to see what files are being processed

### TypeScript Config Problems

**Problem**: WebGPU types not recognized, or module resolution errors.

**Solution**:
- Add `"types": ["@webgpu/types"]` to `tsconfig.json`
- Set `"moduleResolution": "bundler"` for Vite projects
- Ensure `@webgpu/types` is installed as a dev dependency

### Bundler Issues

**Problem**: Build works in development but fails in production.

**Solution**:
- Some bundlers treat development and production differently
- Check that TypeGPU plugin runs in both modes
- Verify that TGSL code doesn't use Node.js-specific features

### Missing 'use gpu' Directive

**Problem**: TGSL function throws runtime errors or behaves unexpectedly.

**Solution**: Always include `'use gpu'` as the first statement in TGSL functions:

```typescript
const shader = tgpu.fn([]).does(() => {
  'use gpu'; // Essential!
  // Shader code here
});
```

### Buffer Usage Flags

**Problem**: Shader can't access buffer, or read/write operations fail.

**Solution**: Ensure buffers have the correct usage flags:
- Use `.$usage('storage')` for read/write access in shaders
- Use `.$usage('uniform')` for read-only data
- Add `.$usage('copy-dst')` if writing from CPU after creation
- Add `.$usage('copy-src')` if reading back to CPU

Multiple usage flags can be combined:
```typescript
const buffer = root
  .createBuffer(schema)
  .$usage('storage')
  .$usage('copy-src'); // Can be used in shaders AND read back to CPU
```

## Related Topics

To continue your TypeGPU journey, explore these related documentation pages:

- **[Data Schemas](https://docs.swmansion.com/TypeGPU/fundamentals/data-schemas/)**: Deep dive into TypeGPU's type system, including structs, arrays, and custom data types
- **[TGSL Functions](https://docs.swmansion.com/TypeGPU/fundamentals/tgsl/)**: Comprehensive guide to writing shaders in TypeScript, including limitations and best practices
- **[Buffers](https://docs.swmansion.com/TypeGPU/fundamentals/buffers/)**: Advanced buffer management, including buffer views, uniform buffers, and optimization techniques
- **[Functions](https://docs.swmansion.com/TypeGPU/fundamentals/functions/)**: Learn about composing shader functions and sharing code between CPU and GPU
- **[WebGPU Interoperability](https://docs.swmansion.com/TypeGPU/integration/webgpu-interoperability/)**: Integrate TypeGPU with existing WebGPU code and libraries

## Additional Resources

- **Official Documentation**: [https://docs.swmansion.com/TypeGPU/](https://docs.swmansion.com/TypeGPU/)
- **GitHub Repository**: [https://github.com/software-mansion/TypeGPU](https://github.com/software-mansion/TypeGPU)
- **npm Package**: [https://www.npmjs.com/package/typegpu](https://www.npmjs.com/package/typegpu)
- **Community Discord**: [Software Mansion Discord](https://discord.gg/8jpfgDqPcM)
- **Examples and Demos**: Check the `apps/` directory in the GitHub repository for real-world examples

## Conclusion

TypeGPU represents a significant advancement in WebGPU development, bringing type safety, developer experience improvements, and code reusability to GPU programming in JavaScript and TypeScript. By following this getting started guide, you should now have a solid foundation for building high-performance GPU-accelerated applications with TypeGPU.

The modular design of TypeGPU means you can adopt it incrementally—start with just buffer management, add TGSL shaders when comfortable, and gradually expand usage as your needs grow. The ability to "granularly eject" to vanilla WebGPU at any point ensures you're never locked in and can always optimize critical paths with hand-written code.

Happy GPU programming with TypeGPU!

---

*This documentation is based on TypeGPU version 0.8.2. For the latest updates and features, always refer to the [official documentation](https://docs.swmansion.com/TypeGPU/).*
