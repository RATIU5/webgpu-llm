// @ts-check
import starlight from "@astrojs/starlight";
import vercelAdapter from "@astrojs/vercel";
import { defineConfig } from "astro/config";
import starlightLlmsTxt from "starlight-llms-txt";

// https://astro.build/config
export default defineConfig({
  site: "http://localhost:4321",
  adapter: vercelAdapter(),
  integrations: [
    starlight({
      title: "RATIU5' TypeGPU Docs",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/RATIU5/webgpu-llm",
        },
      ],
      sidebar: [
        {
          slug: "getting-started",
          label: "Getting Started",
        },
        {
          slug: "installation",
          label: "CLI Installation",
        },
        {
          label: "Fundamentals",
          autogenerate: { directory: "fundamentals" },
        },
        {
          label: "Data and Buffers",
          autogenerate: { directory: "data-and-buffers" },
        },
        {
          label: "Shaders and Pipelines",
          autogenerate: { directory: "shaders-and-pipelines" },
        },
        {
          label: "Resources and Binding",
          autogenerate: { directory: "resources-and-binding" },
        },
        {
          label: "Advanced",
          autogenerate: { directory: "advanced" },
        },
        {
          label: "Misc",
          autogenerate: { directory: "misc" },
        },
      ],
      plugins: [
        starlightLlmsTxt({
          projectName: "WebGPU and TypeGPU Guide",
          optionalLinks: [
            {
              label: "WebGPU Specification",
              url: "https://www.w3.org/TR/webgpu/",
            },
            { label: "WGSL Specification", url: "https://www.w3.org/TR/WGSL/" },
            {
              label: "WebGPU Explainer",
              url: "https://gpuweb.github.io/gpuweb/explainer/",
            },
            {
              label: "WebGPU GitHub Repository",
              url: "https://github.com/gpuweb/gpuweb",
            },

            {
              label: "TypeGPU Documentation",
              url: "https://docs.swmansion.com/TypeGPU/",
            },
            {
              label: "TypeGPU GitHub",
              url: "https://github.com/software-mansion/TypeGPU",
            },
            {
              label: "TypeGPU Getting Started",
              url: "https://docs.swmansion.com/TypeGPU/getting-started/",
            },
            {
              label: "TypeGPU Functions Guide",
              url: "https://docs.swmansion.com/TypeGPU/fundamentals/functions/",
            },
            {
              label: "TypeGPU Pipelines Guide",
              url: "https://docs.swmansion.com/TypeGPU/fundamentals/pipelines/",
            },
            {
              label: "TypeGPU TGSL Guide",
              url: "https://docs.swmansion.com/TypeGPU/fundamentals/tgsl/",
            },
            {
              label: "TypeGPU Timestamp Queries",
              url: "https://docs.swmansion.com/TypeGPU/fundamentals/timestamp-queries/",
            },
            {
              label: "TypeGPU wgpu-matrix Integration",
              url: "https://docs.swmansion.com/TypeGPU/integration/working-with-wgpu-matrix/",
            },
            {
              label: "TypeGPU NPM Package",
              url: "https://www.npmjs.com/package/typegpu",
            },
            {
              label: "TypeGPU Why Page",
              url: "https://docs.swmansion.com/TypeGPU/why-typegpu/",
            },
          ],
          customSets: [
            {
              label: "WebGPU Core Concepts",
              description:
                "Understanding GPUAdapter, GPUDevice, GPUQueue, and the WebGPU architecture",
              paths: ["fundamentals/webgpu-core-concepts"],
            },
            {
              label: "Canvas Configuration and Context",
              description:
                "Setting up the rendering surface with canvas context and presentation",
              paths: ["fundamentals/canvas-configuration"],
            },
            {
              label: "Command Encoders and Submission",
              description:
                "Building and submitting command buffers to the GPU queue",
              paths: ["fundamentals/command-encoders-submission"],
            },
            {
              label: "WGSL Shading Language",
              description:
                "The WebGPU Shading Language syntax, data types, and functions",
              paths: ["fundamentals/wgsl-shading-language"],
            },
            {
              label: "WGSL Address Spaces",
              description:
                "Understanding function, private, workgroup, uniform, and storage address spaces",
              paths: ["fundamentals/wgsl-address-spaces"],
            },
            {
              label: "TypeGPU Getting Started",
              description:
                "Setting up TypeGPU with bundlers and writing your first type-safe GPU program",
              paths: ["fundamentals/typegpu-getting-started"],
            },
            {
              label: "Error Handling and Validation",
              description:
                "WebGPU's error model, device lost handling, and validation errors",
              paths: ["fundamentals/error-handling-validation"],
            },
            {
              label: "Data Schemas in TypeGPU",
              description:
                "Defining structs, vectors, matrices, and arrays with type inference",
              paths: ["data-and-buffers/typegpu-data-schemas"],
            },
            {
              label: "Buffers and Memory Management",
              description:
                "Creating, reading, writing, and managing GPU buffers",
              paths: ["data-and-buffers/buffers-memory-management"],
            },
            {
              label: "Render Pipelines",
              description:
                "Vertex shaders, fragment shaders, and the graphics rendering pipeline",
              paths: ["shaders-and-pipelines/render-pipelines"],
            },
            {
              label: "Compute Pipelines",
              description:
                "General-purpose GPU computing with compute shaders and workgroups",
              paths: ["shaders-and-pipelines/compute-pipelines"],
            },
            {
              label: "TGSL Functions",
              description:
                "Writing shader functions in TypeScript with TypeGPU's TGSL",
              paths: ["shaders-and-pipelines/tgsl-functions"],
            },
            {
              label: "Workgroup Variables and Shared Memory",
              description:
                "Using workgroup-scoped variables for inter-thread communication",
              paths: ["shaders-and-pipelines/workgroup-variables"],
            },
            {
              label: "Slots and Derived Values",
              description:
                "Dynamic resource binding with slots and computed values with derived",
              paths: ["shaders-and-pipelines/slots-derived-values"],
            },
            {
              label: "Bind Groups and Layouts",
              description:
                "Connecting resources to shaders with type-safe bindings",
              paths: ["resources-and-binding/bind-groups-layouts"],
            },
            {
              label: "Textures and Samplers",
              description:
                "Working with images, texture mapping, and sampling operations",
              paths: ["resources-and-binding/textures-samplers"],
            },
            {
              label: "WGSL Builtin Functions",
              description:
                "Reference for math, vector, texture, and synchronization functions in WGSL",
              paths: ["fundamentals/wgsl-builtin-functions"],
            },
            {
              label: "TypeGPU Tooling",
              description:
                "Build tools and CLI utilities for TypeGPU development",
              paths: ["fundamentals/typegpu-tooling"],
            },
            {
              label: "TypeGPU Utilities",
              description:
                "Noise generation, color spaces, and helper packages for TypeGPU",
              paths: ["fundamentals/typegpu-utilities"],
            },
            {
              label: "Loading Images and Textures",
              description:
                "Loading images, video frames, and compressed textures into WebGPU",
              paths: ["data-and-buffers/loading-images-textures"],
            },
            {
              label: "Instancing and Indirect Drawing",
              description:
                "Efficient rendering of many objects with instancing and GPU-driven draws",
              paths: ["shaders-and-pipelines/instancing-indirect-drawing"],
            },
            {
              label: "Render Bundles",
              description:
                "Pre-recorded command sequences for reducing CPU overhead",
              paths: ["shaders-and-pipelines/render-bundles"],
            },
            {
              label: "Storage Textures",
              description:
                "Direct texture writes from compute and fragment shaders",
              paths: ["resources-and-binding/storage-textures"],
            },
            {
              label: "Multisampling (MSAA)",
              description: "Anti-aliasing with multisample render targets",
              paths: ["resources-and-binding/multisampling-msaa"],
            },
            {
              label: "React Integration",
              description:
                "WebGPU patterns for React web and React Native applications",
              paths: ["advanced/react-integration"],
            },
            {
              label: "Camera and View Matrices",
              description:
                "Camera math, projection matrices, and wgpu-matrix usage",
              paths: ["misc/camera-view-matrices"],
            },
            {
              label: "Advanced TypeGPU Patterns",
              description:
                "Externals, resolve API, WebGPU interoperability, and React Native support",
              paths: ["advanced/advanced-typegpu-patterns"],
            },
            {
              label: "Performance Optimization",
              description:
                "Best practices for GPU memory, pipeline efficiency, and profiling",
              paths: ["advanced/performance-optimization"],
            },
            {
              label: "Atomic Operations",
              description:
                "Thread-safe operations in compute shaders for concurrent data access",
              paths: ["advanced/atomic-operations"],
            },
            {
              label: "Coordinate Systems and Clip Space",
              description:
                "Understanding WebGPU's 0-to-1 depth range and coordinate transformations",
              paths: ["advanced/coordinate-systems"],
            },
            {
              label: "Timestamp Queries and Profiling",
              description:
                "Measuring GPU execution time with query sets and performance callbacks",
              paths: ["misc/timestamp-queries-profiling"],
            },
            {
              label: "WebGPU vs WebGL Migration",
              description:
                "Differences, advantages, and migration considerations including coordinate system changes",
              paths: ["misc/webgpu-vs-webgl"],
            },
            {
              label: "Debugging GPU Code",
              description:
                "GPU console.log, validation layers, and debugging techniques",
              paths: ["misc/debugging-gpu-code"],
            },
            {
              label: "Browser Compatibility",
              description:
                "Current support status, feature detection, and fallback strategies",
              paths: ["misc/browser-compatibility"],
            },
          ],
        }),
      ],
    }),
  ],
});
