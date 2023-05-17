# WebGPU For Dummies

## About

In this document, authored by [Amir Sojoodi](https://people.distributive.network/amir), I have jumped into the world of WebGPU compute and explore its potential for High-Performance Computing (HPC) and High-Throughput Computing (HTC) applications. It is based on my personal notes gathered during the development of a project at [Distributive](https://distributive.network/). The aim of this document is to provide readers with a clear and accessible tips and tricks to WebGPU compute, enabling them to harness its power for their own projects.

Now, I must admit, coming from an HPC background with expertise in technologies like CUDA, OpenMP, and MPI, I had underestimated the initial challenges of transitioning to the world of WebGPU and JavaScript. It's like switching from driving a Ferrari to a self balancing hoverboard! However, I've put in every ounce of effort to squeeze out every drop of performance potential from the WebGPU API in my own work, and I hope you can do the same!

**Disclaimer**: Please note that the content in this document primarily references the December 2022 draft of the WebGPU specification as published by the World Wide Web Consortium (W3C) at https://www.w3.org/TR/webgpu/. While the specification may have evolved since then, the fundamental concepts and principles discussed here remain relevant and applicable to understanding WebGPU compute.

Now, let's dive into the fascinating(and frustrating!) world of WebGPU compute!

general notes:
  - Experiences on porting applications to WebGPU

## Introduction to WebGPU Compute

**So, what exactly is WebGPU compute?** Well, it's a cutting-edge web technology that introduces a low-level, high-performance computing API for your favorite web browsers. Gone are the days when GPUs were only used for rendering jaw-dropping graphics. With WebGPU compute, developers like us can tap into the immense computational capabilities of GPUs for a wide range of tasks that go beyond just pixel-pushing.

**But why is this such a big deal?** Well, traditionally, if we wanted to leverage the full power of GPUs, we had to rely on platform-specific technologies like CUDA or OpenCL. Don't get me wrong, those technologies are absolute beasts in terms of power, but they often tied us down to a specific operating system or programming language. WebGPU compute, on the other hand, breaks down those barriers and brings GPU-accelerated computing to the web using a standardized API.

So, in summary, here are the benefits of utilizing WebGPU:

1. Parallel Processing Power
2. Platform Independence
3. Web Integration (Duh!)
4. Ease of Use: That's an unfullfilled promise for now!
5. Performance Portability: This means that applications can achieve similar performance characteristics across a wide range of devices, from laptops to desktops and even mobile devices, without sacrificing efficiency.

Now, before you get carried away, let me warn you: WebGPU compute isn't all rainbows and unicorns. As with any new technology, there are challenges to overcome. From mastering the intricacies of JavaScript to optimizing your code for parallel execution, you'll face a few hurdles along the way. 

## What is out there?

There are many great tutorials and manuals out there:

- WebGPU [Explainer](https://gpuweb.github.io/gpuweb/explainer/)
- Introduction by [Surma](https://surma.dev/things/webgpu/)
- Nikita's great [collection](https://wiki.nikiv.dev/computer-graphics/webgpu)
- Chrome team [article](https://developer.chrome.com/articles/gpu-compute/)
- (more in the References section)

And the list goes on! Therefore, I won't bombard you with redundant details covered in the specification and other tutorials. I'll just provide you with a summary of the key notes that serve as handy reminders.

## Core concepts

Let's familiarize ourselves with some key concepts

1. Adapter and device
2. Initialization
3. Timeline
4. Buffer creation
5. Buffer mapping
6. Pipeline
7. Command buffers
8. Queue

### Adapter and Device

The adapter is like the gateway to the GPU. It represents the physical GPU device available on the user's system. The device, on the other hand, is the driver that manages communication with the adapter. Together, they form the dynamic duo that powers your WebGPU app. I stole this picture from [Andi](https://cohost.org/mcc/post/1406157-i-want-to-talk-about-webgpu):

![insert picture](./Images/wgpu.png):
wgpu.png


### Timeline

A computer system with a user agent at the front-end and GPU at the back-end has components working on different timelines in parallel:

1. **Content timeline**: Associated with the execution of the Web script. It includes calling all methods described by this specification.
2. **Device timeline**: Associated with the GPU device operations that are issued by the user agent. It includes creation of adapters, devices, and GPU resources and state objects, which are typically synchronous operations from the point of view of the user agent part that controls the GPU, but can live in a separate OS process.
3. **Queue timeline**: Associated with the execution of operations on the compute units of the GPU. It includes actual draw, copy, and compute jobs that run on the GPU.

## Core concepts

1. `Adapter`
   - An adapter identifies an implementation of WebGPU on the system:
   - Both an instance of compute/rendering functionality on the platform underlying a browser,
   - and an instance of a browser's implementation of WebGPU on top of that functionality.
2. `GPUDevice`
   - `await adapter.requestDevice(options);`
   - Primary interface for the API
   - Creates resources like Textures, Buffers, Pipelines, etc.
   - Has a `GPUQueue` for executing commands
3. `GPUAdapter.features`
   - Adapter lists which ones are available.
   - Must be specified when the requesting a Device or they won't be active.
4. `GPUAdapter.limits`
   - A sample output can be seen [here](./GTX1060-GPUAdapter.limits.out).
5. Adapter Info - `adapter.requestAdapterInfo()`
   - Information including *vendor, architecture, device, driver, and description*

## Initialization

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const context = canvas.getContext('webgpu');
context.configure({
   device,
   format: 'bgra8unorm',
});
```

## Creating Buffers

```js
const vertexData = new Float32Array([
   0, 1, 1,
   -1, -1, 1,
   1, -1, 1
]);

const vertexBuffer = device.createBuffer({
   size: vertexData.byteLength,
   usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, vertexData);
```

## Buffer Mapping

|                             | Regular ArrayBuffer | Shared Memory | Mappable GPU buffer | Non-mappable GPU buffer (or texture) |
| :-------------------------: | :-----------------: | :-----------: | :-----------------: | :----------------------------------: |
| CPU, in the content process |     **Visible**     |  **Visible**  |     Not visible     |             Not visible              |
|   CPU, in the GPU process   |     Not visible     |  **Visible**  |     **Visible**     |             Not visible              |
|             GPU             |     Not visible     |  Not visible  |     **Visible**     |             **Visible**              |

An application can request to map a `GPUBuffer` so that they can access its content via `ArrayBuffers` that represent part of the `GPUBuffer`'s allocations. Mapping a GPUBuffer is requested asynchronously with `mapAsync()` so that the user agent can ensure the GPU finished using the `GPUBuffer` before the application can access its content. A mapped `GPUBuffer` cannot be used by the GPU and must be unmapped using `unmap()` before work using it can be submitted to the Queue timeline.

**Important** point:
`GPUBuffer` mapping is done as an ownership transfer between the CPU and the GPU. At each instant, only one of the two can access the buffer, so no race is possible. In summary, GPU cannot access mapped buffers, and CPU cannot access unmapped ones.

## Pipelines

Structurally, the pipeline consists of a sequence of programmable stages (shaders) and fixed-function states, such as the blending modes.

- Comes in `GPURenderPipeline` and `GPUComputePipeline`
- Immutable after creation

```js
const pipeline = device.createComputePipeline({
  layout: pipelineLayout,
  compute: {
    module: shaderModule,
    entryPoint: 'computeMain',
  }
});
```

## Queue

- Device has a default `GPUQueue`, which is the only one available now.
- Used to submit commands to the GPU.
- Also has handy helper functions for writing to buffers and textures.
- These are the easiest ways to set the contents of these resources.

```js
device.queue.writeBuffer(buffer, 0, typedArray);
device.queue.writeTexture({ texture: dstTexture },
                          typedArray,
                          { bytesPerRow: 256 },
                          { width: 64, height: 64 });
```

## Recording GPU commands

Command buffers are pre-recorded lists of GPU commands that can be submitted to a `GPUQueue` for execution. Each GPU command represents a task to be performed on the GPU, such as setting state, drawing, copying resources, etc.

- Create a `GPUCommandEncoder` from the device
- Perform copies between buffers/textures
- Begin render or compute passes
- Creates a `GPUCommandBuffer` when finished.
- Command buffers don't do anything until submitted to the queue.
- A `GPUCommandBuffer` can only be submitted once, at which point it becomes invalid. To reuse rendering commands across multiple submissions, use `GPURenderBundle`.

```js
const commandEncoder = device.createCommandEncoder();
commandEncoder.copyBufferToBuffer(bufferA, 0,
                                  bufferB, 0, 256);

const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(128);
passEncoder.end();

const commandBuffer = commandEncoder.finish();
device.queue.submit([commandBuffer]);
```

## Workgroup

```js
dispatch(group_size, group_count)
// group_size => workgroup_size(Sx, Sy, Sz) (similar to thread block)
// group_count => dispatchWorkgroups(Nx, Ny, Nz) (similar to grid)
// Total tasks => (Nx * Ny * Nz * Sx * Sy * Sz)
```

## Debugging WebGPU code

### Label Usage

- Every single object in WebGPU can be given a label, and those labels will be use when reporting error messages.
- Labels have no impact on performance, so there is no reason not to use them!

```js
const vertexBuffer = device.createBuffer({
  label: 'Player vertices',
  size: vertexData.byteLength,
  usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});

const passEncoder = commandEncoder.beginRenderPass({
  label: 'Primary render pass',
  colorAttachments: [{
    view: context.getCurrentTexture().createView(),
    loadOp: 'clear',
    clearValue: [0.0, 0.0, 0.0, 1.0],
    storeOp: 'store',
  }]
});
```

### Debug group usage

- Debug groups are great for telling *where* in the code an error took place.
- They give a personalized stack with for every error that occurs inside them.
- Just like labels, they show up in the native tools as well.
- Plus they're lightweight, so there is no need to worry about stripping them out of the release code.

```js
const commandEncoder = device.createCommandEncoder();
commandEncoder.pushDebugGroup('Main Render Loop');

  commandEncoder.pushDebugGroup('Render Scene');
    renderGameScene(commandEncoder);
  commandEncoder.popDebugGroup();

  commandEncoder.pushDebugGroup('Render UI');
    renderGameUI(commandEncoder);
  commandEncoder.popDebugGroup();

commandEncoder.popDebugGroup();
device.queue.submit([commandEncoder.finish()]);
```

Using error scopes to capture validation errors from a GPUDevice operation that may fail:

```js
gpuDevice.pushErrorScope('validation');

let sampler = gpuDevice.createSampler({
    maxAnisotropy: 0, // Invalid, maxAnisotropy must be at least 1.
});

gpuDevice.popErrorScope().then((error) => {
    if (error) {
        // There was an error creating the sampler, so discard it.
        sampler = null;
        console.error(`An error occured while creating sampler: ${error.message}`);
    }
});
```

## Best practices


1. More pipelines, more state switching, less performance
2. Create pipelines in advance, and don't use them immediately after creation.
Or use the async version. The promise resolves when the pipleline is ready to use without any stalling.

```js
device.createComputePipelineAsync({
 compute: {
   module: shaderModule,
   entryPoint: 'computeMain'
 }
}).then((pipeline) => {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(128);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
});
```

## BindGroup Layout auto

If you use the `auto` layout for the compute pipeline, it will only contain bindings for variables that are directly or transitively referenced by the shader's entry point function. If you don't reference the defined vars, then it won't be added to the automatically generated bind group layout.
One quick way to reference the vars inside the kernel is to add these lines to the top of your entry point:

```
@group(0) @binding(0) var<storage, read_write> results : array<i32>;
@group(0) @binding(1) var<storage, read_write> count : atomic<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(local_invocation_id) threadId: vec3<u32>,
  @builtin(global_invocation_id) globalId : vec3<u32>) {

_ = &results;
_ = &count;

// ...
```

## References

Ordered to be neat!

- Where to run WebGPU: [CanIUse](https://caniuse.com/webgpu)
- WebGPU at [Google IO 2023](https://developer.chrome.com/blog/webgpu-io2023/)
- WebGPU compute [example](https://web.dev/gpu-compute/)
- Awesome list for [WebGPU](https://github.com/mikbry/awesome-webgpu)
- Nikita's great [collection](https://wiki.nikiv.dev/computer-graphics/webgpu)
- Introduction by [Surma](https://surma.dev/things/webgpu/)
- Chrome team [article](https://developer.chrome.com/articles/gpu-compute/)
- WebGPU on [Firefox](https://developer.chrome.com/docs/web-platform/webgpu/)
- WebGPU [explainer](https://gpuweb.github.io/gpuweb/explainer/)
- WebGPU [spec](https://gpuweb.github.io/gpuweb/)
- Andi's [weblog](https://cohost.org/mcc/post/1406157-i-want-to-talk-about-webgpu)

Repos and examples:

- [WebGPT model](https://github.com/0hq/WebGPT/)
- [Native WebGPU](https://github.com/gfx-rs/wgpu-native)
- [WebGPU for rust](https://github.com/gfx-rs/wgpu/)
- [WebGPU in Rust](https://github.com/gfx-rs/wgpu/)
- [WebGPU samples](https://github.com/webgpu/webgpu-samples)
- [Rust WebGPU Wiki](https://github.com/gfx-rs/wgpu/wiki)
- [Rust WebGPU Users](https://github.com/gfx-rs/wgpu/wiki/Users)
- [Rust WebGPU Native](https://github.com/gfx-rs/wgpu-native)
- [WebGPULab examples](https://webgpulab.xbdev.net/)
- [WebGPU for TypeScript](https://github.com/gpuweb/types)
- [An example on YouTube](https://youtu.be/7fiCsG6IILs)
- [Debugging WebGPU Apps](https://github.com/gfx-rs/wgpu/wiki/Debugging-wgpu-Applications)

Other resources:

- [GPU Accelerated JS](https://github.com/gpujs/gpu.js)
- [Khronos WebCL](https://www.khronos.org/webcl/)