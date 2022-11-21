# WebGPU-Playground

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

## Pipelines

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

- Create a `GPUCommandEncoder` from the device
- Perform copies between buffers/textures
- Begin render or compute passes
- Creates a `GPUCommandBuffer` when finished.
- Command buffers don't do anything until submitted to the queue.
- Cannot reuse a command buffer after it's been submitted.

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
