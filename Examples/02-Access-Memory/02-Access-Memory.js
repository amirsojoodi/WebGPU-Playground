/**
 * @file        02-Access-Memory.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        Nov 2022
 *
 * @description Accessing GPU's global memory
 *
 * In short, here's what you need to remember regarding buffer memory
 *
 * operations:
 * 1- GPU buffers have to be unmapped to be used in device queue submission.
 * 2- When mapped, GPU buffers can be read and written in JavaScript.
 * 3- GPU buffers are mapped when mapAsync() and createBuffer() with
 * mappedAtCreation set to true are called.
 *
 */


(async () => {
  if (!('gpu' in navigator)) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.log('Failed to get GPU adapter.');
    return;
  }
  const device = await adapter.requestDevice();

  // Get a GPU buffer in a mapped state and an arrayBuffer for writing.
  const gpuWriteBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: 4,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
  });
  const arrayBuffer = gpuWriteBuffer.getMappedRange();

  // Write bytes to buffer.
  new Uint8Array(arrayBuffer).set([0, 1, 2, 3]);

  // At this point, the GPU buffer is mapped, meaning it is owned by the CPU,
  // and it's accessible in read/write from JavaScript. So that the GPU can
  // access it, it has to be unmapped which is as simple as calling
  // gpuBuffer.unmap().

  // Unmap buffer so that it can be used later for copy.
  gpuWriteBuffer.unmap();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    mappedAtCreation: false,
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(
      gpuWriteBuffer /* source buffer */, 0 /* source offset */,
      gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
      4 /* size */
  );

  // Submit copy commands.
  const copyCommands = copyEncoder.finish();
  device.queue.submit([copyCommands]);

  // Read buffer.
  console.time('mapAsync');
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  console.timeEnd('mapAsync');
  const copyArrayBuffer = gpuReadBuffer.getMappedRange();

  console.log(new Uint8Array(copyArrayBuffer));

  gpuReadBuffer.unmap();
})();
