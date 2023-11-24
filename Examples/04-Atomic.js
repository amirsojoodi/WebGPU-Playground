/**
 * @file        04-Atomic.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        Jan 2023
 *
 * @description Atomic operations in WGSL
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

  // First Matrix

  const firstMatrix =
      new Float32Array([2 /* rows */, 4 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8]);

  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();

  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();


  // Second Matrix

  const secondMatrix =
      new Float32Array([4 /* rows */, 2 /* columns */, 1, 2, 3, 4, 5, 6, 7, 8]);

  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();


  // Result Matrix

  const resultMatrixBufferSize =
      Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Elements counts to test atomic operations

  const countBuffer = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Compute shader code

  const shaderModule = device.createShaderModule({
    code: `
        struct Matrix {
          size : vec2<f32>,
          numbers: array<f32>,
        }
  
        @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
        @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
        @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
        @group(0) @binding(3) var<storage, read_write> count : atomic<u32>;
  
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
          
          // Initialize the counter
          if ((globalId.x == 0) && (globalId.y == 0)) {
            atomicStore(&count, 0);
          }        
          
          // Guard against out-of-bounds work group sizes
          if (globalId.x >= u32(firstMatrix.size.x) || globalId.y >= u32(secondMatrix.size.y)) {
            return;
          }
  
          resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);
  
          let resultCell = vec2(globalId.x, globalId.y);
          var result = 0.0;
          for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
            let a = i + resultCell.x * u32(firstMatrix.size.y);
            let b = resultCell.y + i * u32(secondMatrix.size.y);
            result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
          }
          
          // Out of 64 threads in the work group (8*8) only 4 of them make it to here, so count should be 4
          atomicAdd(&count, 1);

          let index = resultCell.y + resultCell.x * u32(secondMatrix.size.y);
          resultMatrix.numbers[index] = result;
        }
      `
  });

  // Pipeline setup

  const computePipeline = device.createComputePipeline(
      {layout: 'auto', compute: {module: shaderModule, entryPoint: 'main'}});


  // Bind group

  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0 /* index */),
    entries: [
      {binding: 0, resource: {buffer: gpuBufferFirstMatrix}},
      {binding: 1, resource: {buffer: gpuBufferSecondMatrix}},
      {binding: 2, resource: {buffer: resultMatrixBuffer}},
      {binding: 3, resource: {buffer: countBuffer}}
    ]
  });


  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroupCountX = Math.ceil(firstMatrix[0] / 8);
  const workgroupCountY = Math.ceil(secondMatrix[1] / 8);
  console.log(workgroupCountX, workgroupCountY);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const gpuCountBuffer = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */, 0 /* source offset */,
      gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
      resultMatrixBufferSize /* size */
  );

  commandEncoder.copyBufferToBuffer(
      countBuffer, 0, gpuCountBuffer, 0, Uint32Array.BYTES_PER_ELEMENT)

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);


  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  await gpuCountBuffer.mapAsync(GPUMapMode.READ);

  const arrayBuffer = gpuReadBuffer.getMappedRange();
  const countArrayBuffer = gpuCountBuffer.getMappedRange();

  console.log(new Float32Array(arrayBuffer));
  console.log(new Uint32Array(countArrayBuffer));
  // Freeing buffer
  gpuReadBuffer.unmap();
  gpuCountBuffer.unmap();
})();
