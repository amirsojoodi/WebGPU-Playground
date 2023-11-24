/**
 * @file        09-Timing.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        November 2023
 *
 * @description Timestamp feature of WebGPU
*
* This code measures the total time execution of a matrix multiplication, which
* includes the computation time and the transfer data back from device to host
 *
 * Chrome should be run with the flag: --enable-dawn-features=allow_unsafe_apis
 * Make sure all of the chrome windows/tabs are closed so that a new session
 * can start with the specified flags.
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

  const features = [];


  if (adapter.features.has('timestamp-query')) {
    features.push('timestamp-query');
    // features.push('chromium-experimental-timestamp-query-inside-passes');
  } else {
    console.log('The GPU adapter doesn\'t support timestamp query');
    return;
  }

  const device = await adapter.requestDevice({requiredFeatures: features});
  // or:
  // device = await adapter.requestDevice({
  //     features: ['timestamp-query'],
  // });

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

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
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
  
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
          // Guard against out-of-bounds work group sizes
          if (global_id.x >= u32(firstMatrix.size.x) || global_id.y >= u32(secondMatrix.size.y)) {
            return;
          }
  
          resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);
  
          let resultCell = vec2(global_id.x, global_id.y);
          var result = 0.0;
          for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
            let a = i + resultCell.x * u32(firstMatrix.size.y);
            let b = resultCell.y + i * u32(secondMatrix.size.y);
            result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
          }
  
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
      {binding: 2, resource: {buffer: resultMatrixBuffer}}
    ]
  });


  // Commands submission
  const commandEncoder = device.createCommandEncoder();


  const capacity = 3;  // The number of timestamps we want to store

  // Create a timestamp query set that will store the timestamp values.
  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: capacity,  // You can adjust the number of timestamps as needed
  });

  const queryBuffer = device.createBuffer({
    size: 8 * capacity,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
  });

  const queryReadBuffer = device.createBuffer({
    size: 8 * capacity,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  // Initial timestamp
  // Not: writeTimeStamp cannot be called during a pass. ( between calls to
  // beginComputePass(); and passEncoder.end(); )
  commandEncoder.writeTimestamp(querySet, 0);

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroupCountX = Math.ceil(firstMatrix[0] / 8);
  const workgroupCountY = Math.ceil(secondMatrix[1] / 8);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.end();

  commandEncoder.writeTimestamp(querySet, 1);

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */, 0 /* source offset */,
      gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
      resultMatrixBufferSize /* size */
  );

  commandEncoder.writeTimestamp(querySet, 2);

  // Retrieve the timestamps, it has to be done before commandEncoder.finish()
  commandEncoder.resolveQuerySet(querySet, 0, capacity, queryBuffer, 0)
  commandEncoder.copyBufferToBuffer(
      queryBuffer, 0, queryReadBuffer, 0, capacity * 8)

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);

  // Read results
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  console.log(new Float32Array(arrayBuffer));

  // === After `commandEncoder.finish()` is called ===
  // Read the storage query buffer
  // Decode it into an array of timestamps in nanoseconds
  await queryReadBuffer.mapAsync(GPUMapMode.READ);
  const timestamps =
      new BigUint64Array(queryReadBuffer.getMappedRange()).slice();

  console.log(`Total Elapsed time for MatrixMultiplication: ${
      Number(timestamps[2] - timestamps[0]) / 1000.0} us`);
  console.log(`Matrix multiplication on the device took: ${
      Number(timestamps[1] - timestamps[0]) / 1000.0} us`);
  console.log(`Transfering data from device to host took: ${
      Number(timestamps[2] - timestamps[1]) / 1000.0} us`);

  // Freeing buffer
  gpuReadBuffer.unmap();
  queryReadBuffer.unmap();
})();
