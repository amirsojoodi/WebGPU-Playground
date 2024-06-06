/**
 * @file        10-f16.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        June 2024
 *
 * @description Testing f16-shader feature
 *
 * This code measures the execution time of a matrix multiplication, in f16 data
 * format. The feature has to be explicitly requested as of June 2024.
 *
 * Chrome should be run with the flag: --enable-dawn-features=allow_unsafe_apis
 * Make sure all of the chrome windows/tabs are closed so that a new session
 * can start with the specified flags.
 *
 */

(async () => {
  const BLOCK_SIZE_X = 16;
  const BLOCK_SIZE_Y = 16;
  const timestamp_capacity = 3;

  let device = null;
  let adapter = null;
  let matrixWidth;

  let firstMatrix;
  let secondMatrix;

  let gpuBufferFirstMatrix;
  let gpuBufferSecondMatrix;

  let arrayBufferFirstMatrix;
  let arrayBufferSecondMatrix;
  let resultMatrixBufferSize;
  let resultMatrixBuffer;
  let gpuReadBuffer;

  let isUsingTimeStamps;
  let queryBuffer;
  let queryReadBuffer;

  async function initializeWebGPU() {
    if (!('gpu' in navigator)) {
      console.log(
          'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
      return;
    }

    adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log('Failed to get GPU adapter.');
      return;
    }

    const features = [];

    if (adapter.features.has('timestamp-query')) {
      features.push('timestamp-query');
      // features.push('chromium-experimental-timestamp-query-inside-passes');
      isUsingTimeStamps = true;
    } else {
      console.log('The GPU adapter doesn\'t support timestamp query');
      isUsingTimeStamps = false;
    }

    // Exit if shader-f16 is not supported
    if (adapter.features.has('shader-f16')) {
      features.push('shader-f16');
    } else {
      console.log('The GPU adapter doesn\'t support shader-f16');
      return false;
    }

    device = await adapter.requestDevice(
        {powerPreference: 'high-performance', requiredFeatures: features});

    // requestDevice will never return null, but if a valid device request
    // can't be fulfilled for some reason it may resolve to a device which has
    // already been lost. Additionally, devices can be lost at any time after
    // creation for a variety of reasons (ie: browser resource management,
    // driver updates), so it's a good idea to always handle lost devices
    // gracefully.
    device.lost.then((info) => {
      console.error(`WebGPU device was lost: ${info.message}`);

      device = null;

      if (info.reason != 'destroyed') {
        initializeWebGPU();
      }
    });

    return true;
  }

  async function getAdapterInfo() {
    const adapterInfo = await (
        adapter.requestAdapterInfo ? adapter.requestAdapterInfo() : undefined);
    if (adapterInfo === undefined) {
      return 'Warning: cannot get adapterInfo';
    }

    const packedAdapterInfo =
        `IsCompatibilityMode: ${adapter.isCompatibilityMode}\n` +
        `Vendor: ${adapterInfo.vendor}\n` +
        `Architecture: ${adapterInfo.vendor}\n` +
        `Backend: ${adapterInfo.backend}\n` +
        `Description: ${adapterInfo.description}\n` +
        `Type: ${adapterInfo.type}`;

    return packedAdapterInfo;
  }

  async function initDeviceData(matrix_width) {
    // Matrices are like as follows:
    // [rows, columns, (values)]);
    matrixWidth = matrix_width;

    firstMatrix = new Float16Array((matrixWidth * matrixWidth) + 2);
    firstMatrix[0] = matrixWidth;
    firstMatrix[1] = matrixWidth;

    for (let i = 2; i < matrixWidth * matrixWidth + 2; i++) {
      firstMatrix[i] = 1;
    }

    gpuBufferFirstMatrix = device.createBuffer({
      mappedAtCreation: true,
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();

    new Float16Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();


    secondMatrix = new Float16Array((matrixWidth * matrixWidth) + 2);
    secondMatrix[0] = matrixWidth;
    secondMatrix[1] = matrixWidth;

    for (let i = 2; i < matrixWidth * matrixWidth + 2; i++) {
      secondMatrix[i] = 1;
    }

    gpuBufferSecondMatrix = device.createBuffer({
      mappedAtCreation: true,
      size: secondMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
    new Float16Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();

    resultMatrixBufferSize =
        Float16Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
    resultMatrixBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Get a GPU buffer for reading in an unmapped state.
    gpuReadBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    if (isUsingTimeStamps) {
      queryBuffer = device.createBuffer({
        size: 8 * timestamp_capacity,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
      });

      queryReadBuffer = device.createBuffer({
        size: 8 * timestamp_capacity,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      })
    }
  }

  async function benchmark() {
    const shaderModule = device.createShaderModule({
      code: `
        enable f16;

        const BLOCK_SIZE_X = 16;
        const BLOCK_SIZE_Y = 16;

        struct Matrix {
          size : vec2<f16>,
          numbers: array<f16>,
        }
  
        @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
        @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
        @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
  
        @compute @workgroup_size(BLOCK_SIZE_X, BLOCK_SIZE_Y)
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

    const computePipeline = device.createComputePipeline(
        {layout: 'auto', compute: {module: shaderModule, entryPoint: 'main'}});

    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0 /* index */),
      entries: [
        {binding: 0, resource: {buffer: gpuBufferFirstMatrix}},
        {binding: 1, resource: {buffer: gpuBufferSecondMatrix}},
        {binding: 2, resource: {buffer: resultMatrixBuffer}}
      ]
    });


    let querySet;

    if (isUsingTimeStamps) {
      // Create a timestamp query set that will store the timestamp values.
      querySet =
          device.createQuerySet({type: 'timestamp', count: timestamp_capacity});
    }

    const commandEncoder = device.createCommandEncoder();

    // Initial timestamp
    // Not: writeTimeStamp cannot be called during a pass. ( between calls to
    // beginComputePass(); and passEncoder.end(); )
    if (isUsingTimeStamps) {
      commandEncoder.writeTimestamp(querySet, 0);
    }

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroupCountX = Math.ceil(firstMatrix[0] / BLOCK_SIZE_X);
    const workgroupCountY = Math.ceil(secondMatrix[1] / BLOCK_SIZE_Y);
    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    passEncoder.end();

    if (isUsingTimeStamps) {
      commandEncoder.writeTimestamp(querySet, 1);
    }

    commandEncoder.copyBufferToBuffer(
        resultMatrixBuffer /* source buffer */, 0 /* source offset */,
        gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
        resultMatrixBufferSize /* size */
    );

    if (isUsingTimeStamps) {
      commandEncoder.writeTimestamp(querySet, 2);
    }

    // Now after marking all the required timestamps, we can submit their
    // resolve
    if (isUsingTimeStamps) {
      // Retrieve the timestamps, it has to be done before
      // commandEncoder.finish()
      commandEncoder.resolveQuerySet(
          querySet, 0, timestamp_capacity, queryBuffer, 0)
      commandEncoder.copyBufferToBuffer(
          queryBuffer, 0, queryReadBuffer, 0, timestamp_capacity * 8)
    }

    // Now finalizing the GPU commands, ready for submission.
    const gpuCommands = commandEncoder.finish();

    // From the CPU perspective, this is a good place to start measuring
    // GPU time (write before queue.submit call) Anything before here is
    // GPU work preparation. If we have device.queue.writeBuffer calls,
    // as they are synchronous, they will not be included in this measurement.
    const start = performance.now();

    device.queue.submit([gpuCommands]);

    await device.queue.onSubmittedWorkDone();
    const GPU_compute_duration = performance.now() - start;

    // By now we are sure that the matmult operation is done. However,
    // we cannot readily use the results. So, we can kind of differentiate
    // between GPU compute time and GPU total execution time (ignoring
    // the Host-to-Device data transfer time)
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const GPU_total_duration = performance.now() - start;

    // It's unnecessary for the purpose of this benchmark to grab the results.
    // Unless we want to validate the returned result.
    // const arrayBuffer = gpuReadBuffer.getMappedRange();
    // const resultBuffer = new Float16Array(arrayBuffer);
    // For debugging purposes
    // console.log(resultBuffer);

    let logMessage = `Matrix width: ${matrixWidth} \n` +
        `GPU_total_duration: ${GPU_total_duration} ms \n` +
        `GPU_compute_duration: ${GPU_compute_duration} ms \n`;

    // === After `commandEncoder.finish()` is called ===
    // We can now read the storage query buffer (timestamps) and decode
    // it into an array of timestamps in nanoseconds
    if (isUsingTimeStamps) {
      await queryReadBuffer.mapAsync(GPUMapMode.READ);

      const timeStamps =
          new BigUint64Array(queryReadBuffer.getMappedRange()).slice();

      logMessage += `Timestamps: Total Elapsed time for MatrixMultiplication: ${
                        Number(timeStamps[2] - timeStamps[0]) / 1e6} ms \n` +
          `Timestamps: Matrix multiplication on the device took: ${
                        Number(timeStamps[1] - timeStamps[0]) / 1e6} ms \n` +
          `Timestamps: Transfering data from device to host took: ${
                        Number(timeStamps[2] - timeStamps[1]) / 1e6} ms \n`;
    }

    return logMessage;
  }

  async function cleanup() {
    // Freeing buffers
    gpuReadBuffer.unmap();
    queryReadBuffer.unmap();

    gpuBufferFirstMatrix.destroy();
    gpuBufferSecondMatrix.destroy();
    resultMatrixBuffer.destroy();

    if (isUsingTimeStamps) {
      queryBuffer.destroy();
      queryReadBuffer.destroy();
    }
    // This better not be called in prod
    // device.destroy();
  }

  if (await initializeWebGPU() == true) {
    await initDeviceData(2000);
    console.log(await benchmark());
    await cleanup();
  }
})();
