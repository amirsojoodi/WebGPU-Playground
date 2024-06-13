/**
 * @file        07-Pipelined.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        April 2023
 *
 * @description Pipelining the workload
 *
 */

(async () => {
  // Check for WebGPU support
  if (!navigator.gpu) {
    console.error('WebGPU is not supported!');
    return;
  }

  // Get GPU device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Create input data array
  const inputData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const bufferSize = inputData.byteLength;
  const halfLength = Math.ceil(inputData.length / 2);

  // Function to create a buffer with the specified data
  const createBufferWithData = (data, usage) => {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  };

  // Split input data into two halves and create buffers
  const inputDataFirstHalf = inputData.subarray(0, halfLength);
  const inputDataSecondHalf = inputData.subarray(halfLength);
  const gpuInputBufferFirstHalf = createBufferWithData(
      inputDataFirstHalf,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST);
  const gpuInputBufferSecondHalf = createBufferWithData(
      inputDataSecondHalf,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST);

  // Create output GPU buffers for both halves
  const createOutputBuffer = () => device.createBuffer({
    size: bufferSize / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
  });
  const gpuOutputBufferFirstHalf = createOutputBuffer();
  const gpuOutputBufferSecondHalf = createOutputBuffer();

  // Create uniform buffer
  const multiplier = 2.0;
  const uniformBuffer = device.createBuffer(
      {size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([multiplier]));

  // Create shader module
  const shaderModule = device.createShaderModule({
    code: `
      struct Uniforms {
        multiplier: f32
      }
      
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> inputBuffer: array<f32>;
      @group(0) @binding(2) var<storage, read_write> outputBuffer: array<f32>;
  
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
        let index: u32 = globalId.x;
        outputBuffer[index] = inputBuffer[index] * uniforms.multiplier;
      }
    `,
  });

  // Create pipeline
  const pipeline = device.createComputePipeline(
      {layout: 'auto', compute: {module: shaderModule, entryPoint: 'main'}});

  // Function to create bind groups
  const createBindGroup = (inputBuffer, outputBuffer) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {binding: 0, resource: {buffer: uniformBuffer}},
          {binding: 1, resource: {buffer: inputBuffer}},
          {binding: 2, resource: {buffer: outputBuffer}},
        ],
      });

  // Create bind groups for both halves
  const bindGroupFirstHalf =
      createBindGroup(gpuInputBufferFirstHalf, gpuOutputBufferFirstHalf);
  const bindGroupSecondHalf =
      createBindGroup(gpuInputBufferSecondHalf, gpuOutputBufferSecondHalf);

  // Function to encode and submit commands for computing one half
  const computeHalf = async (bindGroup, outputBuffer, dispatchSize) => {
    const commandEncoder = device.createCommandEncoder();
    // Set up compute pass
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(dispatchSize);
    computePass.end();

    // Copy output buffer to CPU
    const gpuReadBuffer = device.createBuffer({
      size: outputBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    commandEncoder.copyBufferToBuffer(
        outputBuffer, 0, gpuReadBuffer, 0, outputBuffer.size);

    // Submit commands
    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);
    await device.queue.onSubmittedWorkDone();

    return gpuReadBuffer;
  };

  // Compute both halves concurrently
  const firstHalfPromise =
      computeHalf(bindGroupFirstHalf, gpuOutputBufferFirstHalf, halfLength);
  const secondHalfPromise = computeHalf(
      bindGroupSecondHalf, gpuOutputBufferSecondHalf,
      inputData.length - halfLength);

  // Wait for both halves to complete
  const gpuOutputBufferFirstHalfResult = await firstHalfPromise;
  const gpuOutputBufferSecondHalfResult = await secondHalfPromise;

  // Function to read back the output data
  const readOutputData = async (gpuOutputBuffer) => {
    await gpuOutputBuffer.mapAsync(GPUMapMode.READ);
    const outputArrayBuffer = gpuOutputBuffer.getMappedRange();
    const outputData = new Float32Array(outputArrayBuffer);
    return outputData;
  };

  // Read back the output data for both halves
  const outputDataFirstHalf =
      await readOutputData(gpuOutputBufferFirstHalfResult);
  const outputDataSecondHalf =
      await readOutputData(gpuOutputBufferSecondHalfResult);

  // Combine output data
  const outputData = new Float32Array(inputData.length);
  outputData.set(outputDataFirstHalf);
  outputData.set(outputDataSecondHalf, halfLength);

  // Log the output data
  console.log('Input data:', inputData);
  console.log('Output data:', outputData);

  // Cleanup
  gpuInputBufferFirstHalf.destroy();
  gpuInputBufferSecondHalf.destroy();
  gpuOutputBufferFirstHalf.destroy();
  gpuOutputBufferSecondHalf.destroy();
  uniformBuffer.destroy();
})();