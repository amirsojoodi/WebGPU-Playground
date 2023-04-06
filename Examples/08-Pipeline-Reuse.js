(async () => {
  // Check for WebGPU support
  if (!navigator.gpu) {
    console.error('WebGPU is not supported!');
    return;
  }

  // Get GPU device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Create input data arrays
  const inputData1 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const inputData2 = new Float32Array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  const bufferSize = inputData1.byteLength;

  // Create input/output GPU buffers
  const [gpuInputBuffer, gpuOutputBuffer] = [
    inputData1, new Float32Array(inputData1.length)
  ].map(data => device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    mappedAtCreation: data ? true : false
  }));

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

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {binding: 0, resource: {buffer: uniformBuffer}},
      {binding: 1, resource: {buffer: gpuInputBuffer}},
      {binding: 2, resource: {buffer: gpuOutputBuffer}}
    ]
  });

  // Function to perform computation on a set of input data
  const compute = async inputData => {
    // Write inputData to gpuInputBuffer
    gpuInputBuffer.unmap();
    gpuOutputBuffer.unmap();
    device.queue.writeBuffer(gpuInputBuffer, 0, inputData);

    // Create command encoder
    const commandEncoder = device.createCommandEncoder();

    // Set up compute pass
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(inputData.length);
    computePass.end();

    // Copy output buffer to CPU
    const gpuReadBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    commandEncoder.copyBufferToBuffer(
        gpuOutputBuffer, 0, gpuReadBuffer, 0, bufferSize);

    // Submit commands
    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);
    await device.queue.onSubmittedWorkDone();

    // Read back the output data
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const outputArrayBuffer = gpuReadBuffer.getMappedRange();
    const outputData = new Float32Array(outputArrayBuffer);

    // Log the input and output data
    console.log('Input data:', inputData);
    console.log('Output data:', outputData);

    // Cleanup
    gpuReadBuffer.unmap();
    gpuReadBuffer.destroy();
  };

  // Perform computations on inputData1 and inputData2 using the same GPU
  // buffers
  await compute(inputData1);
  await compute(inputData2);

  // Cleanup
  gpuInputBuffer.destroy();
  gpuOutputBuffer.destroy();
  uniformBuffer.destroy();
})();
