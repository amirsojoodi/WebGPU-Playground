// Create a GPUBindGroupLayout that describes a binding with a uniform buffer, a
// texture, and a sampler. Then create a GPUBindGroup and a GPUPipelineLayout
// using the GPUBindGroupLayout.

const bindGroupLayout = gpuDevice.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: {}
    },
    {binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {}},
    {binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {}}
  ]
});

const bindGroup = gpuDevice.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: {buffer: buffer},
    },
    {binding: 1, resource: texture}, {binding: 2, resource: sampler}
  ]
});

const pipelineLayout =
    gpuDevice.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});
