/**
 * @file        01-Binding.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        Nov 2022
 *
 * @description Create Binding Group and Binding Layout
 *
 * Create a GPUBindGroupLayout that describes a binding with a uniform buffer, a
 * texture, and a sampler. Then create a GPUBindGroup and a GPUPipelineLayout
 * using the GPUBindGroupLayout.
 * 
*/
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


// In the example below, the bind group layout expects two readonly storage
// buffers at numbered entry bindings 0, 1, and a storage buffer at 2 for the
// compute shader. The bind group on the other hand, defined for this bind group
// layout, associates GPU buffers to the entries: gpuBufferFirstMatrix to the
// binding 0, gpuBufferSecondMatrix to the binding 1, and resultMatrixBuffer to
// the binding 2.

const bindGroupLayout_b = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {type: 'read-only-storage'}
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {type: 'read-only-storage'}
    },
    {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'storage'}}
  ]
});

const bindGroup_b = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {binding: 0, resource: {buffer: gpuBufferFirstMatrix}},
    {binding: 1, resource: {buffer: gpuBufferSecondMatrix}},
    {binding: 2, resource: {buffer: resultMatrixBuffer}}
  ]
});