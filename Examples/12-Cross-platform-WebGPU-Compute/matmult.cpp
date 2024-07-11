#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>
#include <webgpu/webgpu_cpp.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device device;

wgpu::Buffer gpuReadBuffer;
size_t resultMatrixSize;
bool work_done = false;

// GetAdapter gets a callback function that it's get called
// after the RequestAdapter resolves.
void GetAdapter(void (*callback)()) {
  instance.RequestAdapter(
      nullptr,
      [](WGPURequestAdapterStatus status, WGPUAdapter cAdapter,
         const char *message, void *userdata) {
        if (message) {
          std::cout << "RequestAdapter message: " << message << std::endl;
        }
        if (status != WGPURequestAdapterStatus_Success) {
          std::cout << "AdapterRequest was not successfull" << std::endl;
          exit(0);
        }
        adapter = wgpu::Adapter::Acquire(cAdapter);
        // (2) Cast userdata back to the callback and then call it
        reinterpret_cast<void (*)()>(userdata)();
      },
      // (1) Cast the call back to void pointer and pass it in
      reinterpret_cast<void *>(callback));
}

void GetDevice(void (*callback)()) {
  adapter.RequestDevice(
      nullptr,
      [](WGPURequestDeviceStatus status, WGPUDevice cDevice,
         const char *message, void *userdata) {
        if (message) {
          std::cout << "RequestDevice message: " << message << std::endl;
        }
        if (status != WGPURequestDeviceStatus_Success) {
          std::cout << "AdapterRequest was not successfull" << std::endl;
          exit(0);
        }
        device = wgpu::Device::Acquire(cDevice);
        device.SetUncapturedErrorCallback(
            [](WGPUErrorType type, const char *message, void *userdata) {
              std::cout << "Error: " << type << " - message: " << message;
            },
            nullptr);
        // (2) Cast userdata back to the callback and then call it
        reinterpret_cast<void (*)()>(userdata)();
      },
      // (1) Cast the call back to void pointer and pass it in
      reinterpret_cast<void *>(callback));
}

const char shaderCode[] = R"(
    struct Matrix {
        size : vec2<f32>,
        numbers: array<f32>,
    };

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
)";

void BufferMapCallbackFunction(WGPUBufferMapAsyncStatus status,
                               void *userdata) {

  std::cout << "In Buffer async call back, status: " << status << std::endl;

  if (status == WGPUBufferMapAsyncStatus_Success) {
    const float *resultData = static_cast<const float *>(
        gpuReadBuffer.GetConstMappedRange(0, resultMatrixSize));

    std::cout << "Result Matrix: " << std::endl;
    for (int i = 0; i < resultMatrixSize / sizeof(float); i++) {
      std::cout << resultData[i] << " ";
    }
    std::cout << std::endl;

    gpuReadBuffer.Unmap();
  } else {
    std::cout << "Failed to map result buffer" << std::endl;
  }
  *reinterpret_cast<bool *>(userdata) = true;
}

void RunMatMult() {
  // First Matrix
  const std::vector<float> firstMatrix = {2, 4, 1, 2, 3, 4, 5, 6, 7, 8};
  size_t firstMatrixSize = firstMatrix.size() * sizeof(float);

  wgpu::Buffer gpuBufferFirstMatrix =
      device.CreateBuffer(new wgpu::BufferDescriptor{
          .usage = wgpu::BufferUsage::Storage,
          .size = firstMatrixSize,
          .mappedAtCreation = true,
      });
  std::memcpy(gpuBufferFirstMatrix.GetMappedRange(), firstMatrix.data(),
              firstMatrixSize);
  gpuBufferFirstMatrix.Unmap();

  std::cout << "First Matrix: " << std::endl;
  std::copy(firstMatrix.begin(), firstMatrix.end(),
            std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;

  // Second Matrix
  const std::vector<float> secondMatrix = {4, 2, 1, 2, 3, 4, 5, 6, 7, 8};
  size_t secondMatrixSize = secondMatrix.size() * sizeof(float);

  wgpu::Buffer gpuBufferSecondMatrix =
      device.CreateBuffer(new wgpu::BufferDescriptor{
          .usage = wgpu::BufferUsage::Storage,
          .size = secondMatrixSize,
          .mappedAtCreation = true,
      });
  std::memcpy(gpuBufferSecondMatrix.GetMappedRange(), secondMatrix.data(),
              secondMatrixSize);
  gpuBufferSecondMatrix.Unmap();

  std::cout << "Second Matrix: " << std::endl;
  std::copy(secondMatrix.begin(), secondMatrix.end(),
            std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;

  // Result Matrix
  resultMatrixSize =
      sizeof(float) * (2 + static_cast<size_t>(firstMatrix[0]) *
                               static_cast<size_t>(secondMatrix[1]));

  wgpu::Buffer resultMatrixBuffer =
      device.CreateBuffer(new wgpu::BufferDescriptor{
          .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
          .size = resultMatrixSize,
      });

  // Compute shader code
  wgpu::ShaderModuleWGSLDescriptor shaderModuleDesc = {};
  shaderModuleDesc.code = shaderCode;
  wgpu::ShaderModuleDescriptor shaderModuleDescriptor{.nextInChain =
                                                          &shaderModuleDesc};
  wgpu::ShaderModule shaderModule =
      device.CreateShaderModule(&shaderModuleDescriptor);
  // Pipeline setup
  wgpu::ComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.compute.module = shaderModule;
  pipelineDesc.compute.entryPoint = "main";

  wgpu::ComputePipeline computePipeline =
      device.CreateComputePipeline(&pipelineDesc);

  // Bind group
  wgpu::BindGroupDescriptor bindGroupDesc = {};
  wgpu::BindGroupEntry entries[3] = {};
  entries[0].binding = 0;
  entries[0].buffer = gpuBufferFirstMatrix;
  entries[1].binding = 1;
  entries[1].buffer = gpuBufferSecondMatrix;
  entries[2].binding = 2;
  entries[2].buffer = resultMatrixBuffer;
  bindGroupDesc.entryCount = 3;
  bindGroupDesc.entries = entries;
  bindGroupDesc.layout = computePipeline.GetBindGroupLayout(0);

  wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

  // Commands submission
  wgpu::CommandEncoder commandEncoder = device.CreateCommandEncoder();

  wgpu::ComputePassEncoder passEncoder = commandEncoder.BeginComputePass();
  passEncoder.SetPipeline(computePipeline);
  passEncoder.SetBindGroup(0, bindGroup);
  uint32_t workgroupCountX =
      static_cast<uint32_t>(std::ceil(firstMatrix[0] / 8.0f));
  uint32_t workgroupCountY =
      static_cast<uint32_t>(std::ceil(secondMatrix[1] / 8.0f));
  passEncoder.DispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.End();

  // Get a GPU buffer for reading in an unmapped state
  gpuReadBuffer = device.CreateBuffer(new wgpu::BufferDescriptor{
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
      .size = resultMatrixSize,
  });

  // Encode commands for copying buffer to buffer
  commandEncoder.CopyBufferToBuffer(resultMatrixBuffer, 0, gpuReadBuffer, 0,
                                    resultMatrixSize);

  // Submit GPU commands
  wgpu::CommandBuffer commands = commandEncoder.Finish();
  device.GetQueue().Submit(1, &commands);

  std::cout << "Commands submitted to the GPU Queue" << std::endl;

  gpuReadBuffer.MapAsync(wgpu::MapMode::Read, (size_t)0, resultMatrixSize,
                         BufferMapCallbackFunction,
                         reinterpret_cast<void *>(&work_done));
}

extern "C" {
void RunMatMultWrapper() {
  instance = wgpu::CreateInstance();

  GetAdapter([]() {
    std::cout << "GPU Adapter acquired." << std::endl;
    GetDevice([]() {
      std::cout << "GPU Device acquired." << std::endl;
      RunMatMult();
    });
  });

  // https://eliemichel.github.io/LearnWebGPU/getting-started/the-command-queue.html#device-polling
#ifndef __EMSCRIPTEN__
  while (!work_done) {
    instance.ProcessEvents();
  }
#else
  while (!work_done) {
    emscripten_sleep(100);
  }
#endif
}
}

int main() {
  // I put the call to the RunMatMult function in a wrapper, so that
  // I can pass arguements if necessary
  RunMatMultWrapper();
  return 0;
}
