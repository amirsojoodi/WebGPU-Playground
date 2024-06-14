#include <cmath>
#include <cstring>
#include <emscripten.h>
#include <iostream>
#include <vector>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

// Uncomment this flag, or compile with -DDEBUG to enable logging
#define DEBUG 1

#define LOG_MSG(msg) __log_msg(msg)

inline void __log_msg(const char *msg) {
#ifdef DEBUG
  EM_ASM({console.log(msg)});
#else
  // Do nothing!
#endif
}

wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device device;

wgpu::Buffer gpuReadBuffer;
size_t resultMatrixSize;

void GetAdapter(void (*callback)(wgpu::Adapter)) {
  instance.RequestAdapter(
      nullptr,
      // TODO(https://bugs.chromium.org/p/dawn/issues/detail?id=1892): Use
      // wgpu::RequestAdapterStatus and wgpu::Adapter.
      [](WGPURequestAdapterStatus status, WGPUAdapter cAdapter,
         const char *message, void *userdata) {
        if (message) {
          printf("RequestAdapter: %s\n", message);
        }
        if (status != WGPURequestAdapterStatus_Success) {
          exit(0);
        }
        adapter = wgpu::Adapter::Acquire(cAdapter);
        reinterpret_cast<void (*)(wgpu::Adapter)>(userdata)(adapter);
      },
      reinterpret_cast<void *>(callback));
}

void GetDevice(void (*callback)(wgpu::Device)) {
  adapter.RequestDevice(
      nullptr,
      // TODO(https://bugs.chromium.org/p/dawn/issues/detail?id=1892): Use
      // wgpu::RequestDeviceStatus and wgpu::Device.
      [](WGPURequestDeviceStatus status, WGPUDevice cDevice,
         const char *message, void *userdata) {
        if (message) {
          printf("RequestDevice: %s\n", message);
        }
        device = wgpu::Device::Acquire(cDevice);
        device.SetUncapturedErrorCallback(
            [](WGPUErrorType type, const char *message, void *userdata) {
              std::cout << "Error: " << type << " - message: " << message;
            },
            nullptr);
        reinterpret_cast<void (*)(wgpu::Device)>(userdata)(device);
      },
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

void BufferMapCallback(WGPUBufferMapAsyncStatus status, void *userdata) {
  if (status == WGPUBufferMapAsyncStatus_Success) {
    const float *resultData = static_cast<const float *>(
        gpuReadBuffer.GetConstMappedRange(0, resultMatrixSize));

    // std::vector<float> results(resultData, resultData + (resultMatrixSize /
    // sizeof(float)));

    EM_ASM_(
        {
          console.log("Result Matrix: ");
          var resultPtr = $0;
          var size = $1;

          // Create a Float32Array view on the heap
          var resultArray =
              new Float32Array(Module.HEAPF32.buffer, resultPtr, size / 4);

          // Log the contents of the array
          console.log(resultArray);
        },
        resultData, resultMatrixSize);

    gpuReadBuffer.Unmap();
  } else {
    LOG_MSG("Failed to map result buffer");
  }
}

void RunMatMult() {
  // First Matrix
  const float firstMatrix[] = {2, 4, 1, 2, 3, 4, 5, 6, 7, 8};
  size_t firstMatrixSize = sizeof(firstMatrix);

  wgpu::Buffer gpuBufferFirstMatrix =
      device.CreateBuffer(new wgpu::BufferDescriptor{
          .usage = wgpu::BufferUsage::Storage,
          .size = firstMatrixSize,
          .mappedAtCreation = true,
      });
  std::memcpy(gpuBufferFirstMatrix.GetMappedRange(), firstMatrix,
              firstMatrixSize);
  gpuBufferFirstMatrix.Unmap();

  EM_ASM({ console.log("Initialized the first matrix"); });
  EM_ASM_(
      {
        console.log("First Matrix: ");
        var resultPtr = $0;
        var size = $1;

        // Create a Float32Array view on the heap
        var resultArray =
            new Float32Array(Module.HEAPF32.buffer, resultPtr, size / 4);

        // Log the contents of the array
        console.log(resultArray);
      },
      firstMatrix, firstMatrixSize);

  // Second Matrix
  const float secondMatrix[] = {4, 2, 1, 2, 3, 4, 5, 6, 7, 8};
  size_t secondMatrixSize = sizeof(secondMatrix);

  wgpu::Buffer gpuBufferSecondMatrix =
      device.CreateBuffer(new wgpu::BufferDescriptor{
          .usage = wgpu::BufferUsage::Storage,
          .size = secondMatrixSize,
          .mappedAtCreation = true,
      });
  std::memcpy(gpuBufferSecondMatrix.GetMappedRange(), secondMatrix,
              secondMatrixSize);
  gpuBufferSecondMatrix.Unmap();

  EM_ASM({ console.log("Initialized the second matrix"); });
  EM_ASM_(
      {
        console.log("First Matrix: ");
        var resultPtr = $0;
        var size = $1;

        // Create a Float32Array view on the heap
        var resultArray =
            new Float32Array(Module.HEAPF32.buffer, resultPtr, size / 4);

        // Log the contents of the array
        console.log(resultArray);
      },
      secondMatrix, secondMatrixSize);

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

  EM_ASM({ console.log("Commands submitted to the GPU Queue"); });

  // Read buffer
  gpuReadBuffer.MapAsync(wgpu::MapMode::Read, (size_t)0, resultMatrixSize,
                         BufferMapCallback, NULL);
}

int main() {

  instance = wgpu::CreateInstance();

  GetAdapter([](wgpu::Adapter adapter) {
    EM_ASM({ console.log("GPU Adapter acquired."); });
    GetDevice([](wgpu::Device device) {
      EM_ASM({ console.log("GPU Device acquired."); });
      RunMatMult();
    });
  });

  return 0;
}
