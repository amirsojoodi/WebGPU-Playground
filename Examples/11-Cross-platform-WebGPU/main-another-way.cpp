#include <GLFW/glfw3.h>
#include <iostream>
#include <webgpu/webgpu_cpp.h>
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#else
#include <webgpu/webgpu_glfw.h>
#endif

wgpu::Instance instance;
wgpu::Adapter adapter;
wgpu::Device device;
wgpu::RenderPipeline pipeline;
wgpu::SwapChain swapChain;

wgpu::Surface surface;
wgpu::TextureFormat format = wgpu::TextureFormat::BGRA8Unorm;
const uint32_t kWidth = 512;
const uint32_t kHeight = 512;

void ConfigureSwapChain() {
  wgpu::SwapChainDescriptor swapChainDesc{};
  swapChainDesc.usage = wgpu::TextureUsage::RenderAttachment;
  swapChainDesc.format = format;
  swapChainDesc.width = kWidth;
  swapChainDesc.height = kHeight;
  swapChainDesc.presentMode = wgpu::PresentMode::Fifo;

  swapChain = device.CreateSwapChain(surface, &swapChainDesc);
}

void GetAdapter(void (*callback)(wgpu::Adapter)) {
  instance.RequestAdapter(
      nullptr,
      // TODO: Use wgpu::RequestAdapterStatus and wgpu::Adapter when they are
      // available
      [](WGPURequestAdapterStatus status, WGPUAdapter cAdapter,
         const char *message, void *userdata) {
        if (message) {
          printf("RequestAdapter: %s\n", message);
        }
        if (status != WGPURequestAdapterStatus_Success) {
          exit(0);
        }
        wgpu::Adapter adapter = wgpu::Adapter::Acquire(cAdapter);
        reinterpret_cast<void (*)(wgpu::Adapter)>(userdata)(adapter);
      },
      reinterpret_cast<void *>(callback));
}

void GetDevice(void (*callback)(wgpu::Device)) {
  adapter.RequestDevice(
      nullptr,
      // TODO: Use wgpu::RequestDeviceStatus and wgpu::Device when they are
      // available
      [](WGPURequestDeviceStatus status, WGPUDevice cDevice,
         const char *message, void *userdata) {
        if (message) {
          printf("RequestDevice: %s\n", message);
        }
        wgpu::Device device = wgpu::Device::Acquire(cDevice);
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
    @vertex fn vertexMain(@builtin(vertex_index) i : u32) ->
      @builtin(position) vec4f {
        const pos = array<vec2f, 3>(
          vec2f(0, 1), vec2f(-1, -1), vec2f(1, -1)
        );
        return vec4f(pos[i], 0, 1);
    }
    @fragment fn fragmentMain() -> @location(0) vec4f {
        return vec4f(1, 0, 0, 1);
    }
)";

void CreateRenderPipeline() {
  wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
  wgslDesc.code = shaderCode;

  wgpu::ShaderModuleDescriptor shaderModuleDescriptor{.nextInChain = &wgslDesc};
  wgpu::ShaderModule shaderModule =
      device.CreateShaderModule(&shaderModuleDescriptor);

  wgpu::ColorTargetState colorTargetState{.format = format};

  wgpu::FragmentState fragmentState{
      .module = shaderModule, .targetCount = 1, .targets = &colorTargetState};

  wgpu::RenderPipelineDescriptor descriptor{.vertex = {.module = shaderModule},
                                            .fragment = &fragmentState};
  pipeline = device.CreateRenderPipeline(&descriptor);
}

void Render() {
  wgpu::TextureView backbuffer = swapChain.GetCurrentTextureView();

  wgpu::RenderPassColorAttachment attachment{
      .view = backbuffer,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.3f, 0.3f, 0.3f, 1.0f}};

  wgpu::RenderPassDescriptor renderpass{.colorAttachmentCount = 1,
                                        .colorAttachments = &attachment};

  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
  pass.SetPipeline(pipeline);
  pass.Draw(3);
  pass.End();
  wgpu::CommandBuffer commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);
  swapChain.Present();
}

void InitGraphics() {
  ConfigureSwapChain();
  CreateRenderPipeline();
}

#if defined(__EMSCRIPTEN__)
EM_BOOL MainLoop(double time, void *userData) {
  Render();
  return EM_TRUE;
}
#endif

void Start() {
  if (!glfwInit()) {
    return;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window =
      glfwCreateWindow(kWidth, kHeight, "WebGPU window", nullptr, nullptr);

#if defined(__EMSCRIPTEN__)
  wgpu::SurfaceDescriptorFromCanvasHTMLSelector canvasDesc{};
  canvasDesc.selector = "#canvas";

  wgpu::SurfaceDescriptor surfaceDesc{.nextInChain = &canvasDesc};
  surface = instance.CreateSurface(&surfaceDesc);
#else
  surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
#endif

  InitGraphics();

#if defined(__EMSCRIPTEN__)
  emscripten_request_animation_frame_loop(MainLoop, nullptr);
#else
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    Render();
    instance.ProcessEvents();
  }
#endif
}

int main() {
  instance = wgpu::CreateInstance();
  GetAdapter([](wgpu::Adapter a) {
    adapter = a;
    GetDevice([](wgpu::Device d) {
      device = d;
      Start();
    });
  });
}
