# Cross Platform WebGPU sample - Compute API

Similar to the previous example but on Compute API.

## Emscripten

If you have emscripten set up and configured properly, you would only need to run this command in the current directory.

```bash
emsdk install latest
emsdk activate latest
source "/path/to/emsdk/emsdk_env.sh"
```

The current code works with the latest emscripten. (3.1.61)
Install and activate it using `emsdk`.

```bash
emcmake cmake -B build-web && cmake --build build-web
```

## Dawn

If you want to use Dawn, you can add Dawn as submodule and let CMake take care of building it.

```bash
# Dawn is already added as a submodule to this repo
# git submodule add https://dawn.googlesource.com/dawn --branch chromium/6478
cmake -B build && cmake --build build -j8

# For debugging
cmake -DCMAKE_BUILD_TYPE=Debug -B build && cmake --build build -j6
```

## Buffer Access in older versions of Emscriptern

One way of accessing buffers is to use `EM_ASM` macros. For instance:

```cpp
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

#ifdef __EMSCRIPTEN__

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
      secondMatrix.data(), secondMatrixSize);
#endif
```
