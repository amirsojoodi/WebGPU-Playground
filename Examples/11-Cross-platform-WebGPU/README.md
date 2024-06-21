# Cross Platform WebGPU sample

This example is from [here](https://developer.chrome.com/docs/web-platform/webgpu/build-app).

## Emscripten

If you have emscripten set up and configured properly, you would only need to run this command in the current directory.

```bash
emcmake cmake -B build-web && cmake --build build-web
```

## Dawn

If you want to use Dawn, you can add Dawn as submodule and let CMake takes care of building it.

```bash
# Dawn is already added as a submodule to this repo
# git submodule add https://dawn.googlesource.com/dawn --branch chromium/6478
cmake -B build && cmake --build build -j8
```

Currently, the standalone `app` generated out of `main.cpp` doesn't work properly with the configured dawn `chromium/6478`. The current workaround is commenting the following lines from `dawn/src/dawn/native/Surface.cpp` function `ValidateSurfaceConfiguration`, then building dawn again.

```cpp
DAWN_INVALID_IF(presentModeIt == capabilities.presentModes.end(),
                "Present mode (%s) is not supported by the adapter (%s) for this surface.",
                config->format, config->device->GetAdapter());
```

However, the other executable (`app-another-way`) should work without any issues.
