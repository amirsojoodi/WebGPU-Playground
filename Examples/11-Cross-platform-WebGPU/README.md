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
cmake -B build && cmake --build build
```
