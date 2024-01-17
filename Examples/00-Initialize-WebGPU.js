/**
 * @file        00-Initialize-WebGPU.js
 * @author      Amir Sojoodi, amir@distributive.network
 * @date        Nov 2022
 *
 * @description Add an example on WebGPU initialization
 *
 * Depending on your x server settings, either this:
 * `DISPLAY=:1 google-chrome-unstable --enable-unsafe-webgpu
 * --enable-features=Vulkan &`
 *
 * Or this: `DISPLAY=:0 google-chrome-unstable
 * --enable-unsafe-webgpu --enable-features=Vulkan &`
 *
 * An example to run headless chrome using the flags:
 * google-chrome-stable --no-sandbox --headless=new --use-angle=vulkan --enable-features=Vulkan --disable-vulkan-surface --enable-unsafe-webgpu --no-first-run --no-default-browser-check --disable-features=Translate --ash-no-nudges --disable-search-engine-choice-screen --virtual-time-budget=300000 --window-size=1280,720 --print-to-pdf=/content/gpu.pdf https://webgpureport.org
 *
 */

let gpuDevice = null;

async function initializeWebGPU() {
  // Check to ensure the user agent supports WebGPU.
  if (!('gpu' in navigator)) {
    console.error('User agent doesn\'t support WebGPU.');
    return false;
  }

  // Request an adapter.
  const gpuAdapter = await navigator.gpu.requestAdapter();

  // requestAdapter may resolve with null if no suitable adapters are found.
  if (!gpuAdapter) {
    console.error('No WebGPU adapters found.');
    return false;
  }

  const requiredFeatures = {
    maxStorageBufferBindingSize: 4294967292,
    maxBufferSize: 4294967292
  };
  // Request a device.
  // Note that the promise will reject if invalid options are passed to the
  // optional dictionary. To avoid the promise rejecting always check any
  // features and limits against the adapters features and limits prior to
  // calling requestDevice().
  gpuDevice =
      await gpuAdapter.requestDevice({requiredLimits: requiredFeatures});

  // requestDevice will never return null, but if a valid device request can't
  // be fulfilled for some reason it may resolve to a device which has already
  // been lost. Additionally, devices can be lost at any time after creation for
  // a variety of reasons (ie: browser resource management, driver updates), so
  // it's a good idea to always handle lost devices gracefully.
  gpuDevice.lost.then((info) => {
    console.error(`WebGPU device was lost: ${info.message}`);

    gpuDevice = null;

    // Many causes for lost devices are transient, so applications should try
    // getting a new device once a previous one has been lost unless the loss
    // was caused by the application intentionally destroying the device. Note
    // that any WebGPU resources created with the previous device (buffers,
    // textures, etc) will need to be re-created with the new one.
    if (info.reason != 'destroyed') {
      initializeWebGPU();
    }
  });

  onWebGPUInitialized();

  return true;
}

function onWebGPUInitialized() {
  // Begin creating WebGPU resources here...
}

initializeWebGPU();
