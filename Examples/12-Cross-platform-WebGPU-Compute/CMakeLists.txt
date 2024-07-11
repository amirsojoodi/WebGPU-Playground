cmake_minimum_required(VERSION 3.13) 
project(matmult
  LANGUAGES C CXX
)                         
set(CMAKE_CXX_STANDARD 20)           

add_executable(matmult "matmult.cpp")

if(EMSCRIPTEN)
  # set_target_properties(matmult PROPERTIES SUFFIX ".html")
  
  # Create a JS file only, and not the html template file
  set_target_properties(matmult PROPERTIES SUFFIX ".js")

  # Enable WebGPU through (webgpu/webgpu.h)
  target_link_options(matmult PRIVATE "-sUSE_WEBGPU=1")

  # Help with printing stack trace, error prevention
  target_link_options(matmult PRIVATE "-sASSERTIONS=1")

  # Enable memory growth at runtime and refrain from throwing exception
  target_link_options(matmult PRIVATE "-sALLOW_MEMORY_GROWTH=1")

  # Disable WASM module generation. (Everything will be in a JS file)
  target_link_options(matmult PRIVATE "-sWASM=0")
  
  # Whether to support async operations in the compiled code. This makes it
  # possible to call JS functions from synchronous-looking code in C/C++.
  target_link_options(matmult PRIVATE "-sASYNCIFY=1")
  
  # Enable optimization in code speed and size
  target_link_options(matmult PRIVATE "-O3")

  # target_link_options(matmult PRIVATE "-sDISABLE_EXCEPTION_CATCHING=0")
  
  # Whether we will run the main() function. Disable if you embed the generated
  # code in your own, and will call main() yourself at the right time (which you
  # can do with Module.callMain()
  # target_link_options(matmult PRIVATE "-sINVOKE_RUN=0")

  # Specify which runtime environments the JS output will be capable of running
  # in.  For maximum portability this can configured to support all environments
  # or it can be limited to reduce overall code size.
  # var ENVIRONMENT = 'web,webview,worker,node';
  target_link_options(matmult PRIVATE "-sENVIRONMENT=web")

  # If set to 0, does not build in any filesystem support. Useful if you are just
  # doing pure computation, but not reading files or using any streams (including
  # fprintf, and other stdio.h things) or anything related.
  target_link_options(matmult PRIVATE "-sFILESYSTEM=0")
  
else()
  set(DAWN_FETCH_DEPENDENCIES ON)
  add_subdirectory("../../dawn" "build" EXCLUDE_FROM_ALL)
  target_link_libraries(matmult PRIVATE webgpu_cpp webgpu_dawn)
endif()

### Other options

# Symbols that are explicitly exported. These symbols are kept alive through
# LLVM dead code elimination, and also made accessible outside of the
# generated code even after running closure compiler (on "Module").  Native
# symbols listed here require an ``_`` prefix.
# By default if this setting is not specified on the command line the
# ``_main`` function will be implicitly exported.  In STANDALONE_WASM mode the
# default export is ``__start`` (or ``__initialize`` if --no-entry is specified).
# JS Library symbols can also be added to this list (without the leading `$`).
# var EXPORTED_FUNCTIONS = [];

# If true, we export the symbols that are present in JS onto the Module
# object.
# It only does ``Module['X'] = X;``
# var EXPORT_KEEPALIVE = true;

# If set to 1, this is a worker library, a special kind of library that is run
# in a worker. See emscripten.h
# [link]
# var BUILD_AS_WORKER = false;

# By default we emit all code in a straightforward way into the output
# .js file. That means that if you load that in a script tag in a web
# page, it will use the global scope. With ``MODULARIZE`` set, we instead emit
# the code wrapped in a function that returns a promise.
# The default name of the function is ``Module``, but can be changed using the
# ``EXPORT_NAME`` option. We recommend renaming it to a more typical name for a
# factory function, e.g. ``createModule``.
#
# You use the factory function like so::
#   const module = await EXPORT_NAME();
# or::
#   let module;
#   EXPORT_NAME().then(instance => {
#     module = instance;
#   });

# STANDALONE_WASM indicates that we want to emit a wasm file that can run
# without JavaScript. The file will use standard APIs such as wasi as much as
# possible to achieve that.
#
# This option does not guarantee that the wasm can be used by itself - if you
# use APIs with no non-JS alternative, we will still use those (e.g., OpenGL
# at the time of writing this). This gives you the option to see which APIs
# are missing, and if you are compiling for a custom wasi embedding, to add
# those to your embedding.
#
# We may still emit JS with this flag, but the JS should only be a convenient
# way to run the wasm on the Web or in Node.js, and you can run the wasm by
# itself without that JS (again, unless you use APIs for which there is no
# non-JS alternative) in a wasm runtime like wasmer or wasmtime.
#
# Note that even without this option we try to use wasi etc. syscalls as much
# as possible. What this option changes is that we do so even when it means
# a tradeoff with JS size. For example, when this option is set we do not
# import the Memory - importing it is useful for JS, so that JS can start to
# use it before the wasm is even loaded, but in wasi and other wasm-only
# environments the expectation is to create the memory in the wasm itself.
# Doing so prevents some possible JS optimizations, so we only do it behind
# this flag.
#
# When this flag is set we do not legalize the JS interface, since the wasm is
# meant to run in a wasm VM, which can handle i64s directly. If we legalized it
# the wasm VM would not recognize the API. However, this means that the
# optional JS emitted won't run if you use a JS API with an i64. You can use
# the WASM_BIGINT option to avoid that problem by using BigInts for i64s which
# means we don't need to legalize for JS (but this requires a new enough JS
# VM).
#
# Standalone builds require a ``main`` entry point by default.  If you want to
# build a library (also known as a reactor) instead you can pass ``--no-entry``.
# [link]
# var STANDALONE_WASM = false;

# If set to 1, we emit relocatable code from the LLVM backend; both
# globals and function pointers are all offset (by gb and fp, respectively)
# Automatically set for SIDE_MODULE or MAIN_MODULE.
# [compile+link]
# var RELOCATABLE = false;
# 
# A main module is a file compiled in a way that allows us to link it to
# a side module at runtime.
#
# - 1: Normal main module.
# - 2: DCE'd main module. We eliminate dead code normally. If a side
#   module needs something from main, it is up to you to make sure
#   it is kept alive.
#
# [compile+link]
# var MAIN_MODULE = 0;
#
# Corresponds to MAIN_MODULE (also supports modes 1 and 2)
# [compile+link]
# var SIDE_MODULE = 0;
