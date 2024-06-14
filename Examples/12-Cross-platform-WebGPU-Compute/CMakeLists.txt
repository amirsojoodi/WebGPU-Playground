cmake_minimum_required(VERSION 3.13) 
project(matmult)                         
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

  # Enable optimization in code speed and size
  target_link_options(matmult PRIVATE "-O3")
  # target_link_options(matmult PRIVATE "-sDISABLE_EXCEPTION_CATCHING=0")

else()
  set(DAWN_FETCH_DEPENDENCIES ON)
  add_subdirectory("../../dawn" "build" EXCLUDE_FROM_ALL)
  target_link_libraries(matmult PRIVATE webgpu_cpp webgpu_dawn glfw webgpu_glfw)
endif()
