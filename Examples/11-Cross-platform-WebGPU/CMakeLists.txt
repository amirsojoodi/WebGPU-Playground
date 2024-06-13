cmake_minimum_required(VERSION 3.13) 
project(app)                         
set(CMAKE_CXX_STANDARD 20)           

add_executable(app "main.cpp")

if(EMSCRIPTEN)
  set_target_properties(app PROPERTIES SUFFIX ".html")
  target_link_options(app PRIVATE "-sUSE_WEBGPU=1" "-sUSE_GLFW=3")
else()
  set(DAWN_FETCH_DEPENDENCIES ON)
  add_subdirectory("../../dawn" "build" EXCLUDE_FROM_ALL)
  target_link_libraries(app PRIVATE webgpu_cpp webgpu_dawn glfw webgpu_glfw)
endif()
