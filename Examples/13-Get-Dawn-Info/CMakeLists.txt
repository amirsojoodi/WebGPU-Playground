cmake_minimum_required(VERSION 3.13) 
project(DawnQuery
  LANGUAGES C CXX
)                         
set(CMAKE_CXX_STANDARD 20)           

add_executable(DawnQuery "DawnQuery.cpp")

set(DAWN_FETCH_DEPENDENCIES ON)
add_subdirectory("../../dawn" "build" EXCLUDE_FROM_ALL)
target_link_libraries(DawnQuery PRIVATE 
  dawn_internal_config
  dawncpp
  dawn_proc
  dawn_common
  dawn_glfw
  dawn_native
  dawn_wire
  dawn_utils
)
