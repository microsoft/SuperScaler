cmake_minimum_required(VERSION 3.17)

project(
  tfadaptor
  VERSION 0.1
  DESCRIPTION "superscaler plugin for tensorflow"
  LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME} PRIVATE operation.hpp operation.cpp gpu_util.hpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0
                                                   -DNDEBUG)
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:.>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

target_link_libraries(${PROJECT_NAME} PUBLIC superscaler_rt)
target_link_libraries(${PROJECT_NAME} PUBLIC tf_lib)
