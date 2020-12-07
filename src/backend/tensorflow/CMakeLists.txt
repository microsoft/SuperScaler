# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.17)

project(
  tf_comm_ops
  VERSION 0.1
  DESCRIPTION "superscaler communication ops for tensorflow"
  LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} OBJECT)
target_sources(${PROJECT_NAME} PRIVATE operation.cpp gpu_util.hpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0
                                                   -DNDEBUG)
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:.>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} PUBLIC tf_lib superscaler_rt)
