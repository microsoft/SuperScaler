# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_subdirectory(common_runtime)
add_subdirectory(tensorflow)

project(
  superscaler_pywrap
  VERSION 0.1
  DESCRIPTION "superscaler plugin for tensorflow"
  LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} SHARED)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
target_sources(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}.cpp ${PROJECT_NAME}.hpp)
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:.>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} PUBLIC superscaler_rt tf_comm_ops)

