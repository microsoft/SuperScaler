# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.17)

project(superscaler_test)

add_executable(superscaler_op_test)
add_executable(superscaler_rt_test)
target_sources(superscaler_op_test PRIVATE superscaler_op_test.cc)
target_sources(superscaler_rt_test PRIVATE superscaler_rt_test.cc)
target_link_libraries(superscaler_op_test PRIVATE superscaler_rt)
target_link_libraries(superscaler_rt_test PRIVATE superscaler_rt)

find_package(CUDAToolkit 10.0 REQUIRED)
if(CUDAToolkit_FOUND)
  target_link_libraries(superscaler_op_test PRIVATE CUDA::cudart)
  target_link_libraries(superscaler_rt_test PRIVATE CUDA::cudart)
endif()

if(NOT TARGET Threads::Threads)
  find_package(Threads REQUIRED)
endif()

if(Threads_FOUND)
  target_link_libraries(superscaler_op_test PRIVATE gtest gtest_main
                                                    Threads::Threads)
  target_link_libraries(superscaler_rt_test PRIVATE gtest gtest_main
                                                    Threads::Threads)
endif()
