cmake_minimum_required(VERSION 3.17)

project(superscaler_rt_test)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE "superscaler_test.cc")
target_link_libraries(${PROJECT_NAME} PRIVATE superscaler_rt)

find_package(CUDAToolkit 10.0 REQUIRED)
if(CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

if(NOT TARGET Threads::Threads)
  find_package(Threads REQUIRED)
endif()

if(Threads_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE gtest gtest_main
                                                Threads::Threads)
endif()
