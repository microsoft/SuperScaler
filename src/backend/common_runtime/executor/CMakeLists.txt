# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.17)
project(executor)

find_package(CUDA)

if(CUDA_FOUND)
  enable_language(CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
  add_compile_options(-DGPU_SWITCH)
  add_definitions(-DHAVE_CUDA)
endif(CUDA_FOUND)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda_ipc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/samples)

set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Include directories
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/channel)

include_directories(${INCLUDE_DIRS})

# Get number of processors
include(ProcessorCount)
ProcessorCount(CPU_NUM)
message(STATUS "CPU NUM = ${CPU_NUM}")
if(CPU_NUM EQUAL 0)
  set(CPU_NUM 1)
endif()

# Link directories
set(LINK_DIRS)
link_directories(${LINK_DIRS})

set(LIBRARIES ${LIBRARIES})

# SRC
file(GLOB CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
set(SOURCES ${SOURCES} ${CPP_SOURCES} ${CUDA_SOURCES})

# Output
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(
  ${PROJECT_NAME} SYSTEM
  INTERFACE ${INCLUDE_DIRS}
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:.>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CXX_STANDARD 11
                                                 CMAKE_CXX_STANDARD_REQUIRED ON)
target_compile_options(${PROJECT_NAME} PRIVATE)

if(CUDA_FOUND)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                   ON)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                   ON)
  set_source_files_properties(${PROJECT_NAME} PROPERTIES LANGUAGE CUDA)
endif(CUDA_FOUND)

# Test
option(DO_TEST "Build google test？" OFF)
if(NOT DEFINED REPEAT_TIMES)
  set(REPEAT_TIMES "100")
endif()

if(NOT DEFINED TEST_WORKERS)
  set(TEST_WORKERS ${CPU_NUM})
  # Too much test workers will lead to OOM and test failed, because cuda context
  # takes a lot of memory
  if(TEST_WORKERS GREATER 32)
    set(TEST_WORKERS 32)
  endif()
endif()

message(STATUS "REPEAT_TIMES = ${REPEAT_TIMES}")
message(STATUS "TEST_WORKERS = ${TEST_WORKERS}")

if(${DO_TEST})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
