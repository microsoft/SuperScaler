# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

project(superscaler_rt LANGUAGES CUDA CXX)

add_library(util OBJECT)
target_sources(
  util
  PUBLIC util.hpp
  PRIVATE util.cpp)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(util PUBLIC nlohmann_json::nlohmann_json)

add_subdirectory(executor)

# add_library(${PROJECT_NAME} SHARED)
add_library(${PROJECT_NAME} STATIC)

target_sources(
  ${PROJECT_NAME}
  PUBLIC session.hpp
  PRIVATE session.cpp)

target_include_directories(
  ${PROJECT_NAME} SYSTEM
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:.>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} PUBLIC util executor)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
  add_compile_options(-DGPU_SWITCH)
  add_definitions(-DHAVE_CUDA)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
endif()

enable_testing()
add_subdirectory(test)
