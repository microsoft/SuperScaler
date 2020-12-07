# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.17)

set(TEST_PROJECT test_${PROJECT_NAME})
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(${TEST_DIR}/googletest.cmake)
include(${TEST_DIR}/gtest-parallel.cmake)

get_directory_property(PARENT_DIRECTORY PARENT_DIRECTORY)
set(TEST_INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ${PARENT_DIRECTORY}
    ${TEST_DIR}/utils
    ${TEST_INCLUDE_DIRS}
)
include_directories(${TEST_INCLUDE_DIRS})

set(TEST_LIBRARIES
    ${TEST_LIBRARIES}
    ${LIBRARIES}
    ${PROJECT_NAME}
)

if(CUDA_FOUND)
    file(GLOB TEST_CUDA_IPC_SOURCES ${TEST_DIR}/cuda_ipc/*.cpp)
endif()

file(GLOB TEST_MAIN_SOURCES ${TEST_DIR}/*.cpp)
file(GLOB TEST_CHANNEL_SOURCES ${TEST_DIR}/channel/*.cpp)
file(GLOB TEST_UTILS_SOURCES ${TEST_DIR}/utils/*.cpp)

set(TEST_SOURCES
    ${TEST_CUDA_IPC_SOURCES}
    ${TEST_CHANNEL_SOURCES}
    ${TEST_UTILS_SOURCES}
    ${TEST_MAIN_SOURCES})

add_executable(${TEST_PROJECT} ${TEST_SOURCES})
target_link_libraries(${TEST_PROJECT} ${TEST_LIBRARIES})

set_target_properties(${TEST_PROJECT} PROPERTIES
    CMAKE_CXX_STANDARD 11
    CMAKE_CXX_STANDARD_REQUIRED ON
)

if(CUDA_FOUND)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_features(${TEST_PROJECT} PRIVATE cxx_std_11)
endif()

target_compile_options(${TEST_PROJECT}
    PRIVATE
)

add_custom_target(g_test COMMAND ${TEST_PROJECT})
add_dependencies(g_test ${TEST_PROJECT})