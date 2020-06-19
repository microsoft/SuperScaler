# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.5.0)
project(executor)

find_package(CUDA)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda_ipc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils)

SET(CMAKE_CXX_FLAGS "-std=c++11")

# Include directories
set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/channel
)

include_directories(${INCLUDE_DIRS})

# Get number of processors
include(ProcessorCount)
ProcessorCount(CPU_NUM)
message(STATUS "CPU NUM = ${CPU_NUM}")
if( CPU_NUM EQUAL 0)
    set(CPU_NUM 1)
endif()

# Link directories
set(LINK_DIRS
)
link_directories(${LINK_DIRS})

set(LIBRARIES
    ${LIBRARIES}
)

# SRC
file(GLOB CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(SOURCES
    ${SOURCES}
    ${CPP_SOURCES}
)

# Output
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CMAKE_CXX_STANDARD 11
    CMAKE_CXX_STANDARD_REQUIRED ON
)
target_compile_options(${PROJECT_NAME}
    PRIVATE
    -Wall
    -Werror
    -Wextra
)

# Test
OPTION(DO_TEST "Build google test？" OFF)
if(NOT DEFINED REPEAT_TIMES)
    set(REPEAT_TIMES "100")
endif()

if(NOT DEFINED TEST_WORKERS)
    set(TEST_WORKERS 
        ${CPU_NUM})
    # Too much test workers will lead to OOM and test failed,
    # because cuda context takes a lot of memory
    if(TEST_WORKERS GREATER 32)
        set(TEST_WORKERS 32)
    endif()
endif()


message(STATUS "REPEAT_TIMES = ${REPEAT_TIMES}")
message(STATUS "TEST_WORKERS = ${TEST_WORKERS}")

if(${DO_TEST})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
