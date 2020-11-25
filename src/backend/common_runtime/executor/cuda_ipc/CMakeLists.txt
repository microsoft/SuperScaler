# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(CUDA_FOUND)
    file(GLOB CUDA_IPC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

    add_definitions(-D WARNING_AS_ERROR )

    set(SOURCES
        ${SOURCES}
        ${CUDA_IPC_SRC}
        PARENT_SCOPE)

    set(INCLUDE_DIRS
        ${INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDA_INCLUDE_DIRS}
        PARENT_SCOPE)

    set(LIBRARIES
        ${CUDA_LIBRARIES}
        -lcuda
        ${LIBRARIES}
        PARENT_SCOPE)
endif()