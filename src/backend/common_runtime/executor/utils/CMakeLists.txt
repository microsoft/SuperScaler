# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

file(GLOB UTILS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


set(SOURCES
    ${SOURCES}
    ${UTILS_SRCS}
    PARENT_SCOPE)

set(INCLUDE_DIRS
    ${INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)

set(LIBRARIES
    ${LIBRARIES}
    pthread
    rt
    PARENT_SCOPE)