# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${INCLUDE_DIRS})

if(CUDA_FOUND)
    add_executable(cuda_channel cuda_channel.cpp)
    add_executable(cuda_task cuda_task.cpp)
    add_executable(allreduce_sample allreduce_sample.cpp)
    add_executable(ring_allreduce_sample ring_allreduce_sample.cpp)
    add_executable(handle_manager handle_manager.cpp)

    target_link_libraries(cuda_channel
        ${LIBRARIES}
        ${PROJECT_NAME})
    target_link_libraries(cuda_task
        ${LIBRARIES}
        ${PROJECT_NAME})
    target_link_libraries(allreduce_sample
        ${LIBRARIES}
        ${PROJECT_NAME})
    target_link_libraries(ring_allreduce_sample
        ${LIBRARIES}
        ${PROJECT_NAME})
    target_link_libraries(handle_manager
        ${LIBRARIES}
        ${PROJECT_NAME})
    set_source_files_properties(allreduce_sample PROPERTIES
        LANGUAGE CUDA
    )
    set_source_files_properties(ring_allreduce_sample PROPERTIES
        LANGUAGE CUDA
    )
    target_compile_features(allreduce_sample PRIVATE cxx_std_11)
    target_compile_features(ring_allreduce_sample PRIVATE cxx_std_11)
    set_target_properties(cuda_channel PROPERTIES
        CMAKE_CXX_STANDARD 11
        CMAKE_CXX_STANDARD_REQUIRED ON
    )
endif()