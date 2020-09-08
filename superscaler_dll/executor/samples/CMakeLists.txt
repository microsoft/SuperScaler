cmake_minimum_required(VERSION 3.5.0)
SET(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${INCLUDE_DIRS})

if(CUDA_FOUND)
    add_executable(cuda_channel cuda_channel.cpp)
    add_executable(cuda_task cuda_task.cpp)
    target_link_libraries(cuda_channel 
        ${LIBRARIES}
        ${PROJECT_NAME})
    target_link_libraries(cuda_task
        ${LIBRARIES}
        ${PROJECT_NAME})
    set_target_properties(cuda_channel PROPERTIES
        CMAKE_CXX_STANDARD 11
        CMAKE_CXX_STANDARD_REQUIRED ON
    )
endif()