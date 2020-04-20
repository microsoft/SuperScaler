set(GTEST_PARALLEL_DIR gtest-parallel-download)
configure_file(gtest-parallel.cmake.in ${GTEST_PARALLEL_DIR}/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${GTEST_PARALLEL_DIR} )
if(result)
    message(FATAL_ERROR "CMake step for gtest-parallel failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${GTEST_PARALLEL_DIR} )
if(result)
    message(FATAL_ERROR "Build step for gtest-parallel failed: ${result}")
endif()

set(GTEST_PARALLEL ${CMAKE_CURRENT_BINARY_DIR}/gtest-parallel/gtest-parallel)


add_custom_target(gtest_parallel COMMAND ${GTEST_PARALLEL} 
    ${CMAKE_CURRENT_BINARY_DIR}/${TEST_PROJECT}
    -r${REPEAT_TIMES}
    -w${TEST_WORKERS})
add_dependencies(gtest_parallel ${TEST_PROJECT})
