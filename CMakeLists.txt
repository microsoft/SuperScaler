cmake_minimum_required(VERSION 3.17)

project(
  superscaler
  VERSION 0.1
  DESCRIPTION "superscaler, a general distributed system for machine learning")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SUPERSCALER_SOURCE_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(SUPERSCALER_BINARY_ROOT ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(external)
add_subdirectory(src)
