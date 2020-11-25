# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  set(BUILD_SHARED_LIBS ON)
  include(external_gtest)
endif()
