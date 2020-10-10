cmake_minimum_required(VERSION 3.17)

project(tensorflow)

execute_process(
  COMMAND
    bash -c
    "python3 -c \'import tensorflow as tf; print(\"\".join(tf.sysconfig.get_include()), end=\"\")\'"
  OUTPUT_VARIABLE TF_INCLUDE)
execute_process(
  COMMAND
    bash -c
    "python3 -c \'import tensorflow as tf; print(\"\".join(tf.sysconfig.get_lib()) + \"/libtensorflow_framework.so.1\", end=\"\")\'"
  OUTPUT_VARIABLE TF_LIB)

message(STATUS "tensorflow include path :" ${TF_INCLUDE})
message(STATUS "tensorflow lib path :" ${TF_LIB})

add_library(tf_lib INTERFACE)
target_include_directories(tf_lib SYSTEM INTERFACE ${TF_INCLUDE})
target_link_libraries(tf_lib INTERFACE ${TF_LIB})
target_compile_features(tf_lib INTERFACE cxx_std_11)
target_compile_definitions(tf_lib INTERFACE -D_GLIBCXX_USE_CXX11_ABI=0 -DNDEBUG)
