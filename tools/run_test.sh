#!/bin/bash

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

# execute this script at the root dir of SuperScaler project
# function to display commands
exe() { echo "\$ ${@/eval/}" ; "$@" ; }

# use debug mode of shell
set -e

# we must use python3 rather than python because
# we have hard-code use of python3 in python scripts
PYTHON=python3
NVCC=${NVCC:-nvcc}

ROOT_PATH=$(pwd)
BUILD_PATH=${BUILD_PATH:-$ROOT_PATH/build}

# test if python3 work properly
if ! command -v $PYTHON &> /dev/null
then
    echo "ERROR: $PYTHON command not found!"
    exit 1
else
    echo 'Using '$PYTHON', version:' $($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
fi

# Test for backend
echo "checking if"$NVCC" command exist:"
if command -v $NVCC &> /dev/null
then
    echo "nvcc found, build backend:"
    # build the project if haven't been build before
    if [ -d "$BUILD_PATH" ]
    then
        echo 'Using pre-build path: '$BUILD_PATH
        echo "  Remove this path or set BUILD_PATH to another dir if you don't want to use pre-built project"
    else
        exe mkdir $BUILD_PATH
    fi

    exe cd $BUILD_PATH
    exe cmake ..
    exe make -j $(nproc)

    echo "Build success! Running test for backend:"
    exe $BUILD_PATH/src/backend/common_runtime/test/superscaler_op_test
    exe $BUILD_PATH/src/backend/common_runtime/test/superscaler_rt_test
    echo "Backend test success!"
else
    echo "nvcc not found, skip the backend build and test"
fi

# Build library
if [ -f "$BUILD_PATH/lib/libsuperscaler_pywrap.so" ]
then
    exe mkdir -p $ROOT_PATH/src/superscaler/lib
    exe ln -sf $BUILD_PATH/lib/libsuperscaler_pywrap.so $ROOT_PATH/src/superscaler/lib/libsuperscaler_pywrap.so
else
    echo 'libsuperscaler_pywrap.so not found, skip some tests rely on libsuperscaler_pywrap.so'
fi

# Test for executor
echo "Running test for executor:"
exe cd $ROOT_PATH/src/backend/common_runtime/executor/
exe rm -rf build && exe mkdir build && exe cd build
exe cmake .. -DDO_TEST=true
exe make -j $(nproc) gtest_parallel
exe make -j 1 enable_p2p_access
echo "Executor test success!"

# Test for flake8
echo "running flake8 check:"
exe cd $ROOT_PATH
exe $PYTHON -m flake8
echo "flake8 check succeed"

# prepare cifar10 dataset
exe cd $ROOT_PATH/example/tensorflow/dataset/
exe bash $ROOT_PATH/example/tensorflow/dataset/download_cifar10.sh

# Test for superscaler source code
echo "Running test for superscaler:"
exe cd $ROOT_PATH/src

# Runtime tests are executed separately because they will create multiple processes.
# CUDA context cannot be shared between multiple processes
exe python -m pytest ../tests \
                    --ignore=../tests/runtime ;\
                    python -m pytest ../tests/runtime
