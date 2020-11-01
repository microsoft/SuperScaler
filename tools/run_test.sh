#!/bin/bash
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
    exe $BUILD_PATH/src/backend/common_runtime/test/superscaler_rt_test
    echo "Backend test success!"
else
    echo "nvcc not found, skip the backend build and test"
fi

# Test for executor
echo "Running test for executor:"
exe cd $ROOT_PATH/src/backend/common_runtime/executor/
exe rm -rf build && exe mkdir build && exe cd build
exe cmake .. -DDO_TEST=true
exe make -j $(nproc) gtest_parallel
echo "Executor test success!"

# Test for frontend
echo "Running test for frontend:"
exe cd $ROOT_PATH/src/frontend/
exe $PYTHON -m flake8
exe $PYTHON -m pytest -v tests
echo "Frontend test success"

# Test for plan_gen
echo "Running test for plan_gen:"
exe cd $ROOT_PATH/src/frontend/plan_gen
exe $PYTHON -m flake8
exe $PYTHON -m pytest -v
echo "Plan_gen test success"

# Test for ai_simulator
echo "Running test for ai_simulator:"
exe cd $ROOT_PATH/src/frontend/ai_simulator
exe $PYTHON -m flake8
exe $PYTHON -m pytest -v
echo "Ai_simulator test success"

# Test for runtime
echo "Running test for runtime:"

if [ -f "$BUILD_PATH/lib/libtfadaptor.so" ]
then
    exe mkdir -p $ROOT_PATH/lib
    exe ln -sf $BUILD_PATH/lib/libtfadaptor.so $ROOT_PATH/lib/libtfadaptor.so
else
    echo 'libtfadaptor.so not found, skip some tests rely on libtfadptor.so'
fi

exe cd $ROOT_PATH/src/frontend/runtime
exe $PYTHON -m flake8
exe $PYTHON -m pytest -v
echo "Runtime test success"

# Test for scaler_graph
echo "Running test for scaler_graph:"
exe export PYTHONPATH=$ROOT_PATH/src
exe cd $ROOT_PATH/src/frontend/scaler_graph
exe $PYTHON -m flake8
exe $PYTHON -m pytest -v
echo "Scaler_graph test success"
