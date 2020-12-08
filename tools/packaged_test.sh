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

# test if python3 work properly
if ! command -v $PYTHON &> /dev/null
then
    echo "ERROR: $PYTHON command not found!"
    exit 1
else
    echo 'Using '$PYTHON', version:' $($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
fi

# install superscaler package if haven't installed
if python -c "import superscaler" &> /dev/null; then
    echo 'superscaler package has installed, skip installation step.'
else
    echo 'installing superscaler package...'
    exe python -m pip install .
fi

# run test for superscaler package
echo 'running test for superscaler package...'

# Origin Pytest:
# exe python -m pytest -vv tests 

# Hotfix for remaining bugs:
# --------------------------
echo "Skipping runtime for remaining bugs"
exe python -m pytest -vv tests \
                     --ignore=tests/runtime \

echo "Run test for runtime seperately, for remaining bugs which cause failure if Runtime being executed multi-times."
exe python -m pytest tests/runtime
# --------------------------