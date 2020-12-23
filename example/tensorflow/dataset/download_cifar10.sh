#!/bin/bash

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
wget --progress=dot:mega -c https://codeload.github.com/tensorflow/models/tar.gz/v2.3.0 -O /tmp/models-2.3.0.tar.gz
tar -zxf /tmp/models-2.3.0.tar.gz -C ./
patch -p0 < tf_cifar10_dataset.patch
slim_dir="models-2.3.0/research/slim"
echo "Add $slim_dir in your PYTHONPATH"
echo "  command example: export PYTHONPATH=\$PYTHONPATH:$slim_dir"
export PYTHONPATH=$PYTHONPATH:$slim_dir
echo "pass the number of dataset shard you want, 2 by default"
echo "  command example: python3 $slim_dir/datasets/download_and_convert_cifar10.py 2 /tmp/sc_test/cifar10/"
python3 $slim_dir/datasets/download_and_convert_cifar10.py 2 /tmp/sc_test/cifar10/
rm -r models-2.3.0
ls /tmp/sc_test/cifar10/