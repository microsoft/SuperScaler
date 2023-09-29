#! /bin/bash
RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}profiled-time-eurosys-new/
mkdir ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_intra_node.csv

MASTER_ADDR=localhost \
MASTER_PORT=7000 \
NNODES=1 \
GPUS_PER_NODE=2 \
NODE_RANK=0 \
FILE_NAME=$FILE_NAME \
python3 p2p_band_profiler.py