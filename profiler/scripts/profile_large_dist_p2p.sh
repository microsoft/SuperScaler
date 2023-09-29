#! /bin/bash
RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}profiled-time-eurosys-new/
mkdir ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_inter_node.csv

MASTER_ADDR=localhost
NODE_RANK=0

if [[ $NODE_RANK -eq 0 || $NODE_RANK -eq 1 ]]; then
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=7000 \
    NNODES=2 \
    GPUS_PER_NODE=1 \
    NODE_RANK=$NODE_RANK \
    FILE_NAME=$FILE_NAME \
    python3 p2p_band_profiler.py
fi