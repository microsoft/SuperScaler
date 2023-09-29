#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=4
NODE_RANK=0

RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}profiled-time-eurosys-new/

mkdir ${PROFILING_PATH}
MAX_NUM_GPUS=8
MODEL_NAME=t5
MODEL_SIZE=all

for ((tp_size=1; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
do
GPUS_PER_NODE=${tp_size}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
    op_profiler.py \
    --prof-tp-size $tp_size \
    --prof-path $PROFILING_PATH \
    --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_op_profile.pkl \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-repeat-times 40 10 \
    --prof-repeat-threshold 5000 \
    --prof-warmup-times 10 \
    --prof-warmup-threshold 100000 \
    --prof-num-nodes $NNODES \
    --prof-node-rank $NODE_RANK \
    --prof-ref-data ${RUNTIME_PATH}profiled-time-eurosys/${MODEL_NAME}_op_profile.pkl \
    2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_op_tp${tp_size}.log

echo [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
done
