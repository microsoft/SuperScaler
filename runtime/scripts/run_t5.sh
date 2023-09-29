#! /bin/bash

#### Hardware info ####
NNODES=1
GPUS_PER_NODE=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#### Paths ####
config_name=example_config
config_file=path_to_config/$config_name.json
LOG_PATH=path_to_logs/
mkdir -p ${LOG_PATH}csv
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --flexpipe-config $config_file \
       --train-iters 3 \
       --eval-iters 0 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file vocabs/t5-vocab.txt \
       --vocab-extra-ids 100 \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --DDP-impl local \
       --fp16 \
       --log-path $LOG_PATH \
       2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}
