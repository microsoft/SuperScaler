#! /bin/bash

###### 8GPUs * 2nodes ######
#### Model info ####
model_name=t5
model_size=11B

#### Hardware info ####
NNODES=2
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#### Paths ####
RESULT_PATH=../logs-large/aceso/
LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
mkdir -p ${LOG_PATH}csv

for file_name in $(ls $CONFIG_SAVE_PATH)
do
config_name=`basename $file_name .json`
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][RUNTIME]($CURRENT_TIME) start executing cofnig: $config_name ." >> ${RESULT_PATH}full_log.log

NCCL_SOCKET_IFNAME=eno1 \
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
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

echo "[LOG][RUNTIME]($CURRENT_TIME) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

## For distributed running multiple configs:
## some config may cause OOM in some nodes, and the other nodes will not know the OOM and hang forever.
## we use a parallel-ssh command following the normal execution, when some node fails due to OOM, it will 
## kill the tasks in all the other nodes.
## To use this, prepare a pssh-2workers.host or pssh-4workers.host, which contains the host name or IP addresses of other nodes.
parallel-ssh -i -t 0 -h pssh-${NNODES}workers.host "docker restart aceso-ae"
sleep 10s 

done 
