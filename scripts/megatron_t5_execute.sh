#! /bin/bash
ROOT_PATH=$(pwd)
exp_setting=$1
model_name=t5

if [ "$exp_setting" == "small" ]; then
    cd $ROOT_PATH/runtime

    #### Model info ####
    model_size=770M

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
    RESULT_PATH=${ROOT_PATH}/logs/megatron/
    LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
    CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
    mkdir -p ${LOG_PATH}csv

    for file_name in $(ls $CONFIG_SAVE_PATH)
    do
    config_name=`basename $file_name .json`
    CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) start executing config: $config_name ." >> ${RESULT_PATH}full_log.log

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
        
    echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log
    done 

elif [ "$exp_setting" == "large" ]; then
    ## Paths
    RESULT_PATH=${ROOT_PATH}/logs-large/megatron/

    ## 1node (4GPUs and 8GPUs)
    bash scripts/megatron_dist_scripts/run_${model_name}_1node.sh

    ## 2nodes
    parallel-ssh -i -t 0 -h pssh-2workers.host "ps -aux | grep 'pretrain' | grep -v grep | awk '{print \$2}' | xargs kill -9"
    parallel-ssh -i -t 0 -h pssh-2workers.host "cd $ROOT_PATH && bash scripts/megatron_dist_scripts/run_${model_name}_2nodes.sh"

    ## 4nodes
    parallel-ssh -i -t 0 -h pssh-4workers.host "ps -aux | grep 'pretrain' | grep -v grep | awk '{print \$2}' | xargs kill -9"
    parallel-ssh -i -t 0 -h pssh-4workers.host "cd $ROOT_PATH && bash scripts/megatron_dist_scripts/run_${model_name}_4nodes.sh"

fi

python3 $ROOT_PATH/runtime/scripts/show_best_perf.py $model_name $RESULT_PATH 2>&1 | tee -a ${RESULT_PATH}full_log.log