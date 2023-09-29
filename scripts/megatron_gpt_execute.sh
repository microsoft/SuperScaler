#! /bin/bash
ROOT_PATH=$(pwd)
exp_setting=$1
model_name=gpt

if [ "$exp_setting" == "small" ]; then
    cd $ROOT_PATH/external/Megatron-LM
    #### Model info ####
    model_size=1_3B

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

    python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --config-file $CONFIG_SAVE_PATH${file_name} \
        --train-iters 3 \
        --eval-iters 0 \
        --lr-decay-iters 320000 \
        --vocab-file ${ROOT_PATH}/runtime/vocabs/gpt2-vocab.json \
        --merge-file ${ROOT_PATH}/runtime/vocabs/gpt2-merges.txt \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
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

elif [ "$exp_setting" == "scale" ]; then
    cd $ROOT_PATH/external/Megatron-LM

    #### Model info ####
    model_name=scale-layer

    #### Hardware info ####
    NNODES=1
    GPUS_PER_NODE=8
    WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

    #### Distributed info ####
    ## Modify this for distributed training
    NODE_RANK=0
    MASTER_ADDR=localhost
    MASTER_PORT=7000
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

    #### Paths ####
    RESULT_PATH=${ROOT_PATH}/logs-large/megatron/

    num_layers_list=(8 16 32 64 128 256 512 1024)
    for ((index=0; index<8; index=index+1))
    do

        num_layers=${num_layers_list[$index]}
        model_size=${num_layers}layers

        LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
        CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
        mkdir -p ${LOG_PATH}csv

        for file_name in $(ls $CONFIG_SAVE_PATH)
        do
            config_name=`basename $file_name .json`
            CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

            echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) start executing config: $config_name ." >> ${RESULT_PATH}full_log.log

            python -m torch.distributed.launch $DISTRIBUTED_ARGS \
                pretrain_gpt.py \
                --config-file $CONFIG_SAVE_PATH${file_name} \
                --train-iters 3 \
                --eval-iters 0 \
                --lr-decay-iters 320000 \
                --vocab-file ${ROOT_PATH}/runtime/vocabs/gpt2-vocab.json \
                --merge-file ${ROOT_PATH}/runtime/vocabs/gpt2-merges.txt \
                --data-impl mmap \
                --split 949,50,1 \
                --distributed-backend nccl \
                --lr 0.00015 \
                --lr-decay-style cosine \
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
    done

fi

python3 $ROOT_PATH/runtime/scripts/show_best_perf.py $model_name $RESULT_PATH 2>&1 | tee -a ${RESULT_PATH}full_log.log