#! /bin/bash
ROOT_PATH=$(pwd)
cd $ROOT_PATH/search

exp_setting=$1

if [ "$exp_setting" == "small" ]; then
    model_name=t5
    model_size=770M
    global_batch_size=512

    #### Hardware info ####
    num_nodes=1
    gpus_per_node=4
    memory_limit=12000

    #### Paths ####
    DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-miniset/
    RESULT_PATH=${ROOT_PATH}/logs/megatron/

    LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
    CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
    mkdir -p ${LOG_PATH} && mkdir -p ${CONFIG_SAVE_PATH} 

    CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log

    python3 gen_megatron_plan.py \
        --model-name $model_name \
        --model-size $model_size \
        --global-batch-size $global_batch_size \
        --micro-batch-size 1 2 4 8 \
        --num-nodes $num_nodes \
        --num-gpus-per-node $gpus_per_node \
        --memory-limit $memory_limit \
        --log-path $LOG_PATH \
        --profiled-time-path $DATABASE_PATH \
        --config-save-path $CONFIG_SAVE_PATH \
        --config-suffix $CURRENT_TIME \
        --print-debug-info \
        2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_${CURRENT_TIME}.log

    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log

elif [ "$exp_setting" == "large" ]; then

    #### skip the search of 1-GPU case, align the config with baseline system:
    mkdir -p $ROOT_PATH/logs-large/megatron/configs/t5/770M/
    cp single_gpu_configs/t5_770M_mbs2_recomp.json $ROOT_PATH/logs-large/megatron/configs/t5/770M/

    #### Model info ####
    model_name=t5
    global_batch_size=1024

    #### Hardware info ####
    memory_limit=28000

    #### Paths ####
    DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-eurosys/
    RESULT_PATH=${ROOT_PATH}/logs-large/megatron/

    model_sizes=("3B" "6B" "11B" "22B")
    num_nodes_list=(1 1 2 4)
    gpus_per_node_list=(4 8 8 8)

    for ((index=0; index<4; index=index+1))
    do
        model_size=${model_sizes[$index]}
        num_nodes=${num_nodes_list[$index]}
        gpus_per_node=${gpus_per_node_list[$index]}

        LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
        CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
        mkdir -p ${LOG_PATH} && mkdir -p ${CONFIG_SAVE_PATH} 

        CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
        echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log

        python3 gen_megatron_plan.py \
            --model-name $model_name \
            --model-size $model_size \
            --global-batch-size $global_batch_size \
            --micro-batch-size 1 2 4 8 \
            --num-nodes $num_nodes \
            --num-gpus-per-node $gpus_per_node \
            --memory-limit $memory_limit \
            --log-path $LOG_PATH \
            --profiled-time-path $DATABASE_PATH \
            --config-save-path $CONFIG_SAVE_PATH \
            --config-suffix $CURRENT_TIME \
            --print-debug-info \
            2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_${CURRENT_TIME}.log

        echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
        
    done

fi