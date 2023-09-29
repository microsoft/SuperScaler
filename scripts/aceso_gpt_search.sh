#! /bin/bash
ROOT_PATH=$(pwd)
cd $ROOT_PATH/search

exp_setting=$1
search_budget=200

if [ "$exp_setting" == "small" ]; then
    model_name=gpt
    model_size=1_3B
    global_batch_size=512

    #### Hardware info ####
    num_nodes=1
    gpus_per_node=4
    memory_limit=12000

    #### Search algo parameters ####
    budget=$search_budget
    max_num_hops=7
    init_config=balance

    #### Paths ####
    DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-miniset/
    RESULT_PATH=${ROOT_PATH}/logs/aceso/

    LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
    CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
    mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

    CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
        
    python3 aceso_search.py \
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
        --max-num-hops $max_num_hops \
        --time-budget-total $budget \
        --initial-point $init_config \
        2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_budget${budget}_${CURRENT_TIME}.log

    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log

elif [ "$exp_setting" == "large" ]; then

    #### Paths ####
    DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-eurosys/
    RESULT_PATH=${ROOT_PATH}/logs-large/aceso/

    ## Settings used in Aceso Paper
    ## model_size    num_nodes   gpus_per_node  global_batch_size
    ## 350M          1           1              1024
    ## 1_3B          1           4              1024
    ## 2_6B          1           8              1024
    ## 6_7B          2           8              1024
    ## 13B           4           8              1024

    #### skip the search of 1-GPU case, align the config with baseline system:
    config_name=gpt_350M_mbs8_recomp
    config_path=$ROOT_PATH/logs-large/aceso/configs/gpt/350M/
    mkdir -p ${config_path}csv && mkdir mkdir -p ${config_path}top_configs
    cp single_gpu_configs/$config_name.json $ROOT_PATH/logs-large/aceso/configs/gpt/350M/top_configs/
    python3 aceso_cost_model.py \
        --initial-point single_gpu_configs/$config_name.json \
        --profiled-time-path $DATABASE_PATH \
        --num-gpus-per-node 1 \
        --num-nodes 1 \
        --save-to-csv ${config_path}csv/info_$config_name.csv

    #### Model info ####
    model_name=gpt
    global_batch_size=1024

    #### Hardware info ####
    memory_limit=28000

    #### Search algo parameters ####
    budget=$search_budget
    max_num_hops=7
    init_config=balance

    model_sizes=("1_3B" "2_6B" "6_7B" "13B")
    num_nodes_list=(1 1 2 4)
    gpus_per_node_list=(4 8 8 8)

    for ((index=0; index<4; index=index+1))
    do
        model_size=${model_sizes[$index]}
        num_nodes=${num_nodes_list[$index]}
        gpus_per_node=${gpus_per_node_list[$index]}

        LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
        CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
        mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

        CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
        echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
            
        python3 aceso_search.py \
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
            --max-num-hops $max_num_hops \
            --time-budget-total $budget \
            --initial-point $init_config \
            --num-of-saved-configs 3 \
            2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_budget${budget}_${CURRENT_TIME}.log
        echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
        
    done

elif [ "$exp_setting" == "scale" ]; then

    #### Model info ####
    model_name=scale-layer
    global_batch_size=1024

    #### Hardware info ####
    num_nodes=1
    gpus_per_node=8
    memory_limit=28000

    #### Search algo parameters ####
    budget=$search_budget
    max_num_hops=7
    init_config=balance

    #### Paths ####
    DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-eurosys/
    RESULT_PATH=${ROOT_PATH}/logs-large/aceso/

    num_layers_list=(8 16 32 64 128 256 512 1024)
    for ((index=0; index<8; index=index+1))
    do
        num_layers=${num_layers_list[$index]}
        model_size=${num_layers}layers

        LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
        CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
        mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

        CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
        echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
            
        python3 aceso_search.py \
            --model-name $model_name \
            --model-size $model_size \
            --global-batch-size $global_batch_size \
            --micro-batch-size 1 2 4 8 \
            --num-layers $num_layers \
            --num-nodes $num_nodes \
            --num-gpus-per-node $gpus_per_node \
            --memory-limit $memory_limit \
            --log-path $LOG_PATH \
            --profiled-time-path $DATABASE_PATH \
            --config-save-path $CONFIG_SAVE_PATH \
            --config-suffix $CURRENT_TIME \
            --max-num-hops $max_num_hops \
            --time-budget-total $budget \
            --op-group-size 1 \
            --peak-mem-in-backward 1000 \
            --predict-delta-time \
            2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_budget${budget}_${CURRENT_TIME}.log

        echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
                
    done

fi



 