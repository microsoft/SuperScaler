#! /bin/bash

## Settings used in Aceso Paper
## model_size   num_nodes   gpus_per_node  global_batch_size
## 2B           1           4              1536
## 4B           1           8              1536
## 6_8B         2           8              1536
## 13B          4           8              1536

#### Model info ####
model_name=resnet
model_size=500M
global_batch_size=1536

#### Hardware info ####
num_nodes=1
gpus_per_node=1
memory_limit=28000

#### Search algo parameters ####
budget=200
max_num_hops=7
init_config=balance

#### Paths ####
DATABASE_PATH=../profiler/profiled-time-eurosys/
RESULT_PATH=../test_eval_logs_5/

LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][SEARCH]($CURRENT_TIME) searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
    
python3 aceso_search.py \
    --model-name $model_name \
    --model-size $model_size \
    --global-batch-size $global_batch_size \
    --micro-batch-size 16 32 48 64 \
    --num-nodes $num_nodes \
    --num-gpus-per-node $gpus_per_node \
    --memory-limit $memory_limit \
    --log-path $LOG_PATH \
    --profiled-time-path $DATABASE_PATH \
    --config-save-path $CONFIG_SAVE_PATH \
    --config-suffix $CURRENT_TIME \
    --max-num-hops $max_num_hops \
    --time-budget-total $budget \
    --finetune-after-trial 1 \
    --initial-point $init_config \
    2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_budget${budget}_${CURRENT_TIME}.log
