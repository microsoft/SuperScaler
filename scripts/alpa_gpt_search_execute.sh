#!/bin/bash
ROOT_PATH=$(pwd)
exp_setting=$1

if [ "$exp_setting" == "small" ]; then
  log_dir="logs"
elif [ "$exp_setting" == "large" ]; then
  log_dir="logs-large"
elif [ "$exp_setting" == "scale" ]; then
  log_dir="logs-large"
fi

cd external/alpa

CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
model_name=gpt

run_experiment () {
  NUM_HOSTS=$1
  NUM_DEVICES_PER_HOST=$2
  suite=$3
  model_size=$4

  NUM_GPUS=$((NUM_HOSTS * NUM_DEVICES_PER_HOST))
  LOG_PATH=${ROOT_PATH}/${log_dir}/alpa/search/${model_name}/${model_size}/
  RUNTIME_LOG_PATH=${ROOT_PATH}/${log_dir}/alpa/runtime/${model_name}/${model_size}/

  mkdir -p $LOG_PATH && mkdir -p $RUNTIME_LOG_PATH

  echo "--- Running experiment with $NUM_HOSTS hosts and $NUM_DEVICES_PER_HOST devices per host ---"
  echo [TIME] before running auto_${model_name}_${model_size}_${NUM_GPUS}_gpus: $(date '+%Y-%m-%d-%H-%M-%S') >> ${ROOT_PATH}/logs/alpa/full_log.log

  LOG_PATH=$LOG_PATH \
  RUNTIME_LOG_PATH=$RUNTIME_LOG_PATH \
  python3 -u benchmark_3d.py --suite ${model_name}.${suite} \
    --exp_name auto_${NUM_GPUS}_gpus \
    --num-hosts ${NUM_HOSTS} \
    --num-devices-per-host ${NUM_DEVICES_PER_HOST} \
    --disable-tqdm \
    |& tee -a ${LOG_PATH}auto_${model_name}_${model_size}_${NUM_GPUS}_gpus_${CURRENT_TIME}.log
  
  echo [TIME] after running auto_${model_name}_${model_size}_${NUM_GPUS}_gpus: $(date '+%Y-%m-%d-%H-%M-%S') >> ${ROOT_PATH}/logs/alpa/full_log.log
  sleep 0.1 # for ctrl+c to work
}

if [ "$exp_setting" == "small" ]; then
  run_experiment 1 4 ae_small 1_3B
elif [ "$exp_setting" == "large" ]; then
  run_experiment 4 8 ae_large 13B
  run_experiment 2 8 ae_large 6_7B
  run_experiment 1 8 ae_large 2_6B
  run_experiment 1 4 ae_large 1_3B
  run_experiment 1 1 ae_large 350M
elif [ "$exp_setting" == "scale" ]; then
  run_experiment 1 8 ae_scale_8 8layers
  run_experiment 1 8 ae_scale_16 16layers
  run_experiment 1 8 ae_scale_32 32layers
  run_experiment 1 8 ae_scale_64 64layers
  run_experiment 1 8 ae_scale_128 128layers
fi
