#! /bin/bash
ROOT_PATH=$(pwd)
exp_setting=$1
model_name=t5

cd $ROOT_PATH/runtime

if [ "$exp_setting" == "small" ]; then
    echo "[ERROR] this script is specially designed for large-scale running with docker, please run 'bash scripts/aceso_t5_execute.sh scale' instead"

elif [ "$exp_setting" == "large" ]; then
    ## Paths
    RESULT_PATH=${ROOT_PATH}/logs-large/aceso/

    ## 1node (4GPUs and 8GPUs)
    docker exec -i aceso-ae bash -c "cd $ROOT_PATH/runtime && bash scripts/run_${model_name}_1node.sh"

    ## 2nodes
    parallel-ssh -i -t 0 -h pssh-2workers.host "docker restart aceso-ae"
    parallel-ssh -i -t 0 -h pssh-2workers.host "docker exec -i aceso-ae bash -c 'cd $ROOT_PATH/runtime && bash scripts/run_${model_name}_2nodes_docker.sh'"

    ## 4nodes
    parallel-ssh -i -t 0 -h pssh-4workers.host "docker restart aceso-ae"
    parallel-ssh -i -t 0 -h pssh-4workers.host "docker exec -i aceso-ae bash -c 'cd $ROOT_PATH/runtime && bash scripts/run_${model_name}_4nodes_docker.sh'"

    python3 scripts/show_best_perf.py $model_name $RESULT_PATH 2>&1 | tee -a ${RESULT_PATH}full_log.log
fi
