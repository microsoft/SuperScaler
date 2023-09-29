import sys
import os 
import csv
from get_common import get_model_sizes, _format, get_best_config

exp_setting = sys.argv[1]
result_path = "results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

model_names = ["gpt", "t5", "resnet"]

if exp_setting == "small":
    log_path = "logs"
elif exp_setting == "large":
    log_path = "logs-large"

for model_name in model_names:
    all_time, all_memory = {}, {}
    model_sizes = get_model_sizes(model_name, exp_setting)
    for model_size in model_sizes:
        time_actual_and_predict = []
        mem_actual_and_predict = []

        best_config = get_best_config(log_path, "aceso", model_name, model_size)
        if best_config is not None:
            config_name = best_config
            if "stages" in config_name:
                num_stages = int(config_name.split("stages")[0].split("_")[-1])
            else:
                num_stages = 1
            runtime_result_path = os.path.join(log_path, "aceso", "runtime", model_name, model_size, "csv")
            evaluated_stage_index = 0
            if os.path.exists(runtime_result_path):
                files = [f for f in os.listdir(runtime_result_path) if os.path.isfile(os.path.join(runtime_result_path, f))]
                time_list = []
                memory_list = []
                for i in range(num_stages):
                    tmp_time = 0
                    tmp_mem = 0
                    for file_name in files:
                        if config_name in file_name and f"stage{i}" in file_name:
                            with open(os.path.join(runtime_result_path, file_name), 'r') as f:
                                csv_reader = csv.reader(f)
                                lines = list(csv_reader)
                                tmp_time = float(lines[1][1])
                                tmp_mem = float(lines[1][-1])
                    time_list.append(tmp_time)
                    memory_list.append(tmp_mem)
                actual_time = max(time_list)
                actual_mem = max(memory_list)
                max_mem = 0
                for i in range(num_stages):
                    if memory_list[i] > max_mem:
                        max_mem = memory_list[i]
                        evaluated_stage_index = i        
                time_actual_and_predict.append(actual_time)
                mem_actual_and_predict.append(actual_mem)

            ## Aceso prediction
            prediction_path = os.path.join(log_path, "aceso", "configs", model_name, model_size, "csv")
            if os.path.exists(prediction_path):
                files = [f for f in os.listdir(prediction_path) if os.path.isfile(os.path.join(prediction_path, f))]
                for file_name in files:
                    if config_name in file_name:
                        max_time = 0
                        with open(os.path.join(prediction_path, file_name), 'r') as f:
                            csv_reader = csv.reader(f)
                            next(csv_reader)
                            for row in csv_reader:
                                tmp_time = float(row[1])
                                if tmp_time > max_time:
                                    max_time = tmp_time
                                if row[0] == f"stage-{evaluated_stage_index}":
                                    mem_actual_and_predict += [float(row[2]), float(row[3]), float(row[4])] 
                        time_actual_and_predict.append(max_time)
        if len(time_actual_and_predict) > 0:
            all_time[model_size] = list(time_actual_and_predict)
        if len(mem_actual_and_predict) > 0:
            all_memory[model_size] = list(mem_actual_and_predict)

    if len(all_memory) > 0:
        print(f"-------- [{model_name}] Time Prediction (s) --------")
        print(f"Size\t Actual\t\t Predict")

        for model_size in model_sizes:
            if model_size in all_time:
                print(f"{model_size}\t{all_time[model_size][0]:.2f}\t {all_time[model_size][1]:.2f}")

        print(f"-------- [{model_name}] Memory Prediction (MB) --------")
        print(f"Size\t Actual\t\t Predict (normal + extra)")

        for model_size in model_sizes:
            if model_size in all_time:
                print(f"{model_size}\t{all_memory[model_size][0]:.2f}\t {all_memory[model_size][1]:.2f} ({all_memory[model_size][2]:.2f} + {all_memory[model_size][3]:.2f})")

        ## Save into csv files for plotting figures
        actual_time_list = []
        pred_time_list = []
        for model_size in model_sizes:
            if model_size in all_time:
                actual_time_list.append(all_time[model_size][0])
                pred_time_list.append(all_time[model_size][1])
            else:
                actual_time_list.append(0)
                pred_time_list.append(0)
        
        file_name = os.path.join(result_path, f"perf_model_time_{exp_setting}_{model_name}.csv")
        info_to_csv = [["size"] + model_sizes, ["actual"] + _format(actual_time_list), ["predict"] + _format(pred_time_list)]
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            for row in info_to_csv:
                writer.writerow(row)

        actual_mem_list = []
        pred_mem_norm_list = []
        pred_mem_extra_list = []
        for model_size in model_sizes:
            if model_size in all_memory:
                actual_mem_list.append(all_memory[model_size][0])
                pred_mem_norm_list.append(all_memory[model_size][2])
                pred_mem_extra_list.append(all_memory[model_size][3])
            else:
                actual_mem_list.append(0)
                pred_mem_norm_list.append(0)
                pred_mem_extra_list.append(0)
        file_name = os.path.join(result_path, f"perf_model_mem_{exp_setting}_{model_name}.csv")
        info_to_csv = [["size"] + model_sizes, ["actual"] + _format(actual_mem_list), ["predict (norm)"] + _format(pred_mem_norm_list) , ["predict (extra)"] + _format(pred_mem_extra_list)]
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            for row in info_to_csv:
                writer.writerow(row)