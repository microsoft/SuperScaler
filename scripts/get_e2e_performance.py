import sys
import os 
import csv
from get_common import get_thpt, get_alpa_thpt, get_model_sizes, get_normalized_thpt, _format

exp_setting = sys.argv[1]
result_path = "results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

if exp_setting == "small":
    log_path = "logs"
    model_names = ["gpt", "t5", "resnet"]
elif exp_setting == "large":
    log_path = "logs-large"
    model_names = ["gpt", "t5", "resnet"]
elif exp_setting == "scale":
    log_path = "logs-large"
    model_names = ["scale-layer"]

for model_name in model_names:
    all_time = {}
    all_thpt = {}
    model_sizes = get_model_sizes(model_name, exp_setting)

    megatron_thpt, alpa_thpt, aceso_thpt = [], [], []

    ## Print Results
    for model_size in model_sizes:
        tmp_result = ['-', '-', '-']
        tmp_time_result = ['-', '-', '-']
        add_result = False
        ## Megatron-LM
        max_thpt, min_time = get_thpt(log_path, "megatron", model_name, model_size)
        megatron_thpt.append(max_thpt)
        if max_thpt > 0:
            add_result = True
            tmp_result[0] = f"{max_thpt:.2f}"
            tmp_time_result[0] = f"{min_time:.2f}"
        ## Alpa
        max_thpt, min_time = get_alpa_thpt(log_path, model_name, model_size)
        alpa_thpt.append(max_thpt)
        if max_thpt > 0:
            add_result = True
            tmp_result[1] = f"{max_thpt:.2f}"
            tmp_time_result[1] = f"{min_time:.2f}"
        ## Aceso
        max_thpt, min_time = get_thpt(log_path, "aceso", model_name, model_size)
        aceso_thpt.append(max_thpt)
        if max_thpt > 0:
            add_result = True
            tmp_result[2] = f"{max_thpt:.2f}"
            tmp_time_result[2] = f"{min_time:.2f}"
        if add_result:
            all_thpt[model_size] = list(tmp_result)
            all_time[model_size] = list(tmp_time_result)

    # print(f"-------- [{model_name}] End-to-end Time (s) --------")
    # print(f"Size\t Megatron-LM\t Alpa\t Aceso")
    # for model_size in model_sizes:
    #     if model_size in all_time:
    #         print(f"{model_size}\t{all_time[model_size][0]}\t\t {all_time[model_size][1]}\t {all_time[model_size][2]}")
    if len(all_thpt) > 0:
        print(f"-------- [{model_name}] End-to-end Throughput (Samples/s) --------")
        print(f"Size\t Megatron-LM\t Alpa\t Aceso")

        for model_size in model_sizes:
            if model_size in all_thpt:
                print(f"{model_size}\t{all_thpt[model_size][0]}\t\t {all_thpt[model_size][1]}\t {all_thpt[model_size][2]}")

    ## Save into csv files for plotting figures
    file_name = os.path.join(result_path, f"e2e_thpt_{exp_setting}_{model_name}.csv")
    info_to_csv = [["size"] + model_sizes, ["megatron"] + _format(megatron_thpt), ["alpa"] + _format(alpa_thpt), ["aceso"] + _format(aceso_thpt)]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)

    norm_megatron_thpt, norm_alpa_thpt, norm_aceso_thpt = get_normalized_thpt(log_path, model_name, exp_setting)
    file_name = os.path.join(result_path, f"e2e_norm_thpt_{exp_setting}_{model_name}.csv")
    info_to_csv = [["size"] + model_sizes, ["megatron"] + _format(norm_megatron_thpt), ["alpa"] + _format(norm_alpa_thpt), ["aceso"] + _format(norm_aceso_thpt)]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)